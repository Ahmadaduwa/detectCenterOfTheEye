import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset_loader import EyeDataset
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

def build_model(pretrained=True):
    net = models.resnet18(pretrained=pretrained)
    in_f = net.fc.in_features
    # predict normalized x,y in [0,1]
    net.fc = nn.Sequential(
        nn.Linear(in_f, 2),
        nn.Sigmoid()
    )
    return net

def mae_pixels(pred_norm, target_norm, orig_sizes):
    preds = pred_norm.detach().cpu().numpy()
    tars = target_norm.detach().cpu().numpy()
    sum_mae = 0.0
    n = preds.shape[0]
    for i in range(n):
        # support two collated formats:
        # 1) list/tuple of pairs: [(w,h), (w,h), ...]
        # 2) tuple of two lists: ( [w1,w2,...], [h1,h2,...] )  <-- default collate when orig_size is a tuple
        if isinstance(orig_sizes, (tuple, list)) and len(orig_sizes) == 2 and not (
            isinstance(orig_sizes[0], (tuple, list))
        ):
            # orig_sizes == (list_w, list_h)
            w = int(orig_sizes[0][i])
            h = int(orig_sizes[1][i])
        else:
            entry = orig_sizes[i]
            # entry might be tuple, list, numpy array or torch tensor
            try:
                w = int(entry[0])
                h = int(entry[1])
            except Exception:
                # fallback: try to convert whole entry to ints
                try:
                    vals = list(entry)
                    w = int(vals[0]); h = int(vals[1])
                except Exception:
                    # as last resort assume 1 to avoid crash
                    w = 1; h = 1

        px = preds[i, 0] * w
        py = preds[i, 1] * h
        tx = tars[i, 0] * w
        ty = tars[i, 1] * h
        sum_mae += (abs(px - tx) + abs(py - ty)) / 2.0
    return sum_mae / n

def composed_transform(sample):
    # moved to module-level so DataLoader workers can pickle it
    sample = RandomFlip(p=0.5)(sample)
    sample = ToTensor()(sample)
    return sample

def train():
    images_dir = r"d:\Y3COE\eye\Project\eye-detection-project\dataset\Image"
    ann_dir = r"d:\Y3COE\eye\Project\eye-detection-project\dataset\Annotation"
    # use larger input size to improve accuracy
    ds = EyeDataset(images_dir, ann_dir, transform=None, img_size=(320,320), normalize_coords=True)

    ds.transform = composed_transform

    val_percent = 0.1
    n_val = int(len(ds)*val_percent)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.backends.cudnn.benchmark = True

    model = build_model(pretrained=True).to(device)

    criterion = nn.L1Loss()  # MAE on normalized coords
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    best_val_mae = 1e9
    for epoch in range(1, 101):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train E{epoch}"):
            imgs = batch['image'].to(device, non_blocking=True)
            tars = batch['target'].to(device, non_blocking=True)
            origs = batch['orig_size']

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                preds = model(imgs)
                loss = criterion(preds, tars)
            scaler.scale(loss).backward()
            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        running_val = 0.0
        val_mae_pixel = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device, non_blocking=True)
                tars = batch['target'].to(device, non_blocking=True)
                origs = batch['orig_size']
                with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                    preds = model(imgs)
                    running_val += criterion(preds, tars).item() * imgs.size(0)
                    val_mae_pixel += mae_pixels(preds, tars, origs) * imgs.size(0)
        val_loss = running_val / len(val_loader.dataset)
        val_mae_pixel = val_mae_pixel / len(val_loader.dataset)

        scheduler.step(val_loss)
        print(f"Epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_MAE_px={val_mae_pixel:.3f}")
        if val_mae_pixel < best_val_mae:
            best_val_mae = val_mae_pixel
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_mae_px': val_mae_pixel}, "best_cnn.pt")
        if best_val_mae <= 4.5:
            print("Target MAE reached, stopping.")
            break

if __name__ == "__main__":
    train()
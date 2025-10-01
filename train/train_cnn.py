import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from dataset import EyeXYDataset

# Simple CNN regressor
class SimpleRegCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.features(x))

def to_tensors(batch):
    imgs = np.stack([b["image"] for b in batch], axis=0)  # B,H,W,C
    imgs = np.transpose(imgs, (0,3,1,2))  # B,C,H,W
    targets = np.stack([b["target"] for b in batch], axis=0)
    return torch.from_numpy(imgs).float(), torch.from_numpy(targets).float()

def collate_fn(batch):
    return to_tensors(batch)

class InMemoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def train():
    # check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    
    device = torch.device("cuda")
    print("Using device:", device)
    print("GPU:", torch.cuda.get_device_name(0), "count:", torch.cuda.device_count())
    cudnn.benchmark = True

    # load dataset
    img_dir = r"c:\Users\Admin ST\TLek\Eyes\Image144x144"
    ann_dir = r"c:\Users\Admin ST\TLek\Eyes\Annotation144x144"
    ds = EyeXYDataset(img_dir, ann_dir, img_size=(144,144))
    n = len(ds)
    if n == 0:
        raise RuntimeError("No data found.")
    
    val_sz = max(1, int(0.15 * n))
    train_sz = n - val_sz
    train_ds, val_ds = random_split(ds, [train_sz, val_sz])

    # preload dataset to RAM
    preload = os.environ.get("PRELOAD_DATA", "0") == "1"
    if preload:
        print("Preloading dataset into memory...")
        def subset_to_samples(subset):
            base_ds = subset.dataset
            inds = getattr(subset, "indices", None)
            if inds is None:
                return [base_ds[i] for i in range(len(base_ds))]
            return [base_ds[i] for i in inds]
        train_ds = InMemoryDataset(subset_to_samples(train_ds))
        val_ds = InMemoryDataset(subset_to_samples(val_ds))
        print("Data preloaded.")

    # auto-detect batch size
    def find_safe_batch(init_bs=32, img_shape=(3,144,144)):
        bs = init_bs
        model = SimpleRegCNN().to(device)
        applied_channels_last = True
        try:
            model = model.to(device=device, memory_format=torch.channels_last)
        except:
            applied_channels_last = False
        while bs >= 1:
            try:
                with torch.no_grad():
                    fake = torch.randn((bs, *img_shape), device=device)
                    if applied_channels_last:
                        fake = fake.to(memory_format=torch.channels_last)
                    _ = model(fake)
                torch.cuda.empty_cache()
                return bs
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    bs = bs // 2
                else:
                    raise
        return 1

    env_bs = os.environ.get("BATCH_SIZE")
    suggested = int(env_bs) if env_bs and env_bs.isdigit() else 64
    batch_size = find_safe_batch(init_bs=suggested)
    print(f"Auto-selected batch_size={batch_size}")

    # DataLoader
    num_workers = max(1, os.cpu_count() - 1)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size//2), shuffle=False,
        collate_fn=collate_fn, num_workers=max(1,num_workers//2), pin_memory=True
    )

    # build model
    model = SimpleRegCNN().to(device)
    try:
        model = model.to(memory_format=torch.channels_last)
    except:
        pass

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    patience = 6
    wait = 0
    max_epochs = 100

    for epoch in range(1, max_epochs+1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for bidx, (imgs, targets) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            try:
                imgs = imgs.to(memory_format=torch.channels_last)
            except:
                pass

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                preds = model(imgs)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            sys.stdout.write(f"\rEpoch {epoch}/{max_epochs} TRAIN {bidx}/{len(train_loader)}")
            sys.stdout.flush()
        train_mae = train_loss / len(train_loader.dataset)
        sys.stdout.write("  done\n")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vbidx, (imgs, targets) in enumerate(val_loader, start=1):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                try:
                    imgs = imgs.to(memory_format=torch.channels_last)
                except:
                    pass
                with torch.amp.autocast(device_type='cuda'):
                    preds = model(imgs)
                val_loss += torch.mean(torch.abs(preds - targets)).item() * imgs.size(0)
                sys.stdout.write(f"\rEpoch {epoch}/{max_epochs} VAL {vbidx}/{len(val_loader)}")
                sys.stdout.flush()
        val_mae = val_loss / len(val_loader.dataset)
        sys.stdout.write("  done\n")

        print(f"\nEpoch {epoch} summary: train_mae={train_mae:.4f} val_mae={val_mae:.4f} time={time.time()-epoch_start:.1f}s")
        print("GPU mem allocated:", torch.cuda.memory_allocated()//1024**2, "MB")

        # save best model
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "best_cnn_new.pt"))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

        if best_val < 5.0:  # optional target
            print("Target CNN MAE reached:", best_val)
            break

if __name__ == "__main__":
    train()

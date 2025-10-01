import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from dataset_loader import EyeDataset
from torchvision.transforms import ToTensor
from train_cnn import build_model  # import ฟังก์ชันสร้างโมเดล

# ----------------------------
# Transform สำหรับ EyeDataset
# ----------------------------
def composed_transform(sample):
    # sample เป็น dict {'image': PIL/ndarray, 'target': [x,y]}
    img = sample['image']
    img_tensor = ToTensor()(img)  # HWC/ndarray -> CHW tensor
    sample['image'] = img_tensor
    return sample

# ----------------------------
# คำนวณ MAE pixel
# ----------------------------
def mae_pixels(preds, targets):
    # preds, targets: tensor shape [batch, 2], พิกัด 0-144
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    return np.mean(np.abs(preds_np - targets_np))

# ----------------------------
# ทดสอบโมเดล
# ----------------------------
def test_model_and_plot():
    images_dir = "./data/Right/Image144x144"
    ann_dir = "./data/Right/Annotation144x144"

    # โหลด dataset สำหรับ 144x144
    ds = EyeDataset(images_dir, ann_dir, transform=None, img_size=(144,144), normalize_coords=False)
    ds.transform = composed_transform
    test_loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # โหลดโมเดล
    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load("./train3/best_cnn.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    criterion = nn.L1Loss()
    total_loss = 0.0
    total_mae_px = 0.0

    all_imgs, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            targets = batch['target'].to(device)

            preds = model(imgs)

            # Loss
            total_loss += criterion(preds, targets).item() * imgs.size(0)
            total_mae_px += mae_pixels(preds, targets) * imgs.size(0)

            # เก็บไว้ plot
            all_imgs.append(imgs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / len(test_loader.dataset)
    avg_mae_px = total_mae_px / len(test_loader.dataset)
    print(f"Test results: norm_L1_loss={avg_loss:.6f}, pixel_MAE={avg_mae_px:.3f}")

    # รวม tensor
    all_imgs = torch.cat(all_imgs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # ----------------------------
    # Plot 10 ภาพสุ่ม
    # ----------------------------
    idxs = random.sample(range(len(all_imgs)), min(10, len(all_imgs)))
    fig, axes = plt.subplots(2, 5, figsize=(15,6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        idx = idxs[i]
        img = np.transpose(all_imgs[idx], (1,2,0))  # CHW -> HWC
        img = np.clip(img, 0, 1)

        pred = all_preds[idx]
        target = all_targets[idx]

        ax.imshow(img)
        ax.scatter(target[0], target[1], c='lime', marker='o', s=40, label='GT')
        ax.scatter(pred[0], pred[1], c='red', marker='x', s=40, label='Pred')
        ax.axis('off')
        ax.set_title(f"#{idx}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_model_and_plot()

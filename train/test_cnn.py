import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from dataset import EyeXYDataset
from train_cnn import SimpleRegCNN, collate_fn   # import จากไฟล์ train


def mae_pixels(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    diffs = np.abs(preds - targets)
    mae = np.mean(diffs)
    return mae


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # โหลด dataset test
    img_dir = "./data/Right/Image144x144"
    ann_dir = "./data/Right/Annotation144x144"
    ds = EyeXYDataset(img_dir, ann_dir, img_size=(144, 144))

    test_loader = DataLoader(
        ds, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    # โหลดโมเดล
    model = SimpleRegCNN().to(device)
    model.load_state_dict(torch.load("./train/best_cnn_new.pt", map_location=device))
    model.eval()

    criterion = nn.L1Loss()
    total_loss = 0.0
    total_mae_px = 0.0

    all_preds, all_targets, all_imgs = [], [], []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model(imgs)

            # เก็บไว้สำหรับ plot
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_imgs.append(imgs.cpu())

            # loss
            total_loss += criterion(preds, targets).item() * imgs.size(0)
            total_mae_px += mae_pixels(preds, targets) * imgs.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    avg_mae_px = total_mae_px / len(test_loader.dataset)

    print(f"Test results: norm_L1_loss={avg_loss:.4f}, pixel_MAE={avg_mae_px:.3f}")

    # รวม tensor
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_imgs = torch.cat(all_imgs, dim=0).numpy()

    # สุ่มเลือก 10 ภาพมา plot
    idxs = random.sample(range(len(all_imgs)), 10)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        idx = idxs[i]
        img = np.transpose(all_imgs[idx], (1, 2, 0))  # CHW -> HWC
        img = np.clip(img, 0, 1)

        pred = all_preds[idx]
        target = all_targets[idx]

        ax.imshow(img)
        ax.scatter(target[0], target[1], c="lime", marker="o", s=40, label="GT")
        ax.scatter(pred[0], pred[1], c="red", marker="x", s=40, label="Pred")
        ax.set_title(f"#{idx}")
        ax.axis("off")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_model()

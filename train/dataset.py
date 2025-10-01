import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset

class EyeXYDataset(Dataset):
    def __init__(self, img_dir, ann_dir, img_size=(144,144)):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.ann_dir = ann_dir
        self.img_size = img_size

    def _parse_annotation(self, ann_path):
        if not os.path.exists(ann_path):
            return None
        with open(ann_path, "r", encoding="utf-8") as f:
            toks = []
            for line in f:
                toks += line.strip().split()
        nums = [float(t) for t in toks if self._is_number(t)]
        if len(nums) >= 2:
            # take first two numbers as (x, y) in pixel coords
            return np.asarray([nums[0], nums[1]], dtype=np.float32)
        return None

    def _is_number(self, s):
        try:
            float(s); return True
        except:
            return False

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        base = os.path.splitext(os.path.basename(p))[0]
        ann_path = os.path.join(self.ann_dir, base + ".txt")
        target = self._parse_annotation(ann_path)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image {p}")
        h0, w0 = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        if target is None:
            # fallback if annotation missing
            target = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # if original image size not equal to img_size, scale coordinates
            tx = target[0] * (self.img_size[1] / w0)
            ty = target[1] * (self.img_size[0] / h0)
            target = np.array([tx, ty], dtype=np.float32)
        # return image as HWC float32 and target as 2-float array
        return {"image": img, "target": target}
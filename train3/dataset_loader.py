import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset

class EyeDataset(Dataset):
    def __init__(self, images_dir, ann_dir, transform=None, img_size=(224,224), normalize_coords=True):
        self.images_dir = images_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.img_size = img_size
        self.normalize_coords = normalize_coords

        # collect image files (jpg/png)
        exts = ('*.jpg','*.jpeg','*.png','*.bmp')
        files = []
        for e in exts:
            files += glob.glob(os.path.join(images_dir, e))
        files = sorted(files)
        self.items = []
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            # try matching annotation file
            ann_candidates = glob.glob(os.path.join(ann_dir, base + '*.txt'))
            if len(ann_candidates) == 0:
                continue
            ann = ann_candidates[0]
            self.items.append((f, ann))

    def __len__(self):
        return len(self.items)

    def _read_ann(self, ann_path, w, h):
        # Read floats from annotation file (flexible parsing)
        with open(ann_path, 'r', encoding='utf-8', errors='ignore') as fh:
            s = fh.read().replace(',', ' ').split()
            vals = []
            for token in s:
                try:
                    vals.append(float(token))
                except:
                    pass
            if len(vals) >= 2:
                x, y = vals[0], vals[1]
            else:
                # fallback center
                x, y = w/2.0, h/2.0
        if self.normalize_coords:
            return x / w, y / h
        return x, y

    def __getitem__(self, idx):
        img_path, ann_path = self.items[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR->RGB
        h, w = img.shape[:2]
        target = self._read_ann(ann_path, w, h)  # normalized
        # resize
        img_resized = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        sample = {'image': img_resized, 'target': np.array(target, dtype=np.float32), 'orig_size': (w,h)}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
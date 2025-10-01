import os
import joblib
from sklearn.metrics import mean_absolute_error
import numpy as np
import glob
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import random

# path ของโมเดล
RF_MODEL_PATH = r"d:\Coding\Python\Project\eyeCenter\train\rf_model.joblib"

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def load_data(img_dir, ann_dir, size=(64,64)):
    X, Y, imgs_raw = [], [], []
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    for p in imgs:
        base = os.path.splitext(os.path.basename(p))[0]
        annp = os.path.join(ann_dir, base + ".txt")
        if not os.path.exists(annp):
            continue
        with open(annp, "r", encoding="utf-8") as f:
            toks = []
            for line in f:
                toks += line.strip().split()
        nums = [float(t) for t in toks if is_number(t)]
        if len(nums) < 2:
            continue
        y = [nums[0], nums[1]]
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        imgs_raw.append(im)  # เก็บภาพจริงเพื่อ plot
        imr = cv2.resize(im, size)
        feats = hog(imr, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        X.append(feats)
        Y.append(y)
    return np.asarray(X), np.asarray(Y), imgs_raw

def test_rf_plot(img_dir, ann_dir, plot_num=10):
    X, Y, imgs_raw = load_data(img_dir, ann_dir, size=(64,64))
    if len(X) == 0:
        raise RuntimeError("No data found")

    if os.path.exists(RF_MODEL_PATH):
        print("Testing RandomForest...")
        rf = joblib.load(RF_MODEL_PATH)
        y_pred = rf.predict(X)
        mae = mean_absolute_error(Y, y_pred)
        print("RF pixel MAE:", mae)
    else:
        print("RF model not found at", RF_MODEL_PATH)
        return

    # สุ่มเลือก plot_num ภาพ
    idxs = random.sample(range(len(imgs_raw)), min(plot_num, len(imgs_raw)))

    fig, axes = plt.subplots(1, len(idxs), figsize=(15,3))
    if len(idxs) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        idx = idxs[i]
        img = imgs_raw[idx]
        ax.imshow(img, cmap='gray')
        ax.scatter(Y[idx][0], Y[idx][1], c="lime", marker="o", s=40, label="GT")
        ax.scatter(y_pred[idx][0], y_pred[idx][1], c="red", marker="x", s=40, label="Pred")
        ax.set_title(f"#{idx}")
        ax.axis("off")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_dir = r"d:\Coding\Python\Project\eyeCenter\data\Right\Image144x144"
    ann_dir = r"d:\Coding\Python\Project\eyeCenter\data\Right\Annotation144x144"
    test_rf_plot(img_dir, ann_dir)

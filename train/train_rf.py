import os
import glob
import cv2
import numpy as np
import sys
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Optional additional models
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None

from skimage.feature import hog

# --- จำกัด CPU ใช้ ~80% ---
num_threads = max(1, int(os.cpu_count() * 0.8))
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
if torch:
    torch.set_num_threads(num_threads)

print(f"CPU threads limited to {num_threads}")

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def _print_progress(prefix, current, total):
    pct = current / total * 100 if total>0 else 0.0
    sys.stdout.write(f"\r{prefix} {current}/{total} ({pct:.1f}%)")
    sys.stdout.flush()

def load_data(img_dir, ann_dir, size=(64,64)):
    X, Y = [], []
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    total = len(imgs)
    for i, p in enumerate(imgs, start=1):
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
        imr = cv2.resize(im, size)
        feats = hog(imr, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        X.append(feats)
        Y.append(y)
        if i % max(1, total//50) == 0 or i==total:
            _print_progress("Loading images/features", i, total)
    sys.stdout.write("\n")
    return np.asarray(X), np.asarray(Y)

# --- PyTorch simple MLP ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_pytorch_mlp(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(X_train.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).float().to(device)

    for epoch in range(1, epochs+1):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0
        for i in range(0, X_train_tensor.size(0), batch_size):
            idx = permutation[i:i+batch_size]
            inputs, targets = X_train_tensor[idx], y_train_tensor[idx]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        train_mae = epoch_loss / X_train_tensor.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_mae = torch.mean(torch.abs(val_outputs - y_val_tensor)).item()
        if epoch % 10 == 0 or epoch==1:
            print(f"Epoch {epoch}/{epochs} train_mae={train_mae:.4f} val_mae={val_mae:.4f}")
    torch.save(model.state_dict(), "mlp_model.pt")
    return val_mae

def train_all_models():
    img_dir = r"c:\Users\Admin ST\TLek\Eyes\Image144x144"
    ann_dir = r"c:\Users\Admin ST\TLek\Eyes\Annotation144x144"
    X, Y = load_data(img_dir, ann_dir, size=(64,64))
    if len(X) == 0:
        raise RuntimeError("No data found")

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)

    results = {}

    # --- RandomForest ---
    print("\nTraining RandomForest...")
    rf = RandomForestRegressor(n_jobs=num_threads, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print("RF val MAE:", mae)
    joblib.dump(rf, "rf_model.joblib")
    results["RandomForest"] = mae

    # --- XGBoost ---
    if xgb:
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBRegressor(n_jobs=num_threads, random_state=42, tree_method="gpu_hist" if torch and torch.cuda.is_available() else "hist")
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print("XGB val MAE:", mae)
        joblib.dump(xgb_model, "xgb_model.joblib")
        results["XGBoost"] = mae

    # --- LightGBM ---
    if lgb:
        print("\nTraining LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_jobs=num_threads, random_state=42, device="gpu" if torch and torch.cuda.is_available() else "cpu")
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print("LGB val MAE:", mae)
        joblib.dump(lgb_model, "lgb_model.joblib")
        results["LightGBM"] = mae

    # --- PyTorch MLP ---
    if torch:
        print("\nTraining PyTorch MLP...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mae = train_pytorch_mlp(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, device=device)
        print("MLP val MAE:", mae)
        results["MLP"] = mae

    print("\nAll model results:", results)
    return results

if __name__ == "__main__":
    train_all_models()

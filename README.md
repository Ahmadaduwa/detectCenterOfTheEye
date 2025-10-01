# Eye Center Detection

Eye/pupil center detection with classic ML (HOG + RandomForest) and deep learning (PyTorch CNN and TensorFlow/Keras transfer learning), plus real-time OpenCV + MediaPipe demos.

This repo includes:
- Training pipelines under `train`, `train3`, `train4`
- Traditional ML feature pipeline and benchmarking in `test\opencv_rf.py`
- CNN training in `train\train_cnn.py` and variants in `train3`, `train4`
- Real-time demos: `test\opencv_rf.py` (RF/HOG) and `test\opencv_cnn.py` (CNN)

## Directory Structure

```
eyeCenter/
  data/                          # Your datasets (images + annotations) - not included
  train/
    dataset.py                   # Dataset loader (EyeXYDataset)
    train_cnn.py                 # PyTorch CNN training
    train_rf.py                  # Classical ML training (RF, XGB, LGB, MLP)
    test_cnn.py, test_rf.py      # Evaluation helpers
    requirements.txt             # Minimal training requirements
  train2/, train3/, train4/      # Additional experiments, models, logs
  test/
    opencv_rf.py                 # Real-time RF/HOG + MediaPipe demo
    opencv_cnn.py                # Real-time CNN + MediaPipe demo
  README.md
```

## Requirements

Python 3.9+ recommended.

Base training requirements (see `train/requirements.txt`):

```
torch
torchvision
scikit-learn
opencv-python
scikit-image
joblib
numpy
```

Additional (optional) packages depending on what you run:
- MediaPipe (for demos): `mediapipe`
- TensorFlow/Keras (for `test/opencv_cnn.py` and models in `train4/models_tl`): `tensorflow`
- XGBoost, LightGBM (optional in `train/train_rf.py`): `xgboost`, `lightgbm`

### Create environment

```bash
python -m venv .venv
.\n+venv\Scripts\activate  # on Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r train/requirements.txt
# Optional extras
pip install mediapipe tensorflow xgboost lightgbm
```

## Data Preparation

Training scripts expect image folders and matching annotation folders with the same base filenames. Update paths as needed.

- CNN trainer (`train/train_cnn.py`) uses:
  - `img_dir = C:\Users\Admin ST\TLek\Eyes\Image144x144`
  - `ann_dir = C:\Users\Admin ST\TLek\Eyes\Annotation144x144`

- RF/HOG pipeline (`train/train_rf.py`) also references the same pattern.

If your data lives elsewhere, edit those variables or set your own paths before running.

Annotation format: text files containing at least two numbers per image line corresponding to eye center coordinates `[x y]`. Values can be absolute pixels or normalized to [0,1] (the code handles both cases).

## Training

### Train CNN (PyTorch)

`train/train_cnn.py` trains a simple CNN regressor to predict center `(x, y)`.

Notes:
- Requires CUDA GPU; script will error if CUDA is not available.
- Automatically selects a safe batch size based on available memory.
- Saves best weights to `best_cnn_new.pt` in the working directory.

Run:

```bash
python train/train_cnn.py
```

Environment variables (optional):
- `BATCH_SIZE`: initial guess for batch size (auto-tuned down if OOM)
- `PRELOAD_DATA=1`: preload train/val subsets into RAM to speed up I/O

### Train classical models (RF / XGB / LGB / MLP)

`train/train_rf.py` extracts HOG features and trains several regressors. It saves models like `rf_model.joblib` in the working directory.

Run:

```bash
python train/train_rf.py
```

Optional dependencies are used if installed:
- XGBoost: saves `xgb_model.joblib`
- LightGBM: saves `lgb_model.joblib`
- PyTorch MLP: saves `mlp_model.pt`

## Real-time Demos

Two optimized OpenCV + MediaPipe demos are provided. Both require a working webcam and MediaPipe installed.

### RF/HOG demo

File: `test/opencv_rf.py`

Edit `RF_MODEL_PATH` near the top if needed. By default it points to:

```
D:\Coding\Python\Project\eyeCenter\train4\models_gridsearch_with_radius\RandomForest_best.joblib
```

Run:

```bash
python test/opencv_rf.py
```

Keys:
- `q`: quit
- `s`: save snapshot PNG

### CNN demo (Keras)

File: `test/opencv_cnn.py`

Edit `CNN_MODEL_PATH` to point to a trained model, default:

```
D:\Coding\Python\Project\eyeCenter\train4\models_tl\tl_trained_final.h5
```

Run:

```bash
python test/opencv_cnn.py
```

Keys:
- `q`: quit
- `s`: save snapshot PNG

## Tips & Troubleshooting

- If MediaPipe cannot access the camera on Windows, try using `cv2.CAP_DSHOW` as in the scripts, or run from a terminal with camera permissions.
- If CUDA is not available but you want to run the CNN trainer, switch to CPU in the code or use the classical RF pipeline which is CPU-friendly.
- Make sure your image size and annotation normalization match the expectations in your chosen script (`HOG_SIZE` and `MODEL_SIZE`).

## License

Specify your license here (e.g., MIT). If uncertain, add an SPDX header or include a `LICENSE` file.

## Citation

If you use this project in academic work, please cite this repository.



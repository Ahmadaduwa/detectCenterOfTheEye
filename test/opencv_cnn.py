# eye_center_cnn_realtime.py
# Real-time eye center detection using CNN (MobileNetV2) with OpenCV
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import threading
import queue

# ---------------- Configuration ----------------
CNN_MODEL_PATH = r"D:\Coding\Python\Project\eyeCenter\train4\models_tl\tl_trained_final.h5"
CAM_INDEX = 0

# Model input size (must match training)
MODEL_SIZE = (144, 144)  # width, height
ROI_PAD = 0.35
SMOOTH_ALPHA = 0.45

# Visualization
DOT_COLOR = (0, 0, 255)
DOT_RADIUS = 3
CIRCLE_COLOR = (0, 255, 255)
CIRCLE_THICK = 1

# Speed optimizations
DETECT_DOWNSCALE = 0.4
PROCESS_EVERY_N = 2
MP_MIN_DETECTION = 0.3
MP_MIN_TRACKING = 0.3

# Camera settings
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# Queue settings
FRAME_QUEUE_MAXSIZE = 2
PRED_QUEUE_MAXSIZE = 2

# Preprocessing
USE_CLAHE = True

# ---------------- Load CNN Model ----------------
if not os.path.exists(CNN_MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {CNN_MODEL_PATH}")

print("Loading CNN model...")
cnn_model = keras.models.load_model(CNN_MODEL_PATH)
print(f"Model loaded: {CNN_MODEL_PATH}")
print(f"Input shape: {cnn_model.input_shape}")
print(f"Output shape: {cnn_model.output_shape}")

# ---------------- MediaPipe Setup ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=MP_MIN_DETECTION,
    min_tracking_confidence=MP_MIN_TRACKING
)

# Eye landmarks (same as your training)
LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398]

# ---------------- Preprocessing Functions ----------------
if USE_CLAHE:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_eye_patch(patch_bgr, target_size=MODEL_SIZE):
    """Preprocess eye patch for CNN (same as training)"""
    # Convert to grayscale
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    if USE_CLAHE:
        gray = clahe.apply(gray)
    
    # Resize to model input size
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert to 3-channel (for MobileNetV2)
    rgb = np.stack([normalized] * 3, axis=-1)
    
    # Add batch dimension
    batched = np.expand_dims(rgb, axis=0)
    
    return batched

def predict_eye_center(model, patch_bgr, roi_info):
    """Predict eye center using CNN"""
    x1, y1, rw, rh = roi_info
    
    # Preprocess
    input_tensor = preprocess_eye_patch(patch_bgr, MODEL_SIZE)
    
    # Predict
    pred = model.predict(input_tensor, verbose=0)[0]  # [cx, cy, radius]
    
    # Map predictions back to original frame coordinates
    cx_model = pred[0]
    cy_model = pred[1]
    r_model = pred[2]
    
    # Convert from model space to ROI space
    cx_roi = (cx_model / MODEL_SIZE[0]) * rw
    cy_roi = (cy_model / MODEL_SIZE[1]) * rh
    r_roi = (r_model / ((MODEL_SIZE[0] + MODEL_SIZE[1]) / 2)) * ((rw + rh) / 2)
    
    # Map to frame coordinates
    cx_frame = x1 + cx_roi
    cy_frame = y1 + cy_roi
    
    return cx_frame, cy_frame, r_roi

# ---------------- Smoothing Buffer ----------------
class SmoothingBuffer:
    """Exponential moving average for smooth predictions"""
    def __init__(self, alpha=0.45):
        self.alpha = alpha
        self.left = None
        self.right = None
    
    def update(self, preds):
        """preds: list of (side, x, y, r)"""
        left_new = None
        right_new = None
        
        for side, x, y, r in preds:
            if side == "left":
                left_new = (x, y, r)
            else:
                right_new = (x, y, r)
        
        # Smooth
        if left_new:
            if self.left is None:
                self.left = left_new
            else:
                self.left = tuple(self.alpha * n + (1-self.alpha) * o 
                                for n, o in zip(left_new, self.left))
        
        if right_new:
            if self.right is None:
                self.right = right_new
            else:
                self.right = tuple(self.alpha * n + (1-self.alpha) * o 
                                 for n, o in zip(right_new, self.right))
        
        result = []
        if self.left:
            result.append(("left", *self.left))
        if self.right:
            result.append(("right", *self.right))
        return result

# ---------------- Shared State ----------------
frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
pred_q = queue.Queue(maxsize=PRED_QUEUE_MAXSIZE)
run_event = threading.Event()
run_event.set()

stats_lock = threading.Lock()
proc_fps = 0.0
cap_fps = 0.0

smoother = SmoothingBuffer(alpha=SMOOTH_ALPHA)

# ---------------- Capture Thread ----------------
def capture_thread_fn(cam_index, frame_q, run_event):
    global cap_fps
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_index}")
        run_event.clear()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(cap.get(cv2.CAP_PROP_FPS))}fps")
    
    last_time = time.time()
    while run_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        now = time.time()
        with stats_lock:
            cap_fps = 0.9 * cap_fps + 0.1 / max(1e-6, now - last_time)
        last_time = now
        
        # Drop old frames
        while not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                break
        
        try:
            frame_q.put_nowait(frame)
        except queue.Full:
            pass
    
    cap.release()

# ---------------- Worker Thread (CNN Processing) ----------------
def worker_thread_fn(frame_q, pred_q, run_event):
    global proc_fps
    frame_idx = 0
    last_proc_time = time.time()
    
    small_size = (int(CAM_WIDTH * DETECT_DOWNSCALE), int(CAM_HEIGHT * DETECT_DOWNSCALE))
    
    while run_event.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        
        frame_idx += 1
        if PROCESS_EVERY_N > 1 and (frame_idx % PROCESS_EVERY_N != 0):
            continue
        
        t0 = time.time()
        frame_h, frame_w = frame.shape[:2]
        
        # Detect face/eyes with MediaPipe
        small = cv2.resize(frame, small_size, interpolation=cv2.INTER_LINEAR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_small)
        
        local_preds = []
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # Process each eye
            for idxs in (LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS):
                pts = np.array([(lm[i].x * frame_w, lm[i].y * frame_h) for i in idxs])
                x_min, y_min = pts.min(axis=0).astype(int)
                x_max, y_max = pts.max(axis=0).astype(int)
                
                w = max(8, x_max - x_min)
                h = max(8, y_max - y_min)
                p = int(max(w, h) * ROI_PAD)
                
                x1 = max(0, x_min - p)
                y1 = max(0, y_min - p)
                x2 = min(frame_w, x_max + p)
                y2 = min(frame_h, y_max + p)
                
                rw = x2 - x1
                rh = y2 - y1
                if rw <= 4 or rh <= 4:
                    continue
                
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                
                try:
                    # Use CNN to predict eye center
                    cx, cy, r = predict_eye_center(cnn_model, patch, (x1, y1, rw, rh))
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue
                
                # Determine side
                eye_center_x = (x_min + x_max) * 0.5
                side = "right" if eye_center_x < (frame_w * 0.5) else "left"
                
                local_preds.append((side, float(cx), float(cy), float(r)))
        
        # Put predictions in queue
        while not pred_q.empty():
            try:
                pred_q.get_nowait()
            except queue.Empty:
                break
        
        try:
            pred_q.put_nowait(local_preds)
        except queue.Full:
            pass
        
        # Update processing FPS
        t1 = time.time()
        with stats_lock:
            proc_fps = 0.9 * proc_fps + 0.1 / max(1e-6, t1 - last_proc_time)
        last_proc_time = t1

# ---------------- Main (UI) ----------------
print("Starting threads...")
cap_thread = threading.Thread(target=capture_thread_fn, args=(CAM_INDEX, frame_q, run_event), daemon=True)
worker_thread = threading.Thread(target=worker_thread_fn, args=(frame_q, pred_q, run_event), daemon=True)
cap_thread.start()
worker_thread.start()

prev_time = time.time()
display_fps = 0.0
preds_snapshot = []

print("Starting display loop... Press 'q' to quit, 's' to save snapshot")

try:
    while run_event.is_set():
        # Get latest frame
        frame = None
        try:
            while True:
                frame = frame_q.get_nowait()
        except queue.Empty:
            pass
        
        if frame is None:
            time.sleep(0.001)
            continue
        
        # Get latest predictions
        try:
            while True:
                preds_snapshot = pred_q.get_nowait()
        except queue.Empty:
            pass
        
        # Apply smoothing
        preds_smooth = smoother.update(preds_snapshot)
        
        # Draw predictions
        for (side, sx, sy, sr) in preds_smooth:
            cx_i = int(round(sx))
            cy_i = int(round(sy))
            r_i = int(round(sr))
            
            # Draw center point
            cv2.circle(frame, (cx_i, cy_i), DOT_RADIUS, DOT_COLOR, -1)
            
            # Draw iris circle
            if r_i > 0:
                cv2.circle(frame, (cx_i, cy_i), r_i, CIRCLE_COLOR, CIRCLE_THICK)
            
            # Label
            cv2.putText(frame, side, (cx_i + 6, cy_i - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        
        # Calculate display FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        display_fps = 0.9 * display_fps + 0.1 / max(1e-6, dt)
        
        # Get stats
        with stats_lock:
            pf = proc_fps
            cf = cap_fps
        
        # Draw FPS info
        cv2.putText(frame, f"Display:{display_fps:.1f} Process:{pf:.1f} Capture:{cf:.1f} [CNN]", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Eye Center (CNN)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            run_event.clear()
            break
        elif key == ord('s'):
            fname = f"snapshot_cnn_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")

finally:
    print("Shutting down...")
    run_event.clear()
    cap_thread.join(timeout=2.0)
    worker_thread.join(timeout=2.0)
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Stopped.")
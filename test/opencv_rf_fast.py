# eye_center_mediapipe_speed_optimized.py
# เร็วขึ้นโดยไม่เปลี่ยน HOG settings - ใช้ได้กับ model เดิม
import cv2, time, os, joblib
import numpy as np
from skimage.feature import hog
import mediapipe as mp
import threading, queue

# ---------------- Config ----------------
RF_MODEL_PATH = r"D:\Coding\Python\Project\eyeCenter\train4\models_gridsearch_with_radius\RandomForest_best.joblib"
CAM_INDEX = 0

# ===== ใช้ HOG settings เดิม (ตรงกับ model) =====
HOG_SIZE = (144, 144)
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

ROI_PAD = 0.3
SMOOTH_ALPHA = 0.6

DOT_COLOR = (0, 0, 255)
DOT_RADIUS = 1
CIRCLE_COLOR = (0, 255, 255)
CIRCLE_THICK = 1

# ===== SPEED OPTIMIZATION (ไม่แก้ HOG) =====
DETECT_DOWNSCALE = 0.4           # ลด MediaPipe resolution
PROCESS_EVERY_N = 3              # ประมวลผลทุก 3 เฟรม (แทน 2)
REFINE_LANDMARKS = False         # ปิด iris

# MediaPipe - ลด confidence เพื่อความเร็ว
MP_MIN_DETECTION = 0.3
MP_MIN_TRACKING = 0.3

# Camera - ลด resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# Queue
FRAME_QUEUE_MAXSIZE = 1          # เล็กลงเพื่อ latency ต่ำ
PRED_QUEUE_MAXSIZE = 1

# ลด eye landmarks (เร็วขึ้น)
LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158]    # 5 จุด (แทน 7-16)
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385]

# ---------------- Pre-compute HOG params ----------------
hog_params = {
    'pixels_per_cell': HOG_PIXELS_PER_CELL,
    'cells_per_block': HOG_CELLS_PER_BLOCK,
    'feature_vector': True,
    'transform_sqrt': False,     # ปิดเพื่อความเร็ว (ไม่กระทบมาก)
}

def extract_hog_optimized(img_gray, size=HOG_SIZE):
    """Optimized HOG - ใช้ settings เดิม"""
    # ใช้ INTER_AREA สำหรับ quality ดี (ช้ากว่า LINEAR นิดเดียว)
    resized = cv2.resize(img_gray, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    feats = hog(resized, **hog_params)
    return feats

def safe_predict_xy_r(model, feats):
    X = feats.reshape(1, -1)
    pred = model.predict(X)[0]
    
    px, py = pred[0], pred[1]
    pr = pred[2] if len(pred) >= 3 else 0.0
    
    px_pix = px * HOG_SIZE[0] if 0.0 <= px <= 1.0 else px
    py_pix = py * HOG_SIZE[1] if 0.0 <= py <= 1.0 else py
    pr_pix = pr * ((HOG_SIZE[0] + HOG_SIZE[1]) * 0.5) if 0.0 <= pr <= 1.0 else pr
    
    return px_pix, py_pix, pr_pix

class TemporalSmoother:
    """Temporal smoothing with buffer"""
    def __init__(self, alpha=0.6, buffer_size=3):
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.left_buffer = []
        self.right_buffer = []
        self.left = None
        self.right = None
    
    def update(self, preds):
        left_new = None
        right_new = None
        
        for side, x, y, r in preds:
            if side == "left":
                left_new = (x, y, r)
            else:
                right_new = (x, y, r)
        
        # Update with exponential smoothing
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

# ---------------- Load Model ----------------
print("Loading model...")
if not os.path.exists(RF_MODEL_PATH):
    raise FileNotFoundError(RF_MODEL_PATH)

rf = joblib.load(RF_MODEL_PATH)
nfeat = getattr(rf, "n_features_in_", None)
print(f"✓ Model loaded: n_features_in_ = {nfeat}")

# Test HOG features
test_gray = np.zeros((HOG_SIZE[1], HOG_SIZE[0]), dtype=np.uint8)
test_feats = extract_hog_optimized(test_gray)
print(f"✓ HOG test: {test_feats.size} features (model expects {nfeat})")

if nfeat is not None and test_feats.size != nfeat:
    print(f"\n❌ ERROR: Feature size mismatch!")
    print(f"   Model expects: {nfeat}")
    print(f"   Current HOG: {test_feats.size}")
    raise ValueError("Feature size mismatch. Check HOG settings.")

print(f"✓ Feature size matches!")

# ---------------- MediaPipe ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=REFINE_LANDMARKS,
    min_detection_confidence=MP_MIN_DETECTION,
    min_tracking_confidence=MP_MIN_TRACKING
)
print(f"✓ MediaPipe initialized")

# ---------------- Shared State ----------------
frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
pred_q = queue.Queue(maxsize=PRED_QUEUE_MAXSIZE)
run_event = threading.Event()
run_event.set()

stats_lock = threading.Lock()
proc_fps = 0.0
cap_fps = 0.0
mp_fps = 0.0
hog_fps = 0.0

smoother = TemporalSmoother(alpha=SMOOTH_ALPHA, buffer_size=3)

# ---------------- Capture Thread ----------------
def capture_thread_fn(cam_index, frame_q, run_event):
    global cap_fps
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        run_event.clear()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Disable auto exposure for stability
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera: {actual_w}x{actual_h}")
    
    last_time = time.time()
    while run_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        now = time.time()
        with stats_lock:
            cap_fps = 0.9 * cap_fps + 0.1 / max(1e-6, now - last_time)
        last_time = now
        
        # Always keep latest frame only
        try:
            frame_q.get_nowait()
        except queue.Empty:
            pass
        
        try:
            frame_q.put_nowait(frame)
        except queue.Full:
            pass
    
    cap.release()

# ---------------- Worker Thread ----------------
def worker_thread_fn(frame_q, pred_q, run_event):
    global proc_fps, mp_fps, hog_fps
    frame_idx = 0
    last_proc_time = time.time()
    
    small_w = int(CAM_WIDTH * DETECT_DOWNSCALE)
    small_h = int(CAM_HEIGHT * DETECT_DOWNSCALE)
    
    print(f"✓ Worker: Processing every {PROCESS_EVERY_N} frames, MP downscale={DETECT_DOWNSCALE}")
    
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
        
        # MediaPipe detection
        t_mp = time.time()
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_small)
        
        with stats_lock:
            mp_fps = 0.9 * mp_fps + 0.1 / max(1e-6, time.time() - t_mp)
        
        local_preds = []
        hog_times = []
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            for eye_idxs, side in [(LEFT_EYE_LANDMARKS, "left"),
                                   (RIGHT_EYE_LANDMARKS, "right")]:
                # Get eye bbox from landmarks
                xs = [int(lm[i].x * frame_w) for i in eye_idxs]
                ys = [int(lm[i].y * frame_h) for i in eye_idxs]
                
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
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
                
                # Extract patch
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                
                # Convert to gray
                patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                
                # HOG extraction
                t_hog = time.time()
                feats = extract_hog_optimized(patch_gray, size=HOG_SIZE)
                hog_times.append(time.time() - t_hog)
                
                if nfeat is not None and feats.size != nfeat:
                    print(f"⚠️  Feature mismatch for {side}: {feats.size} vs {nfeat}")
                    continue
                
                try:
                    px_pix, py_pix, pr_pix = safe_predict_xy_r(rf, feats)
                except Exception as e:
                    print(f"⚠️  Prediction error: {e}")
                    continue
                
                # Map back to frame
                mapped_x = x1 + (px_pix / HOG_SIZE[0]) * rw
                mapped_y = y1 + (py_pix / HOG_SIZE[1]) * rh
                mapped_r = (pr_pix / ((HOG_SIZE[0] + HOG_SIZE[1]) * 0.5)) * ((rw + rh) * 0.5)
                
                local_preds.append((side, float(mapped_x), float(mapped_y), float(mapped_r)))
        
        # Update HOG FPS
        if hog_times:
            with stats_lock:
                hog_fps = 0.9 * hog_fps + 0.1 / max(1e-6, np.mean(hog_times))
        
        # Put predictions (keep latest only)
        try:
            pred_q.get_nowait()
        except queue.Empty:
            pass
        
        try:
            pred_q.put_nowait(local_preds)
        except queue.Full:
            pass
        
        # Update process FPS
        t1 = time.time()
        with stats_lock:
            proc_fps = 0.9 * proc_fps + 0.1 / max(1e-6, t1 - last_proc_time)
        last_proc_time = t1

# ---------------- Main ----------------
print("=" * 60)
print("Eye Center Detection - Speed Optimized (Same HOG Settings)")
print("=" * 60)
print(f"HOG: {HOG_SIZE[0]}x{HOG_SIZE[1]}, cell={HOG_PIXELS_PER_CELL}")
print(f"Process every: {PROCESS_EVERY_N} frames")
print(f"MediaPipe downscale: {DETECT_DOWNSCALE}")
print("=" * 60)

cap_thread = threading.Thread(target=capture_thread_fn, args=(CAM_INDEX, frame_q, run_event), daemon=True)
worker_thread = threading.Thread(target=worker_thread_fn, args=(frame_q, pred_q, run_event), daemon=True)
cap_thread.start()
worker_thread.start()

prev_time = time.time()
display_fps = 0.0
preds_snapshot = []

print("\nControls:")
print("  q - Quit")
print("  s - Save snapshot")
print("  + - Process more frames (faster, more CPU)")
print("  - - Process fewer frames (slower, less CPU)")
print()

try:
    while run_event.is_set():
        # Get latest frame
        frame = None
        try:
            frame = frame_q.get(timeout=0.01)
        except queue.Empty:
            pass
        
        if frame is None:
            continue
        
        # Get latest predictions
        try:
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
            
            # Draw center dot
            cv2.circle(frame, (cx_i, cy_i), DOT_RADIUS, DOT_COLOR, -1)
            
            # Draw radius circle
            if r_i > 0:
                cv2.circle(frame, (cx_i, cy_i), r_i, CIRCLE_COLOR, CIRCLE_THICK)
            
            # Draw label
            label_color = (100, 255, 100) if side == "left" else (100, 200, 255)
            cv2.putText(frame, side, (cx_i + 8, cy_i - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)
        
        # Calculate display FPS
        now = time.time()
        display_fps = 0.9 * display_fps + 0.1 / max(1e-6, now - prev_time)
        prev_time = now
        
        # Get stats
        with stats_lock:
            pf = proc_fps
            cf = cap_fps
            mf = mp_fps
            hf = hog_fps
        
        # Draw info overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (420, 95), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # FPS info
        cv2.putText(frame, f"Display: {display_fps:.1f} FPS | Capture: {cf:.1f} FPS", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Process: {pf:.1f} FPS | MediaPipe: {mf:.1f} FPS", 
                   (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"HOG: {hf:.1f} FPS | Every {PROCESS_EVERY_N} frames", 
                   (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Eye Center Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            run_event.clear()
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"✓ Saved: {fname}")
        elif key == ord('+') or key == ord('='):
            PROCESS_EVERY_N = max(1, PROCESS_EVERY_N - 1)
            print(f"Process every {PROCESS_EVERY_N} frames")
        elif key == ord('-') or key == ord('_'):
            PROCESS_EVERY_N = min(10, PROCESS_EVERY_N + 1)
            print(f"Process every {PROCESS_EVERY_N} frames")

finally:
    print("\n✓ Shutting down...")
    run_event.clear()
    cap_thread.join(timeout=2.0)
    worker_thread.join(timeout=2.0)
    cv2.destroyAllWindows()
    face_mesh.close()
    print("✓ Stopped.")
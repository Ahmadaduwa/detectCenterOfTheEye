# eye_center_mediapipe_optimized.py
# Highly optimized threaded capture + worker for maximum FPS
import cv2, time, os, math, joblib
import numpy as np
from skimage.feature import hog
import mediapipe as mp
import threading, queue

# ---------------- Config (optimized for speed) ----------------
RF_MODEL_PATH = r"D:\Coding\Python\Project\eyeCenter\train4\models_gridsearch_with_radius\RandomForest_best.joblib"
CAM_INDEX = 0

HOG_SIZE = (144, 144)
ROI_PAD = 0.35
SMOOTH_ALPHA = 0.45

DOT_COLOR = (0, 0, 255)
DOT_RADIUS = 3
CIRCLE_COLOR = (0, 255, 255)
CIRCLE_THICK = 1

# ===== SPEED OPTIMIZATIONS =====
DETECT_DOWNSCALE = 0.4       # More aggressive downscale (was 0.5)
PROCESS_EVERY_N = 3          # Process every 3rd frame (was 2)
REFINE_LANDMARKS = False     # Keep False for speed
MP_MIN_DETECTION = 0.3       # Lower threshold for faster detection (was 0.5)
MP_MIN_TRACKING = 0.3        # Lower threshold for faster tracking (was 0.5)

# Camera settings
CAM_WIDTH = 640              # Lower resolution = faster
CAM_HEIGHT = 480
CAM_FPS = 30

# HOG params
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Queue settings
FRAME_QUEUE_MAXSIZE = 2      # Slightly larger buffer
PRED_QUEUE_MAXSIZE = 2

# Smoothing buffer
SMOOTH_BUFFER_SIZE = 3       # Temporal smoothing

# ---------------- helpers ----------------
def extract_hog_from_gray(img_gray, size=HOG_SIZE):
    resized = cv2.resize(img_gray, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR faster than INTER_AREA
    feats = hog(resized, pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK, feature_vector=True)
    return feats

def safe_predict_xy_r(model, feats):
    X = feats.reshape(1, -1)  # Already float32 from HOG
    pred = model.predict(X)[0]  # Direct indexing faster
    
    px, py = pred[0], pred[1]
    pr = pred[2] if len(pred) >= 3 else 0.0
    
    # Vectorized conversion
    if 0.0 <= px <= 1.0:
        px_pix = px * HOG_SIZE[0]
    else:
        px_pix = px
        
    if 0.0 <= py <= 1.0:
        py_pix = py * HOG_SIZE[1]
    else:
        py_pix = py
        
    if 0.0 <= pr <= 1.0:
        pr_pix = pr * ((HOG_SIZE[0] + HOG_SIZE[1]) * 0.5)
    else:
        pr_pix = pr
        
    return px_pix, py_pix, pr_pix

class SmoothingBuffer:
    """Simple exponential moving average for predictions"""
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

# ---------------- load RF model ----------------
if not os.path.exists(RF_MODEL_PATH):
    raise FileNotFoundError(RF_MODEL_PATH)
rf = joblib.load(RF_MODEL_PATH)
nfeat = getattr(rf, "n_features_in_", None)
print("Loaded RF model:", RF_MODEL_PATH, "n_features_in_:", nfeat)

# ---------------- MediaPipe setup (optimized) ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=REFINE_LANDMARKS,
    min_detection_confidence=MP_MIN_DETECTION,
    min_tracking_confidence=MP_MIN_TRACKING
)

# Optimized landmark sets (fewer points for faster processing)
LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173]  # Reduced from 9 to 7
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398]

# ---------------- shared state ----------------
frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
pred_q = queue.Queue(maxsize=PRED_QUEUE_MAXSIZE)
run_event = threading.Event()
run_event.set()

# metrics
stats_lock = threading.Lock()
proc_fps = 0.0
cap_fps = 0.0

smoother = SmoothingBuffer(alpha=SMOOTH_ALPHA)

# ---------------- capture thread (optimized) ----------------
def capture_thread_fn(cam_index, frame_q, run_event):
    global cap_fps
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if not cap.isOpened():
        print("Cannot open camera", cam_index)
        run_event.clear()
        return
    
    # Set camera properties for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
    
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(cap.get(cv2.CAP_PROP_FPS))}fps")
    
    last_time = time.time()
    while run_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update capture FPS
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

# ---------------- worker thread (optimized) ----------------
def worker_thread_fn(frame_q, pred_q, run_event):
    global proc_fps
    frame_idx = 0
    last_proc_time = time.time()
    
    # Pre-allocate for speed
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
        
        # Faster resize
        small = cv2.resize(frame, small_size, interpolation=cv2.INTER_LINEAR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_small)
        
        local_preds = []
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            for idxs in (LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS):
                # Vectorized operations
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
                
                patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                feats = extract_hog_from_gray(patch_gray, size=HOG_SIZE)
                
                if nfeat is not None and feats.size != nfeat:
                    continue
                
                try:
                    px_pix, py_pix, pr_pix = safe_predict_xy_r(rf, feats)
                except Exception:
                    continue
                
                # Map back to original frame
                mapped_x = x1 + (px_pix / HOG_SIZE[0]) * rw
                mapped_y = y1 + (py_pix / HOG_SIZE[1]) * rh
                mapped_r = (pr_pix / ((HOG_SIZE[0] + HOG_SIZE[1]) * 0.5)) * ((rw + rh) * 0.5)
                
                # Determine side
                eye_center_x = (x_min + x_max) * 0.5
                side = "right" if eye_center_x < (frame_w * 0.5) else "left"
                
                local_preds.append((side, float(mapped_x), float(mapped_y), float(mapped_r)))
        
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

# ---------------- main (UI) ----------------
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
            time.sleep(0.001)  # Very short sleep
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
            
            cv2.circle(frame, (cx_i, cy_i), DOT_RADIUS, DOT_COLOR, -1)
            if r_i > 0:
                cv2.circle(frame, (cx_i, cy_i), r_i, CIRCLE_COLOR, CIRCLE_THICK)
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
        cv2.putText(frame, f"Display:{display_fps:.1f} Process:{pf:.1f} Capture:{cf:.1f}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Eye Center (Optimized)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            run_event.clear()
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print("Saved", fname)

finally:
    print("Shutting down...")
    run_event.clear()
    cap_thread.join(timeout=2.0)
    worker_thread.join(timeout=2.0)
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Stopped.")
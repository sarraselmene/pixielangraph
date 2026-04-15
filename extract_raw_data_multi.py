import os
import re
import sys
import csv
import time
import queue
import shutil
import logging
import threading
import subprocess
import gc

import cv2
import torch
import numpy as np
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)
from sixdrepnet import SixDRepNet

# ==============================================================================
# CONFIG & PATHS
# ==============================================================================
INPUT_SOURCE     = "aya2.mov"  # Can be patched by graph_nodes
work_dir         = "."         # Can be patched by graph_nodes

BODY_OUTPUT      = "raw_body_multi.csv"
HEAD_POSE_OUTPUT = "raw_head_pose_multi.csv"
AU_OUTPUT        = "raw_action_units_multi.csv"
GAZE_OUTPUT      = "raw_gaze_multi.csv"
FACE_CROPS_DIR   = "face_crops"
OPENFACE_OUT_DIR = "openface_output"

# Define workspace as the parent directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POSE_MODEL_PATH      = os.path.join(BASE_DIR, "yolo11m-pose.pt")
FACE_YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov11m-face.pt")

# If OpenFace fails or is elsewhere, we expect paths to remain valid over absolute overrides.
OPENFACE_DIR = "/Users/sarahselmene/OpenFace/build/bin"
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FaceLandmarkImg")

# Performance tuning
FRAME_STRIDE        = 1
INFERENCE_SIZE      = 480
EXPAND_RATIO        = 0.20
OPENFACE_BATCH_SIZE = 300

# EMA smoothing
EMA_ALPHA = 0.6

# ID stabilizer thresholds
ID_MEMORY_FRAMES = 60    
ID_IOU_THRESH    = 0.25  
ID_DIST_THRESH   = 0.25  

# COCO keypoint indices
KP_LEFT_WRIST  = 9
KP_RIGHT_WRIST = 10
KP_LEFT_KNEE   = 13
KP_RIGHT_KNEE  = 14
KNEE_INDICES   = {KP_LEFT_KNEE, KP_RIGHT_KNEE}
WRIST_INDICES  = {KP_LEFT_WRIST, KP_RIGHT_WRIST}
KNEE_CONF_MIN  = 0.5

AU_INTENSITY_COLS = ["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r"]
AU_BINARY_COLS    = ["AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]
GAZE_COLS         = ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_1_x","gaze_1_y","gaze_1_z","gaze_angle_x","gaze_angle_y"]
FILENAME_PATTERN  = re.compile(r"frame_(\d+)_track_(\d+)")

# ==============================================================================
# ID STABILIZER & EMA 
# ==============================================================================
class IDStabilizer:
    def __init__(self, frame_w: int, frame_h: int):
        self.diag = (frame_w**2 + frame_h**2) ** 0.5
        self._lost: dict[int, dict] = {}
        self._remap: dict[int, int] = {}

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
        inter = iw * ih
        if inter == 0: return 0.0
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / max(union, 1)

    def _dist(self, a, b) -> float:
        cax, cay = (a[0]+a[2])/2, (a[1]+a[3])/2
        cbx, cby = (b[0]+b[2])/2, (b[1]+b[3])/2
        return ((cax-cbx)**2 + (cay-cby)**2) ** 0.5 / self.diag

    def update(self, current: dict[int, list]) -> dict[int, int]:
        remap: dict[int, int] = {}
        seen_stable: set[int] = set()

        for raw_id, bbox in current.items():
            if raw_id in self._remap:
                stable = self._remap[raw_id]
                remap[raw_id] = stable
                seen_stable.add(stable)
                self._lost.pop(stable, None)
                continue

            best_stable = raw_id
            best_score  = -1.0
            for s_id, info in self._lost.items():
                iou  = self._iou(bbox, info["bbox"])
                dist = self._dist(bbox, info["bbox"])
                score = iou - dist
                if iou >= ID_IOU_THRESH and dist <= ID_DIST_THRESH and score > best_score:
                    best_score  = score
                    best_stable = s_id

            if best_stable != raw_id:
                self._remap[raw_id] = best_stable
                self._lost.pop(best_stable, None)

            remap[raw_id] = best_stable
            seen_stable.add(best_stable)

        for s_id in list(self._lost.keys()):
            if s_id not in seen_stable:
                self._lost[s_id]["lost_frames"] += 1
                if self._lost[s_id]["lost_frames"] > ID_MEMORY_FRAMES:
                    del self._lost[s_id]
                    self._remap = {k: v for k, v in self._remap.items() if v != s_id}

        for raw_id, bbox in current.items():
            s_id = remap.get(raw_id, raw_id)
            self._lost[s_id] = {"bbox": list(bbox), "lost_frames": 0}

        return remap

ema_state: dict[int, dict[int, list[float]]] = {}
def apply_ema(track_id: int, kp_idx: int, x: float, y: float):
    ts = ema_state.setdefault(track_id, {})
    if kp_idx not in ts:
        ts[kp_idx] = [x, y]
    else:
        px, py = ts[kp_idx]
        x = EMA_ALPHA * x + (1.0 - EMA_ALPHA) * px
        y = EMA_ALPHA * y + (1.0 - EMA_ALPHA) * py
        ts[kp_idx] = [x, y]
    return x, y

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    elif torch.cuda.is_available(): return "cuda"
    return "cpu"

def expand_bbox(x1, y1, x2, y2, frame_h, frame_w, expand=EXPAND_RATIO):
    w, h = x2-x1, y2-y1
    x1 = max(0,       int(x1 - w*expand/2))
    y1 = max(0,       int(y1 - h*expand/2))
    x2 = min(frame_w, int(x2 + w*expand/2))
    y2 = min(frame_h, int(y2 + h*expand/2))
    return x1, y1, x2, y2

# ==============================================================================
# OPENFACE BACKGROUND WORKER
# ==============================================================================
class OpenFaceWorker:
    def __init__(self, face_crops_dir, openface_out_dir, batch_size=OPENFACE_BATCH_SIZE):
        self.face_crops_dir   = face_crops_dir
        self.openface_out_dir = openface_out_dir
        self.batch_size       = batch_size
        self.pending_crops    = []
        self.batch_count      = 0
        self.lock             = threading.Lock()
        self.task_queue       = queue.Queue()
        self.thread           = threading.Thread(target=self._worker_loop, daemon=True)
        self.total_processed  = 0

    def start(self):
        os.makedirs(self.openface_out_dir, exist_ok=True)
        self.thread.start()
        print("[OpenFace] Background worker started")

    def add_crop(self, filename):
        with self.lock:
            self.pending_crops.append(filename)
            if len(self.pending_crops) >= self.batch_size:
                batch = self.pending_crops.copy()
                self.pending_crops.clear()
                self.batch_count += 1
                self.task_queue.put(("batch", self.batch_count, batch))

    def flush_and_stop(self):
        with self.lock:
            if self.pending_crops:
                self.batch_count += 1
                batch = self.pending_crops.copy()
                self.pending_crops.clear()
                self.task_queue.put(("batch", self.batch_count, batch))
        self.task_queue.put(("stop", None, None))
        self.thread.join(timeout=600)

    def _worker_loop(self):
        while True:
            action, batch_id, data = self.task_queue.get()
            if action == "stop": break
            self._process_batch(batch_id, data)

    def _process_batch(self, batch_id, filenames):
        batch_dir     = os.path.join(self.face_crops_dir, f"_batch_{batch_id}")
        batch_out_dir = os.path.join(self.openface_out_dir, f"batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(batch_out_dir, exist_ok=True)
        for fname in filenames:
            src = os.path.join(self.face_crops_dir, fname)
            if os.path.exists(src): os.rename(src, os.path.join(batch_dir, fname))
        cmd = [OPENFACE_EXE, "-fdir", os.path.abspath(batch_dir),
               "-out_dir", os.path.abspath(batch_out_dir),
               "-aus", "-gaze", "-multi_view", "1"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=OPENFACE_DIR, timeout=300)
            print(f"[OpenFace] Batch {batch_id} {'complete' if r.returncode==0 else 'warning exit'}")
        except Exception as e:
            pass
        self.total_processed += len(filenames)
        for fname in filenames:
            src = os.path.join(batch_dir, fname)
            if os.path.exists(src): os.rename(src, os.path.join(self.face_crops_dir, fname))
        try:
            os.rmdir(batch_dir)
        except OSError: pass


def merge_openface_outputs(openface_out_dir, au_csv, gaze_csv):
    au_rows, gaze_rows = [], []
    for root, _, files in os.walk(openface_out_dir):
        for csv_file in files:
            if not csv_file.endswith(".csv"): continue
            match = FILENAME_PATTERN.search(os.path.splitext(csv_file)[0])
            if not match: continue
            frame_id, track_id = int(match.group(1)), int(match.group(2))
            with open(os.path.join(root, csv_file), "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cleaned = {k.strip(): v.strip() for k, v in row.items()}
                    conf    = float(cleaned.get("confidence", 0))
                    success = int(cleaned.setdefault("success", 1))
                    common  = {"frame_id": frame_id, "track_id": track_id,
                               "confidence": f"{conf:.4f}", "success": success}
                    out_au, out_gz = dict(common), dict(common)
                    for c in AU_INTENSITY_COLS + AU_BINARY_COLS: out_au[c] = cleaned.get(c, "") if success else ""
                    for c in GAZE_COLS: out_gz[c] = cleaned.get(c, "") if success else ""
                    au_rows.append(out_au)
                    gaze_rows.append(out_gz)

    au_rows.sort(key=lambda r: (r["frame_id"], r["track_id"]))
    gaze_rows.sort(key=lambda r: (r["frame_id"], r["track_id"]))

    with open(au_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id","track_id","confidence","success"]+AU_INTENSITY_COLS+AU_BINARY_COLS)
        w.writeheader(); w.writerows(au_rows)
    with open(gaze_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id","track_id","confidence","success"]+GAZE_COLS)
        w.writeheader(); w.writerows(gaze_rows)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    device = get_device()
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading YOLOv11-pose (Body Detection)...")
    pose_model = YOLO(POSE_MODEL_PATH)
    if device == "mps": pose_model.model.to("mps")
    else: pose_model.to(device)

    print("[INFO] Loading YOLOv11m-face (Facial Refinement)...")
    face_det_model = YOLO(FACE_YOLO_MODEL_PATH)
    if device == "mps": face_det_model.model.to("mps")
    else: face_det_model.to(device)

    print("[INFO] Initializing 6DRepNet (Safe CPU Mode)...")
    class SafeHeadPoseEstimator:
        def __init__(self):
            # Enforce CPU operation due to Apple MPS batch norm incompatibilities
            self.model = SixDRepNet(gpu_id=-1)
        def predict(self, face_crop):
            try: return self.model.predict(face_crop)
            except Exception: return None, None, None
    sixd_model = SafeHeadPoseEstimator()

    for d in [FACE_CROPS_DIR, OPENFACE_OUT_DIR]:
        if os.path.isdir(d): shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    body_csv_file      = open(BODY_OUTPUT,      "w", newline="", encoding="utf-8")
    head_pose_csv_file = open(HEAD_POSE_OUTPUT, "w", newline="", encoding="utf-8")
    body_writer        = csv.writer(body_csv_file)
    head_pose_writer   = csv.writer(head_pose_csv_file)

    body_writer.writerow([
        "frame_id","timestamp_sec","track_id","landmark_idx",
        "x","y","visibility","bbox_x1","bbox_y1","bbox_x2","bbox_y2","global_desk_y",
    ])
    head_pose_writer.writerow(["frame_id","timestamp_sec","track_id","pitch","yaw","roll"])

    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened(): sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30.0

    stabilizer = IDStabilizer(frame_w, frame_h)
    
    openface_available = os.path.isfile(OPENFACE_EXE)
    of_worker = OpenFaceWorker(FACE_CROPS_DIR, OPENFACE_OUT_DIR) if openface_available else None
    if of_worker: of_worker.start()

    frame_id = 0
    processed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            print(f"Processing frame {frame_id}/{total_frames}...", end="\r", flush=True)
            if frame_id % FRAME_STRIDE != 0:
                frame_id += 1; continue

            # ── 1. POSE TRACKING ──
            results = pose_model.track(
                source=frame, tracker="botsort.yaml", persist=True,
                conf=0.25, iou=0.5, classes=[0], imgsz=INFERENCE_SIZE,
                stream=False, verbose=False, device=device
            )
            boxes = results[0].boxes
            keypoints_data = results[0].keypoints

            if boxes is None or len(boxes) == 0 or boxes.id is None:
                frame_id += 1; continue

            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            raw_ids   = boxes.id.cpu().numpy().astype(int)
            all_kpts  = keypoints_data.data.cpu().numpy() if (keypoints_data is not None and keypoints_data.data is not None) else None

            # ── 2. STABILIZE IDS ──
            current_tracks = {int(raw_ids[i]): list(xyxy_list[i]) for i in range(len(raw_ids))}
            remap = stabilizer.update(current_tracks)

            # Global Desk
            wrist_ys = []
            if all_kpts is not None:
                for p_idx in range(len(all_kpts)):
                    for kp_idx in WRIST_INDICES:
                        kp = all_kpts[p_idx][kp_idx]
                        if float(kp[2]) >= 0.3: wrist_ys.append(float(kp[1]))
            gdsk = float(np.median(wrist_ys)) if wrist_ys else None
            gdsk_str = f"{gdsk:.4f}" if gdsk is not None else ""

            # ── 3. EXTRACT FOR EACH PERSON ──
            for person_idx, (bbox, raw_id) in enumerate(zip(xyxy_list, raw_ids)):
                x1, y1, x2, y2 = bbox
                tid = remap.get(int(raw_id), int(raw_id))

                # Body Keypoints -> body_csv
                if all_kpts is not None and person_idx < len(all_kpts):
                    kpts = all_kpts[person_idx]
                    for lm_idx in range(kpts.shape[0]):
                        rx, ry, conf = float(kpts[lm_idx, 0]), float(kpts[lm_idx, 1]), float(kpts[lm_idx, 2])
                        if lm_idx in KNEE_INDICES and conf < KNEE_CONF_MIN:
                            body_writer.writerow([frame_id, f"{frame_id/video_fps:.3f}", tid, lm_idx, None, None, f"{conf:.4f}", x1, y1, x2, y2, gdsk_str])
                        else:
                            sx, sy = apply_ema(tid, lm_idx, rx, ry)
                            body_writer.writerow([frame_id, f"{frame_id/video_fps:.3f}", tid, lm_idx, f"{sx:.4f}", f"{sy:.4f}", f"{conf:.4f}", x1, y1, x2, y2, gdsk_str])
                else:
                    body_writer.writerow([frame_id, f"{frame_id/video_fps:.3f}", tid, None, None, None, None, x1, y1, x2, y2, gdsk_str])

                # ── 4. ISOLATE FACE WITHIN BODY CROP ──
                cx1, cy1, cx2, cy2 = expand_bbox(x1, y1, x2, y2, frame_h, frame_w, EXPAND_RATIO)
                if cx2-cx1 < 10 or cy2-cy1 < 10:
                    head_pose_writer.writerow([frame_id, f"{frame_id/video_fps:.3f}", tid, None, None, None])
                    continue

                crop_bgr = frame[cy1:cy2, cx1:cx2]
                
                try:
                    face_results = face_det_model(crop_bgr, verbose=False)
                    face_boxes = face_results[0].boxes if len(face_results) > 0 else None
                    best_face = None
                    if face_boxes is not None and len(face_boxes) > 0:
                        confs = face_boxes.conf.cpu().numpy()
                        best_idx = np.argmax(confs)
                        if confs[best_idx] > 0.5:
                            best_face = face_boxes.xyxy.cpu().numpy()[best_idx].astype(int)

                    if best_face is None:
                        head_pose_writer.writerow([frame_id, f"{frame_id/video_fps:.3f}", tid, None, None, None])
                        continue

                    fx1, fy1, fx2, fy2 = best_face
                    fw, fh = fx2-fx1, fy2-fy1
                    cf1 = max(0, int(fx1 - fw*0.05))
                    cf2 = max(0, int(fy1 - fh*0.05))
                    cf3 = min(cx2-cx1, int(fx2 + fw*0.05))
                    cf4 = min(cy2-cy1, int(fy2 + fh*0.05))
                    face_crop = crop_bgr[cf2:cf4, cf1:cf3]

                    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                        head_pose_writer.writerow([frame_id, f"{frame_id/video_fps:.3f}", tid, None, None, None])
                        continue

                    fname = f"frame_{frame_id:06d}_track_{tid}.jpg"
                    cv2.imwrite(os.path.join(FACE_CROPS_DIR, fname), face_crop)
                    if of_worker: of_worker.add_crop(fname)

                    pitch, yaw, roll = sixd_model.predict(face_crop)
                    if pitch is not None and yaw is not None and roll is not None:
                        head_pose_writer.writerow([
                            frame_id, f"{frame_id/video_fps:.3f}", tid,
                            f"{float(np.ravel(pitch)[0]):.4f}", f"{float(np.ravel(yaw)[0]):.4f}", f"{float(np.ravel(roll)[0]):.4f}"
                        ])
                    else:
                        head_pose_writer.writerow([frame_id, f"{frame_id/video_fps:.3f}", tid, None, None, None])

                except Exception as e:
                    head_pose_writer.writerow([frame_id, f"{frame_id/video_fps:.3f}", tid, None, None, None])

            processed_frames += 1
            frame_id += 1
            if processed_frames % 50 == 0: 
                body_csv_file.flush()
                head_pose_csv_file.flush()
            
    except KeyboardInterrupt: pass
    finally:
        cap.release()
        body_csv_file.close()
        head_pose_csv_file.close()
        print(f"\n[INFO] Done extraction. Waiting for OpenFace worker...")
        if of_worker:
            of_worker.flush_and_stop()
            merge_openface_outputs(OPENFACE_OUT_DIR, AU_OUTPUT, GAZE_OUTPUT)
        print("[INFO] Pipeline complete.")

if __name__ == "__main__":
    main()
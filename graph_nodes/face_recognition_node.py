import os
import glob
import csv
import numpy as np
from collections import Counter


def run_face_recognition(state: dict) -> dict:
    """
    LangGraph node: Recognizes faces from face_crops and maps track_ids to Student Names.
    Uses SixDRepNet for robust identification by selecting only front-facing samples.
    """
    # ── Lazy imports for heavy ML packages ────────────────────────────────────
    try:
        import cv2
        CV2_OK = True
    except ImportError:
        CV2_OK = False

    try:
        import torch
        TORCH_OK = True
    except ImportError:
        TORCH_OK = False

    try:
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
    except ImportError:
        DEEPFACE_AVAILABLE = False

    try:
        from sixdrepnet import SixDRepNet
        from torchvision import transforms
        SIXD_AVAILABLE = True
    except ImportError:
        SIXD_AVAILABLE = False

    work_dir = state.get("work_dir", "")
    face_crops_dir = os.path.join(work_dir, "face_crops")
    db_path = state.get("face_db_path", "student_database")
    
    identity_map = state.get("identity_map", {})
    
    if not os.path.exists(face_crops_dir):
        print(f"[FaceRec] WARNING: face_crops dir not found. Skipping face recognition.")
        return {"identity_map": identity_map}

    if not DEEPFACE_AVAILABLE:
        print(f"[FaceRec] WARNING: DeepFace not installed. Identities will default to Track IDs.")
        for crop_file in glob.glob(os.path.join(face_crops_dir, "*.jpg")):
            parts = os.path.basename(crop_file).split("_")
            if len(parts) >= 4 and parts[2] == "track":
                try:
                    tid = int(parts[3].split(".")[0])
                    identity_map[tid] = f"Student_{tid}"
                except ValueError:
                    continue
        return {"identity_map": identity_map}

    # Initialize Head Pose Model for Robustness
    device = "cpu"
    if TORCH_OK:
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
    
    head_pose_model = None
    transform = None
    if SIXD_AVAILABLE and TORCH_OK:
        try:
            # Match extraction node device logic (safe CPU fallback if needed)
            gpu_id = 0 if device == "cuda" else -1
            head_pose_model = SixDRepNet(gpu_id=gpu_id)
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(f"[FaceRec] SixDRepNet initialised on {device}")
        except Exception as e:
            print(f"[FaceRec] Error initialising SixDRepNet: {e}")
            head_pose_model = None

    print(f"\n{'='*60}")
    print(f"[Node: Face Recognition] Running identity mapping with pose-robust filtering...")
    
    # 1. Group crops by track_id
    track_crops = {}
    for crop_file in glob.glob(os.path.join(face_crops_dir, "*.jpg")):
        parts = os.path.basename(crop_file).split("_")
        if len(parts) >= 4 and parts[2] == "track":
            try:
                tid = int(parts[3].split(".")[0])
                if tid not in track_crops:
                    track_crops[tid] = []
                track_crops[tid].append(crop_file)
            except ValueError:
                continue

    # 2. Identify each track ID
    for tid, crops in track_crops.items():
        # Sample to save time (max 15 crops per track)
        step = max(1, len(crops) // 15)
        sample_crops = crops[::step][:15] 
        
        # Robust Selection: Filter for most frontal faces
        id_crops = []
        if head_pose_model and transform:
            scored_crops = []
            for cp in sample_crops:
                try:
                    img = cv2.imread(cp)
                    if img is None: continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # --- Robust Snippet Integration ---
                    face_img = transform(img).unsqueeze(0)
                    face_img = face_img.to(device) 
                    prediction = head_pose_model(face_img) # Returns [pitch, yaw, roll]
                    # ----------------------------------
                    
                    pitch, yaw, roll = prediction
                    # Score by frontality (closer to 0 is better)
                    front_score = abs(float(pitch[0])) + abs(float(yaw[0]))
                    scored_crops.append((front_score, cp))
                except:
                    continue
            
            scored_crops.sort(key=lambda x: x[0])
            id_crops = [x[1] for x in scored_crops if x[0] < 40] # Frontal only (<40 deg combined dev)
            if not id_crops:
                id_crops = [x[1] for x in scored_crops[:5]] # Fallback to best 5
        else:
            id_crops = sample_crops[:5]

        predictions = []
        for crop_path in id_crops:
            try:
                dfs = DeepFace.find(
                    img_path=crop_path,
                    db_path=db_path,
                    model_name="ArcFace", 
                    detector_backend="skip",
                    enforce_detection=False,
                    silent=True
                )
                if len(dfs) > 0 and not dfs[0].empty:
                    student_name = os.path.basename(os.path.dirname(dfs[0].iloc[0]["identity"]))
                    predictions.append(student_name)
            except:
                pass
                
        if predictions:
            most_common = Counter(predictions).most_common(1)[0][0]
            identity_map[tid] = most_common
            print(f"[FaceRec] Track ID {tid} -> {most_common} (from {len(predictions)} matches)")
        else:
            identity_map[tid] = f"Student_{tid}"
            print(f"[FaceRec] Track ID {tid} -> Unidentified")

    # Save mapping to CSV
    map_csv = os.path.join(work_dir, "face_identity_map.csv")
    with open(map_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "student_name"])
        for tid_key, name in identity_map.items():
            w.writerow([tid_key, name])
            
    print(f"[Node: Face Recognition] ✅ Done. Mapping saved to {map_csv}")
    print(f"{'='*60}\n")
    
    return {
        "identity_map": identity_map,
        "identity_map_csv": map_csv
    }


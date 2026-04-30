"""
graph_nodes/preprocessor_node.py
================================
LangGraph node: Preprocesses merged analysis CSV for BiLSTM inference.

Input:  full_analysis.csv (from merge_node)
Output: processed_features.npy + metadata CSV (frame_id, track_id mapping)

Pipeline:
  full_analysis.csv → feature extraction → StandardScaler → one-hot encode
                    → processed_features.npy  (N, 26)  float32
                    → processed_metadata.csv  (N, 2)   frame_id + track_id

Compatible with PixieBiLSTM model (train_lstm.py):
  - 4 numerical features (scaled)
  - 5 boolean features (cast to float)
  - 17 one-hot encoded features (from 5 categorical columns)
  - Total: 26 features per frame
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocessor_node")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SPECIFICATION (must match train_lstm.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

NUM_COLS = [
    "gaze_stability",
    "eye_head_divergence",
    "expressiveness_score",
    "disp_score",
]
BOOL_COLS = [
    "genuine_smile",
    "fatigue_indicator",
    "yawning",
    "talking_flag",
    "hand_raised",
]
CAT_COLS = [
    "pose_label",
    "gaze_h_label",
    "gaze_v_label",
    "posture",
    "context",
]

DEFAULT_SCALER   = "scaler.pkl"
DEFAULT_ENCODERS = "encoders.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN DERIVATION
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist, deriving missing ones."""
    df = df.copy()

    # posture_label → posture (merge_node renames posture → posture_label)
    if "posture" not in df.columns and "posture_label" in df.columns:
        df["posture"] = df["posture_label"]

    # hand_raised: derive from action column if available
    if "hand_raised" not in df.columns:
        if "action" in df.columns:
            df["hand_raised"] = (
                df["action"].astype(str).str.lower().str.contains("hand_rais")
            ).astype(float)
        else:
            df["hand_raised"] = 0.0

    # context: doesn't exist at inference time — default to "lecture"
    if "context" not in df.columns:
        df["context"] = "lecture"

    # Fill missing numerical columns
    for col in NUM_COLS:
        if col not in df.columns:
            log.warning(f"  Missing numerical column '{col}' → 0.0")
            df[col] = 0.0

    # Fill missing boolean columns
    for col in BOOL_COLS:
        if col not in df.columns:
            log.warning(f"  Missing boolean column '{col}' → 0")
            df[col] = 0.0

    # Fill missing categorical columns
    cat_defaults = {
        "pose_label": "Up", "gaze_h_label": "Center",
        "gaze_v_label": "Level", "posture": "sitting", "context": "lecture",
    }
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = cat_defaults.get(col, "unknown")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_for_lstm(
    input_csv: str,
    scaler_path: str,
    encoders_path: str,
    output_npy: str,
    output_meta_csv: str,
) -> tuple[str, str]:
    """CSV → scaled/encoded .npy + metadata CSV.  Returns (npy_path, meta_path)."""

    log.info("=" * 58)
    log.info("  PIXIE — PREPROCESSOR NODE (PyTorch BiLSTM)")
    log.info("=" * 58)

    # 1. Load
    df = pd.read_csv(input_csv, low_memory=False)
    log.info(f"  Loaded {len(df)} rows × {df.shape[1]} cols from {Path(input_csv).name}")

    # 2. Ensure columns
    df = _ensure_columns(df)

    # 3. Save metadata
    meta_df = df[["frame_id", "track_id"]].copy()
    meta_df.to_csv(output_meta_csv, index=False)

    # 4. Load fitted artifacts
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoders_path, "rb") as f:
        enc_data = pickle.load(f)

    encoders   = enc_data["encoders"]
    input_dim  = enc_data["input_dim"]

    # 5. Numerical → scale
    num_df = df[NUM_COLS].copy()
    for c in NUM_COLS:
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce").fillna(0.0)
    num_df = num_df.astype(np.float32).replace([np.inf, -np.inf], 0.0)
    num_arr = scaler.transform(num_df.values).astype(np.float32)

    # 6. Boolean → float (handle Python bool, string "True"/"False", and numeric)
    bool_df = df[BOOL_COLS].copy()
    for c in BOOL_COLS:
        col = bool_df[c]
        if col.dtype == bool or col.dtype == object:
            bool_df[c] = col.map(
                lambda v: 1.0 if v is True or str(v).strip().lower() in ("true", "1", "1.0") else 0.0
            )
        else:
            bool_df[c] = pd.to_numeric(col, errors="coerce").fillna(0.0)
    bool_arr = bool_df.astype(np.float32).values

    # 7. Categorical → one-hot
    ohe_parts = []
    for col in CAT_COLS:
        le = encoders[col]
        safe = df[col].astype(str).apply(lambda v: v if v in le.classes_ else le.classes_[0])
        indices = le.transform(safe)
        ohe = np.eye(len(le.classes_), dtype=np.float32)[indices]
        ohe_parts.append(ohe)
    cat_arr = np.concatenate(ohe_parts, axis=1)

    # 8. Concatenate
    X = np.concatenate([num_arr, bool_arr, cat_arr], axis=1).astype(np.float32)

    # Pad/trim if needed
    if X.shape[1] != input_dim:
        log.warning(f"  Dim mismatch: got {X.shape[1]}, expected {input_dim}")
        if X.shape[1] < input_dim:
            X = np.concatenate([X, np.zeros((len(X), input_dim - X.shape[1]), np.float32)], axis=1)
        else:
            X = X[:, :input_dim]

    # 9. Save
    os.makedirs(os.path.dirname(os.path.abspath(output_npy)) or ".", exist_ok=True)
    np.save(output_npy, X)
    log.info(f"  ✅ Features: {X.shape}  → {output_npy}")

    return output_npy, output_meta_csv


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessor_node(state: dict) -> dict:
    """
    LangGraph node: Preprocess merged CSV for LSTM inference.

    Reads:   full_analysis_csv, work_dir
    Produces: processed_features_npy, processed_metadata_csv, preprocessor_done
    """
    work_dir = state.get("work_dir", ".")
    full_csv = state.get("full_analysis_csv", "")

    print(f"\n{'='*60}")
    print(f"[Node: Preprocessor] Preparing features for BiLSTM")
    print(f"{'='*60}")

    if not full_csv or not os.path.isfile(full_csv):
        msg = f"[Preprocessor] ERROR: full_analysis.csv not found → {full_csv}"
        print(msg)
        return {"preprocessor_done": False, "error": msg}

    scaler_path   = os.path.join(work_dir, DEFAULT_SCALER)
    encoders_path = os.path.join(work_dir, DEFAULT_ENCODERS)

    if not os.path.isfile(scaler_path) or not os.path.isfile(encoders_path):
        msg = (f"[Preprocessor] ERROR: Missing artifacts: "
               f"scaler={os.path.isfile(scaler_path)}, "
               f"encoders={os.path.isfile(encoders_path)}")
        print(msg)
        return {"preprocessor_done": False, "error": msg}

    output_npy  = os.path.join(work_dir, "processed_features.npy")
    output_meta = os.path.join(work_dir, "processed_metadata.csv")

    try:
        preprocess_for_lstm(full_csv, scaler_path, encoders_path, output_npy, output_meta)
    except Exception as exc:
        msg = f"[Preprocessor] ERROR: {exc}"
        print(msg)
        import traceback; traceback.print_exc()
        return {"preprocessor_done": False, "error": msg}

    print(f"[Node: Preprocessor] ✅ Done")
    return {
        "processed_features_npy":  output_npy,
        "processed_metadata_csv":  output_meta,
        "preprocessor_done":       True,
        "error":                   None,
    }
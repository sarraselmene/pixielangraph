"""
graph_nodes/lstm_node.py
========================
LangGraph node: BiLSTM engagement score inference.

Input:  processed_features.npy + processed_metadata.csv (from preprocessor_node)
Output: lstm_predictions.csv with columns:
    frame_id, track_id, engagement_score, risk_level, window_center_frame

Uses the PyTorch PixieBiLSTM model (train_lstm.py architecture):
  - Input:  (batch, window=54, features=26)
  - Output: engagement_score ∈ [0, 1]

Risk level mapping:
  engagement >= 0.65 → "low"
  engagement >= 0.35 → "medium"
  engagement <  0.35 → "high"
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lstm_node")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION (must match train_lstm.py)
# ══════════════════════════════════════════════════════════════════════════════

WINDOW_SIZE  = 54
STRIDE       = 5
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 256

DEFAULT_MODEL_PATH = "pixie_lstm_v1.pth"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (copied from train_lstm.py for self-contained inference)
# ══════════════════════════════════════════════════════════════════════════════

def _load_torch():
    """Lazy-load PyTorch to avoid import errors when not needed."""
    import torch
    import torch.nn as nn
    return torch, nn


class PixieBiLSTM:
    """
    Wrapper that lazily loads PyTorch and builds the model.
    Avoids module-level torch import which breaks multiprocessing.
    """

    @staticmethod
    def build_model(input_dim: int = 26):
        torch, nn = _load_torch()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(input_dim, HIDDEN_SIZE),
                    nn.LayerNorm(HIDDEN_SIZE),
                    nn.GELU(),
                )
                self.lstm = nn.LSTM(
                    input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS, batch_first=True,
                    bidirectional=True,
                    dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
                )
                lstm_out_dim = HIDDEN_SIZE * 2
                self.attn = nn.Sequential(
                    nn.Linear(lstm_out_dim, 32), nn.Tanh(), nn.Linear(32, 1),
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(lstm_out_dim),
                    nn.Linear(lstm_out_dim, 32), nn.GELU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(32, 1), nn.Sigmoid(),
                )

            def forward(self, x):
                x = self.input_proj(x)
                lstm_out, _ = self.lstm(x)
                attn_w = torch.softmax(self.attn(lstm_out), dim=1)
                context = (attn_w * lstm_out).sum(dim=1)
                score = self.head(context).squeeze(-1)
                return score, attn_w.squeeze(-1)

            def predict(self, x):
                with torch.no_grad():
                    scores, _ = self.forward(x)
                return scores

        return _Model()


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _risk_level(engagement: float) -> str:
    if engagement >= 0.65:
        return "low"
    elif engagement >= 0.35:
        return "medium"
    return "high"


def run_lstm_inference(
    features_npy: str,
    metadata_csv: str,
    model_path: str,
    encoders_path: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Full LSTM inference pipeline.

    Returns DataFrame with columns:
        frame_id, track_id, engagement_score, risk_level
    """
    torch, nn = _load_torch()
    t0 = time.perf_counter()

    log.info("=" * 58)
    log.info("  PIXIE — LSTM INFERENCE NODE (PyTorch BiLSTM)")
    log.info("=" * 58)

    # 1. Load features + metadata
    X_all = np.load(features_npy).astype(np.float32)
    meta  = pd.read_csv(metadata_csv)
    log.info(f"  Features: {X_all.shape}  Metadata: {len(meta)} rows")

    assert len(X_all) == len(meta), "Feature/metadata length mismatch"

    # 2. Load model
    with open(encoders_path, "rb") as f:
        enc_data = pickle.load(f)
    input_dim = enc_data["input_dim"]

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    log.info(f"  Device: {device}")

    model = PixieBiLSTM.build_model(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    log.info(f"  Model loaded: {model_path}")

    # 3. Build sliding windows PER TRACK (no bleeding)
    all_results = []
    track_ids = meta["track_id"].unique()

    for tid in sorted(track_ids):
        mask = meta["track_id"] == tid
        X_track = X_all[mask]
        frames_track = meta.loc[mask, "frame_id"].values
        n = len(X_track)

        if n < WINDOW_SIZE:
            log.warning(f"  Track {tid}: only {n} frames (need {WINDOW_SIZE}), skipping")
            continue

        # Build windows
        n_windows = (n - WINDOW_SIZE) // STRIDE + 1
        windows = np.empty((n_windows, WINDOW_SIZE, input_dim), dtype=np.float32)
        center_frames = np.empty(n_windows, dtype=np.int64)

        for i in range(n_windows):
            start = i * STRIDE
            end = start + WINDOW_SIZE
            windows[i] = X_track[start:end]
            center_frames[i] = frames_track[start + WINDOW_SIZE // 2]

        # Batch inference
        scores_list = []
        for b_start in range(0, n_windows, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, n_windows)
            batch = torch.from_numpy(windows[b_start:b_end]).to(device)
            preds = model.predict(batch).cpu().numpy()
            scores_list.append(preds)

        scores = np.concatenate(scores_list)

        for i in range(n_windows):
            all_results.append({
                "frame_id":         int(center_frames[i]),
                "track_id":         int(tid),
                "engagement_score": round(float(scores[i]), 4),
                "risk_level":       _risk_level(float(scores[i])),
            })

        log.info(f"  Track {tid}: {n_windows} windows, "
                 f"avg_engagement={scores.mean():.3f}")

    # 4. Save
    df_pred = pd.DataFrame(all_results)
    df_pred.to_csv(output_csv, index=False)
    elapsed = time.perf_counter() - t0
    log.info(f"  ✅ {len(df_pred)} predictions → {output_csv}  ({elapsed:.1f}s)")

    return df_pred


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE
# ══════════════════════════════════════════════════════════════════════════════

def run_lstm_node(state: dict) -> dict:
    """
    LangGraph node: Run BiLSTM engagement inference.

    Reads:   processed_features_npy, processed_metadata_csv, work_dir
    Produces: lstm_predictions_csv, lstm_done
    """
    work_dir     = state.get("work_dir", ".")
    features_npy = state.get("processed_features_npy", "")
    metadata_csv = state.get("processed_metadata_csv", "")

    print(f"\n{'='*60}")
    print(f"[Node: LSTM] BiLSTM Engagement Inference")
    print(f"{'='*60}")

    if not features_npy or not os.path.isfile(features_npy):
        msg = f"[LSTM] ERROR: features .npy not found → {features_npy}"
        print(msg)
        return {"lstm_done": False, "error": msg}

    model_path    = os.path.join(work_dir, DEFAULT_MODEL_PATH)
    encoders_path = os.path.join(work_dir, "encoders.pkl")
    output_csv    = os.path.join(work_dir, "lstm_predictions.csv")

    if not os.path.isfile(model_path):
        msg = f"[LSTM] ERROR: model not found → {model_path}"
        print(msg)
        return {"lstm_done": False, "error": msg}

    try:
        df_pred = run_lstm_inference(
            features_npy  = features_npy,
            metadata_csv  = metadata_csv,
            model_path    = model_path,
            encoders_path = encoders_path,
            output_csv    = output_csv,
        )
    except Exception as exc:
        msg = f"[LSTM] ERROR: {exc}"
        print(msg)
        import traceback; traceback.print_exc()
        return {"lstm_done": False, "error": msg}

    print(f"[Node: LSTM] ✅ Done — {len(df_pred)} predictions")
    return {
        "lstm_predictions_csv": output_csv,
        "lstm_done":            True,
        "error":                None,
    }
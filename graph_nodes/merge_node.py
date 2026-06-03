"""
graph_nodes/merge_node.py
==========================
Fusionne les sorties des nœuds Gaze, HeadPose et Body en un seul
vecteur de features normalisé, EXACTEMENT comme le fait
preprocessor.py (fonctions compute_deltas, compute_visibility_score,
compute_gaze_changes).

Ce nœud est la traduction frame-à-frame du pipeline batch du preprocessor.

Clés State consommées :
    landmarks, student_id, frame_id, gaze_state, _scaler, _face_db,
    buffer_dict

Clés State produites :
    feature_vector  np.ndarray (1, n_features)
    gaze_state      (mis à jour)
    _scaler         (chargé si None)
"""

from __future__ import annotations
import os
from collections import deque
from typing import Any, Dict

import numpy as np

from pixie_state import PixieState, FEATURE_COLS, WINDOW_SIZE

# ─── Encodages (mêmes maps que step1_generate_target_csv.py) ──────────────────
_CTX_ENC  = {"lecture": 0, "writing from board": 1, "group work": 2, "teacher left": 3}
_GAZE_H_ENC = {"Center": 0, "Left": -1, "Right": 1, "Board": 2}
_GAZE_V_ENC = {"Level": 0, "Up": 1, "Down": -1}
_POSE_ENC   = {"Up": 1, "Down": 0}
_TILT_ENC   = {"No-Tilt": 0, "Left": -1, "Right": 1}
_POST_ENC   = {"sitting": 1, "slouching": 0}

# ─── Historique inter-frame pour les deltas (par élève) ───────────────────────
_prev_features: Dict[str, Dict[str, float]] = {}


def _infer_context(frame_id: int, total_frames: int) -> str:
    """
    Infère le contexte pédagogique à partir de la position temporelle
    dans la vidéo (heuristique basique si pas d'annotation disponible).
    """
    if total_frames <= 0:
        return "lecture"
    ratio = frame_id / total_frames
    if ratio < 0.25:
        return "lecture"
    elif ratio < 0.50:
        return "writing from board"
    elif ratio < 0.75:
        return "group work"
    else:
        return "teacher left"


def run_merge_node(state: PixieState) -> PixieState:
    """
    Fusionne landmarks → vecteur de features normalisé (FEATURE_COLS).
    Implémente les mêmes calculs que preprocessor.compute_deltas()
    et compute_visibility_score() mais en mode online (frame par frame).
    """
    landmarks   = state.get("landmarks") or {}
    student_id  = state.get("student_id", "unknown")
    frame_id    = state.get("frame_id", 0)
    total_frames = state.get("total_frames", 0)
    gaze_state  = dict(state.get("gaze_state") or {})

    # ── Extraction des sous-dictionnaires ─────────────────────────────────────
    gaze  = landmarks.get("gaze",     {})
    head  = landmarks.get("head",     {})
    body  = landmarks.get("body",     {})
    beh   = landmarks.get("behavior", {})
    ctx   = landmarks.get("context", _infer_context(frame_id, total_frames))

    # ── Features brutes ───────────────────────────────────────────────────────
    context_enc    = _CTX_ENC.get(ctx, 0)
    gaze_stability = float(gaze.get("gaze_stability",      0.8))
    eye_head_div   = float(gaze.get("eye_head_divergence", 0.3))
    genuine_smile  = float(beh.get("genuine_smile",        0.0))
    fatigue_ind    = float(beh.get("fatigue_indicator",    0.0))
    yawning        = float(beh.get("yawning",              0.0))
    talking_flag   = float(beh.get("talking_flag",         0.0))
    expressiveness = float(beh.get("expressiveness_score", 0.35))
    hand_raised    = float(body.get("hand_raised",         0.0))
    disp_score     = float(body.get("disp_score",          0.0))

    pose_label_enc = _POSE_ENC.get(head.get("pose_label", "Up"), 1)
    tilt_label_enc = _TILT_ENC.get(head.get("tilt_label", "No-Tilt"), 0)
    gaze_h_enc     = _GAZE_H_ENC.get(gaze.get("gaze_h_label", "Center"), 0)
    gaze_v_enc     = _GAZE_V_ENC.get(gaze.get("gaze_v_label", "Level"), 0)
    posture_enc    = _POST_ENC.get(body.get("posture", "sitting"), 1)

    # ── Deltas inter-frame (comme preprocessor.compute_deltas) ────────────────
    prev = _prev_features.get(student_id, {})
    delta_yaw    = abs(float(head.get("yaw",   0.0)) - prev.get("yaw",   0.0))
    delta_pitch  = abs(float(head.get("pitch", 0.0)) - prev.get("pitch", 0.0))
    delta_gaze_x = abs(float(gaze.get("gaze_angle_x", 0.0)) - prev.get("gaze_x", 0.0))

    _prev_features[student_id] = {
        "yaw":    float(head.get("yaw",   0.0)),
        "pitch":  float(head.get("pitch", 0.0)),
        "gaze_x": float(gaze.get("gaze_angle_x", 0.0)),
    }

    # ── Gaze changes rolling (comme preprocessor.compute_gaze_changes) ────────
    if student_id not in gaze_state:
        gaze_state[student_id] = {
            "prev_gaze_h": gaze.get("gaze_h_label", "Center"),
            "deque":       deque(maxlen=WINDOW_SIZE),
        }
    gs = gaze_state[student_id]
    curr_gaze_h = gaze.get("gaze_h_label", "Center")
    changed = 1 if curr_gaze_h != gs["prev_gaze_h"] else 0
    gs["deque"].append(changed)
    gs["prev_gaze_h"] = curr_gaze_h
    gaze_changes_50f = float(sum(gs["deque"]))
    gaze_state[student_id] = gs

    # ── Visibility score (comme preprocessor.compute_visibility_score) ─────────
    visibility_score = (
        0.30 * float(pose_label_enc)
      + 0.25 * (1.0 if gaze_v_enc == 0 else 0.5)
      + 0.25 * min(1.0, gaze_stability)
      + 0.20 * float(posture_enc)
    )
    visibility_score = round(min(1.0, max(0.0, visibility_score)), 4)

    # ── Assemblage du vecteur dans l'ordre de FEATURE_COLS ────────────────────
    raw_vec = np.array([
        context_enc,
        gaze_stability,
        eye_head_div,
        genuine_smile,
        fatigue_ind,
        yawning,
        talking_flag,
        expressiveness,
        hand_raised,
        disp_score,
        delta_yaw,
        delta_pitch,
        delta_gaze_x,
        gaze_changes_50f,
        visibility_score,
        float(pose_label_enc),
        float(tilt_label_enc),
        float(gaze_h_enc),
        float(gaze_v_enc),
        float(posture_enc),
    ], dtype=np.float32)

    # ── Normalisation via scaler (chargé depuis scaler_path) ──────────────────
    scaler = state.get("_scaler")
    if scaler is None:
        scaler_path = state.get("scaler_path", "scaler.pkl")
        if os.path.isfile(scaler_path):
            import joblib
            scaler = joblib.load(scaler_path)
            print(f"[merge] Scaler chargé depuis {scaler_path}")
        else:
            print(f"[merge] WARN: scaler.pkl introuvable → pas de normalisation")

    if scaler is not None:
        try:
            raw_vec = scaler.transform(raw_vec.reshape(1, -1))[0]
        except Exception as exc:
            print(f"[merge] WARN scaler.transform : {exc}")

    feature_vector = raw_vec.reshape(1, -1).astype(np.float32)

    return {
        **state,
        "feature_vector": feature_vector,
        "gaze_state":     gaze_state,
        "_scaler":        scaler,
    }

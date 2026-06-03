"""
pixie_state.py
==============
Schéma d'état central du graphe LangGraph offline Pixie.
Toutes les clés sont optionnelles (total=False).
"""
from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional, Union

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


def _add_errors(
    left:  Optional[List[str]],
    right: Union[Optional[List[str]], str, None],
) -> List[str]:
    result = list(left) if left else []
    if right is None:
        return result
    if isinstance(right, str):
        result.append(right)
    elif isinstance(right, list):
        result.extend(right)
    return result


# ─── Constantes partagées ─────────────────────────────────────────────────────

BEHAVIOR_LABELS: Dict[int, str] = {
    0: "ENGAGÉ",
    1: "DISTRAIT",
    2: "AGITÉ",
    3: "FATIGUÉ",
    4: "ANXIÉTÉ_SOCIALE",
}

# Couleurs BGR pour OpenCV
BEHAVIOR_COLORS: Dict[int, tuple] = {
    0: (50,  200, 50),    # vert   — engagé
    1: (30,  165, 255),   # orange — distrait
    2: (0,   80,  255),   # rouge  — agité
    3: (200, 100, 50),    # bleu foncé — fatigué
    4: (160, 60,  200),   # violet — anxiété sociale
}

# Colonnes numériques dans l'ordre attendu par le scaler / LSTM
FEATURE_COLS: List[str] = [
    "context_enc", "gaze_stability", "eye_head_divergence",
    "genuine_smile", "fatigue_indicator", "yawning", "talking_flag",
    "expressiveness_score", "hand_raised", "disp_score",
    "delta_yaw", "delta_pitch", "delta_gaze_x", "gaze_changes_50f",
    "visibility_score", "pose_label_enc", "tilt_label_enc",
    "gaze_h_label_enc", "gaze_v_label_enc", "posture_enc",
]

WINDOW_SIZE = 50


class PixieState(TypedDict, total=False):

    # ── Inputs de session ────────────────────────────────────────────────────
    video_path:      str
    work_dir:        str
    mapping_csv:     str   # mapping_faces.csv  → face_id: student_name
    model_path:      str   # best_lstm_model.keras
    scaler_path:     str   # scaler.pkl
    clusters_path:   str   # clusters.pkl (optionnel)
    window_size:     int

    # ── Frame courante ───────────────────────────────────────────────────────
    frame:           Any   # np.ndarray BGR
    frame_id:        int
    total_frames:    int

    # ── Sorties des modèles de vision ─────────────────────────────────────────
    landmarks: Dict[str, Any]
    # landmarks["gaze"]     → gaze_angle_x/y, gaze_h/v_label, gaze_stability,
    #                          eye_head_divergence
    # landmarks["head"]     → yaw, pitch, roll, pose_label, tilt_label
    # landmarks["body"]     → posture, disp_score, hand_raised
    # landmarks["behavior"] → genuine_smile, fatigue_indicator, yawning,
    #                          talking_flag, expressiveness_score
    # landmarks["context"]  → str (lecture / writing from board / …)

    # ── Identité ─────────────────────────────────────────────────────────────
    student_id:      str
    face_encoding:   Any   # vecteur 128-d
    face_location:   Any   # (top, right, bottom, left)

    # ── Feature vector normalisé ──────────────────────────────────────────────
    feature_vector:  Any   # np.ndarray (1, len(FEATURE_COLS))

    # ── Buffer multi-élèves {student_id: deque(maxlen=window_size)} ──────────
    buffer_dict:     Dict[str, Any]

    # ── Compteur gaze_changes rolling {student_id: {"prev_gaze_h", "deque"}} ─
    gaze_state:      Dict[str, Any]

    # ── Dernière prédiction ───────────────────────────────────────────────────
    prediction: Dict[str, Any]
    # { cluster: int, label: str, confidence: float,
    #   scores: list[float], student_id: str, frame_id: int }

    # ── Historique dashboard (60 dernières prédictions) ───────────────────────
    history: List[Dict[str, Any]]

    # ── Modèles cachés entre les frames ──────────────────────────────────────
    _lstm_model:     Any   # keras.Model
    _scaler:         Any   # StandardScaler
    _face_db:        Any   # list[(encoding, name)]
    _video_capture:  Any   # cv2.VideoCapture
    _video_writer:   Any   # cv2.VideoWriter

    # ── Outputs ──────────────────────────────────────────────────────────────
    output_video_path: str
    session_csv:       str
    frame_done:        bool   # True quand toutes les frames sont traitées

    # ── Erreurs accumulées ────────────────────────────────────────────────────
    error: Annotated[List[str], _add_errors]

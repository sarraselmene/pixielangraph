"""
╔══════════════════════════════════════════════════════════════════╗
║              PIXIE — Behavioural Analysis Pipeline               ║
║         Extraction Script v2.0  |  macOS / Apple Silicon        ║
╠══════════════════════════════════════════════════════════════════╣
║  Contexte : 1-2 élèves · caméra fixe · distance ≤ 3 m          ║
╠══════════════════════════════════════════════════════════════════╣
║  Modèles v2 (100 % Python, zéro binaire externe) :              ║
║    • YOLOv11-pose  → corps + tracking (inchangé)                ║
║    • YOLOv11-face  → détection visage (inchangé)                ║
║    • WHENet        → head pose  (remplace SixDRepNet)           ║
║    • py-feat       → Action Units  (remplace OpenFace)          ║
║    • L2CS-Net      → Gaze  (remplace OpenFace gaze)             ║
╠══════════════════════════════════════════════════════════════════╣
║  Installation :                                                  ║
║    pip install ultralytics py-feat l2cs                         ║
║    pip install onnxruntime   # ou onnxruntime-silicon sur M1/M2 ║
║    # WHENet : voir section 7 ci-dessous                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Sorties :                                                       ║
║    raw_body_multi.csv                                            ║
║    raw_head_pose_multi.csv                                       ║
║    raw_action_units_multi.csv                                    ║
║    raw_gaze_multi.csv                                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────
import sys
import cv2
import csv
import logging
import argparse
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────
# 1. CONFIGURATION UTILISATEUR  ← seule section à éditer
# ──────────────────────────────────────────────────────────────────

# Poids YOLO (téléchargés automatiquement par Ultralytics au 1er lancement)
YOLO_POSE_WEIGHTS = "yolo11n-pose.pt"
YOLO_FACE_WEIGHTS = "yolov11m-face.pt"   # ou "yolov8n-face.pt"

# WHENet — chemin vers le fichier ONNX
# Téléchargement : https://github.com/Ascend-Research/HeadPoseEstimation-WHENet
# Fichier attendu : WHENet.onnx  (ou WHENet_V2.onnx)
WHENET_ONNX = "WHENet.onnx"

# L2CS-Net — chemin vers les poids (.pkl)
# Téléchargement :
#   pip install gdown
#   gdown "https://drive.google.com/uc?id=1Dby7_OEuPAGCLkCBJSOOYMkFE8mUErFl" -O l2cs_weights.pkl
L2CS_WEIGHTS = "l2cs_weights.pkl"

# Répertoire de sortie
OUTPUT_DIR = Path("pixie_output")

# ──────────────────────────────────────────────────────────────────
# 2. CONSTANTES
# ──────────────────────────────────────────────────────────────────
EXPAND_RATIO     = 0.40   # expansion crop visage pour WHENet / py-feat
MIN_FACE_SIZE    = 20     # px — ignorer les visages trop petits (vidéo basse résolution)
UPSCALE_FACTOR   = 3      # upscale frames avant détection (288px → 864px)
ID_BUFFER_FRAMES = 30     # fenêtre de stabilisation des track IDs

# Indices keypoints COCO
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10

# AUs produites par py-feat (FAb-Net) — mêmes colonnes qu'OpenFace
AU_INTENSITY = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
AU_BINARY = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]

# ──────────────────────────────────────────────────────────────────
# 3. LOGGING
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pixie")

# ──────────────────────────────────────────────────────────────────
# 4. DEVICE  (MPS → CUDA → CPU)
# ──────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log.info("🍎 Apple Silicon MPS détecté")
        return torch.device("mps")
    if torch.cuda.is_available():
        log.info("⚡ CUDA GPU détecté")
        return torch.device("cuda")
    log.info("💻 CPU uniquement")
    return torch.device("cpu")

DEVICE = get_device()

# ──────────────────────────────────────────────────────────────────
# 5. KALMAN FILTER 2D (stabilisation keypoints)
# ──────────────────────────────────────────────────────────────────
class KalmanFilter2D:
    """Filtre de Kalman à vitesse constante pour un point 2D."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],
                                                 [0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self._init = False

    def update(self, x: float, y: float) -> tuple[float, float]:
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._init:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self._init = True
        self.kf.correct(meas)
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

# ──────────────────────────────────────────────────────────────────
# 6. TRACK-ID BUFFER (fenêtre 30 frames)
# ──────────────────────────────────────────────────────────────────
class TrackIDBuffer:
    def __init__(self, max_frames: int = ID_BUFFER_FRAMES):
        self._buf: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_frames)
        )

    def update(self, track_id: int, frame_id: int) -> int:
        self._buf[track_id].append(frame_id)
        return track_id

# ──────────────────────────────────────────────────────────────────
# 7. WHENet — HEAD POSE ESTIMATOR
#
#  WHENet couvre ±180° en yaw (parfait pour profils en classe) avec
#  une précision ±3° à courte distance. Il tourne via ONNX Runtime,
#  compatible MPS sur Apple Silicon via onnxruntime-silicon.
#
#  Téléchargement du modèle ONNX :
#    https://github.com/Ascend-Research/HeadPoseEstimation-WHENet/releases
#    → WHENet.onnx  (ou WHENet_V2.onnx, meilleur)
#
#  Installation :
#    pip install onnxruntime          # CPU / CUDA
#    pip install onnxruntime-silicon  # Apple M-series
# ──────────────────────────────────────────────────────────────────
class WHENetEstimator:
    """Head pose (pitch, yaw, roll) via WHENet ONNX."""

    # Normalisation ImageNet
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, onnx_path: str):
        self._session = None
        self._path    = onnx_path
        self._failed  = False   # ← évite le spam de warnings

    def _lazy_load(self):
        if self._session is not None or self._failed:
            return
        try:
            import onnxruntime as ort
            # Providers : CoreML pour MPS sur macOS, puis CPU
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # silence ONNX verbosity
            self._session = ort.InferenceSession(
                self._path, sess_options=opts, providers=providers
            )
            self._input_name = self._session.get_inputs()[0].name
            log.info("WHENet chargé ✓  (providers: %s)",
                     self._session.get_providers())
        except Exception as e:
            log.warning("WHENet indisponible (%s). Head-pose sera vide.", e)
            self._session = None
            self._failed  = True   # ← ne plus retenter

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """BGR → (1,3,224,224) float32 normalisé."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        rgb = (rgb - self._MEAN) / self._STD
        return rgb.transpose(2, 0, 1)[np.newaxis]   # (1,3,224,224)

    def predict(self, face_bgr: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Retourne (pitch_deg, yaw_deg, roll_deg) ou None."""
        self._lazy_load()
        if self._session is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            inp  = self._preprocess(face_bgr)
            outs = self._session.run(None, {self._input_name: inp})

            # WHENet peut retourner soit :
            #   - 3 sorties séparées (yaw, pitch, roll) chacune shape (1,) ou (1,1)
            #   - 1 seule sortie shape (1,3) selon la version du modèle
            def _extract(arr) -> float:
                """Aplatit n'importe quelle forme numpy en scalaire float."""
                return float(np.array(arr).flatten()[0])

            if len(outs) == 3:
                # Format standard : [yaw_arr, pitch_arr, roll_arr]
                yaw   = _extract(outs[0])
                pitch = _extract(outs[1])
                roll  = _extract(outs[2])
            elif len(outs) == 1:
                # Format alternatif : [[yaw, pitch, roll]]
                flat  = np.array(outs[0]).flatten()
                yaw, pitch, roll = float(flat[0]), float(flat[1]), float(flat[2])
            else:
                log.warning("WHENet : format de sortie inattendu (%d tenseurs)", len(outs))
                return None

            log.debug("WHENet raw → yaw=%.1f pitch=%.1f roll=%.1f", yaw, pitch, roll)
            return pitch, yaw, roll
        except Exception as e:
            log.warning("WHENet predict error: %s", e)
            return None

# ──────────────────────────────────────────────────────────────────
# 8. py-feat — ACTION UNITS ESTIMATOR
#
#  py-feat embarque FAb-Net, un modèle entraîné sur DISFA/AffectNet
#  qui produit les mêmes 17 AUs (intensité _r) que OpenFace.
#  Les colonnes binaires (_c) sont dérivées par seuillage (>0.5).
#
#  Installation : pip install py-feat
#  Docs        : https://py-feat.org
# ──────────────────────────────────────────────────────────────────
class PyFeatAUEstimator:
    """Extraction des Action Units via py-feat (FAb-Net)."""

    # Seuil de binarisation pour les colonnes _c
    _BINARY_THRESHOLD = 0.5

    # Mapping py-feat → noms AU standard
    # py-feat retourne : AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10,
    #                    AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45
    _AU_MAP = {
        "AU1": "AU01_r", "AU2": "AU02_r", "AU4": "AU04_r",
        "AU5": "AU05_r", "AU6": "AU06_r", "AU7": "AU07_r",
        "AU9": "AU09_r", "AU10":"AU10_r", "AU12":"AU12_r",
        "AU14":"AU14_r", "AU15":"AU15_r", "AU17":"AU17_r",
        "AU20":"AU20_r", "AU23":"AU23_r", "AU25":"AU25_r",
        "AU26":"AU26_r", "AU45":"AU45_r",
    }

    def __init__(self):
        self._detector = None
        self._failed   = False   # ← évite le spam de warnings

    def _lazy_load(self):
        if self._detector is not None or self._failed:
            return
        try:
            from feat import Detector  # pip install py-feat  (package = py-feat, module = feat)
            # au_model="svm" est le plus robuste sur petites résolutions
            self._detector = Detector(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                au_model="svm",
                emotion_model="resmasknet",
                facepose_model="img2pose",
                device=str(DEVICE) if DEVICE.type != "mps" else "cpu",
            )
            log.info("py-feat (FAb-Net / SVM AU) chargé ✓")
        except Exception as e:
            log.warning("py-feat indisponible (%s). AUs seront vides.", e)
            self._detector = None
            self._failed   = True   # ← ne plus retenter

    def predict(
        self, face_bgr: np.ndarray
    ) -> Optional[dict[str, float]]:
        """
        Retourne un dict {AU01_r: val, …, AU01_c: 0/1, …}
        ou None si échec.

        py-feat v0.6+ attend soit :
          - un chemin fichier image (le plus fiable)
          - un array numpy HxWx3 RGB avec un visage bien cadré
        On utilise un fichier temp pour garantir la compatibilité.
        """
        self._lazy_load()
        if self._detector is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            import tempfile, os

            # ── Validation taille minimale ─────────────────────
            h, w = face_bgr.shape[:2]
            if h < 48 or w < 48:
                log.debug("py-feat: crop trop petit (%dx%d), ignoré", w, h)
                return None

            # ── Upscale si visage trop petit (< 112px) ─────────
            # FAb-Net est entraîné sur des faces ~112px min
            if h < 112 or w < 112:
                scale = 112 / min(h, w)
                face_bgr = cv2.resize(
                    face_bgr,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC
                )

            # ── Sauvegarde temp + detect_image ─────────────────
            # py-feat est plus robuste avec un chemin fichier qu'un array
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tmp_path = tf.name
            cv2.imwrite(tmp_path, face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            try:
                result = self._detector.detect_image(tmp_path)
            finally:
                os.unlink(tmp_path)   # nettoyage immédiat

            if result is None:
                log.debug("py-feat: detect_image a retourné None")
                return None

            # ── Extraction AUs ─────────────────────────────────
            # py-feat retourne un FexCollection avec attribut .aus
            # Les noms de colonnes varient selon la version : "AU1" ou "AU01"
            aus_df = result.aus if hasattr(result, "aus") else None
            if aus_df is None or len(aus_df) == 0:
                log.debug("py-feat: aucune AU détectée dans le crop")
                return None

            aus_row = aus_df.iloc[0]
            log.debug("py-feat colonnes disponibles: %s", list(aus_row.index))

            out: dict[str, float] = {}
            for feat_key, csv_key in self._AU_MAP.items():
                # Essayer "AU1" puis "AU01" (différences selon version py-feat)
                val = aus_row.get(feat_key,
                      aus_row.get(feat_key.replace("AU", "AU0").lstrip("AU0").rjust(2,"0"),
                      0.0))
                # Cherche aussi la clé exacte dans l'index
                if feat_key not in aus_row.index:
                    # Fallback : cherche par suffixe numérique
                    num = feat_key[2:]  # ex "1", "12"
                    candidates = [c for c in aus_row.index
                                  if c.upper().replace("AU","").lstrip("0") == num.lstrip("0")]
                    val = float(aus_row[candidates[0]]) if candidates else 0.0
                else:
                    val = float(aus_row[feat_key])

                out[csv_key] = round(val, 4)
                bin_key = csv_key.replace("_r", "_c")
                out[bin_key] = 1 if val >= self._BINARY_THRESHOLD else 0

            # AU28_c (lip suck) — non produit par FAb-Net
            out["AU28_c"] = 0
            return out

        except Exception as e:
            log.warning("py-feat predict error: %s", e)
            return None

# ──────────────────────────────────────────────────────────────────
# 9. L2CS-Net — GAZE ESTIMATOR
#
#  L2CS-Net est entraîné sur MPIIGaze + Gaze360.
#  Il produit yaw/pitch angulaires → on les mappe sur les colonnes
#  gaze_angle_x / gaze_angle_y du CSV original.
#  Les vecteurs 3D gaze_0/gaze_1 sont reconstruits depuis les angles.
#
#  Installation : pip install l2cs
#  Docs        : https://github.com/edavalosanaya/L2CS-Net
# ──────────────────────────────────────────────────────────────────
class L2CSGazeEstimator:
    """Estimation du regard (gaze) via L2CS-Net."""

    def __init__(self):
        self._pipeline = None
        self._failed   = False   # ← évite le spam de warnings

    def _lazy_load(self):
        if self._pipeline is not None or self._failed:
            return
        try:
            from l2cs import Pipeline
            import torch
            from pathlib import Path as _Path

            weights_path = _Path(L2CS_WEIGHTS)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Poids L2CS introuvables : {weights_path.resolve()}\n"
                    "  → Télécharger avec :\n"
                    "    pip install gdown\n"
                    '    gdown "https://drive.google.com/uc?id=1Dby7_OEuPAGCLkCBJSOOYMkFE8mUErFl" -O l2cs_weights.pkl'
                )

            # L2CS supporte cpu / cuda ; MPS via CPU fallback
            device = torch.device("cpu")  # plus stable sur macOS M-series
            self._pipeline = Pipeline(
                weights=str(weights_path.resolve()),
                arch="ResNet50",
                device=device,
            )
            log.info("L2CS-Net chargé ✓")
        except Exception as e:
            log.warning("L2CS-Net indisponible (%s). Gaze sera vide.", e)
            self._pipeline = None
            self._failed   = True   # ← ne plus retenter

    @staticmethod
    def _angles_to_vector(pitch_rad: float, yaw_rad: float) -> tuple[float,float,float]:
        """
        Convertit (pitch, yaw) en vecteur unitaire 3D
        convention : x=droite, y=bas, z=profondeur (vers la caméra).
        """
        x =  np.cos(pitch_rad) * np.sin(yaw_rad)
        y = -np.sin(pitch_rad)
        z = -np.cos(pitch_rad) * np.cos(yaw_rad)
        return float(x), float(y), float(z)

    def predict(
        self, face_bgr: np.ndarray
    ) -> Optional[dict[str, float]]:
        """
        Retourne un dict avec toutes les colonnes gaze du CSV,
        ou None si échec.
        """
        self._lazy_load()
        if self._pipeline is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            results = self._pipeline.step(face_bgr)
            if results is None or len(results.pitch) == 0:
                return None

            pitch_rad = float(results.pitch[0])
            yaw_rad   = float(results.yaw[0])

            # Vecteur gaze (les deux yeux supposés identiques à cette distance)
            gx, gy, gz = self._angles_to_vector(pitch_rad, yaw_rad)

            return {
                "gaze_0_x":    round(gx, 5),
                "gaze_0_y":    round(gy, 5),
                "gaze_0_z":    round(gz, 5),
                "gaze_1_x":    round(gx, 5),   # même vecteur pour les 2 yeux
                "gaze_1_y":    round(gy, 5),
                "gaze_1_z":    round(gz, 5),
                "gaze_angle_x": round(float(np.degrees(yaw_rad)),   4),
                "gaze_angle_y": round(float(np.degrees(pitch_rad)), 4),
            }
        except Exception as e:
            log.debug("L2CS predict error: %s", e)
            return None

# ──────────────────────────────────────────────────────────────────
# 10. SCHÉMAS CSV (colonnes exactes Pixie)
# ──────────────────────────────────────────────────────────────────
BODY_COLS = [
    "frame_id", "timestamp_sec", "track_id", "landmark_idx",
    "x", "y", "visibility",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "global_desk_y",
]
HEAD_POSE_COLS = [
    "frame_id", "timestamp_sec", "track_id", "pitch", "yaw", "roll",
]
AU_COLS = (
    ["frame_id", "track_id", "confidence", "success"]
    + AU_INTENSITY
    + AU_BINARY
)
GAZE_COLS = [
    "frame_id", "track_id", "confidence", "success",
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]

def open_csv_writer(path: Path, fieldnames: list) -> tuple:
    fh = open(path, "w", newline="", encoding="utf-8")
    w  = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    return fh, w

# ──────────────────────────────────────────────────────────────────
# 11. HELPERS GÉOMÉTRIE
# ──────────────────────────────────────────────────────────────────
def shoulder_width(kps: np.ndarray) -> float:
    ls = kps[KP_LEFT_SHOULDER]
    rs = kps[KP_RIGHT_SHOULDER]
    d  = float(np.linalg.norm(ls[:2] - rs[:2]))
    return d if d > 1.0 else 1.0

def expand_bbox(
    x1: int, y1: int, x2: int, y2: int,
    ratio: float, W: int, H: int
) -> tuple[int, int, int, int]:
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w,  h  = (x2 - x1) * (1 + ratio), (y2 - y1) * (1 + ratio)
    return (
        max(0, int(cx - w / 2)),
        max(0, int(cy - h / 2)),
        min(W, int(cx + w / 2)),
        min(H, int(cy + h / 2)),
    )

def match_face_to_track(
    face_cx: float, face_cy: float,
    track_centroids: dict[int, np.ndarray],
) -> int:
    """Retourne le track_id le plus proche du centre du visage détecté."""
    if not track_centroids:
        return -1
    fc = np.array([face_cx, face_cy])
    return min(track_centroids, key=lambda t: np.linalg.norm(track_centroids[t] - fc))

# ──────────────────────────────────────────────────────────────────
# 12. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────
def run_extraction(video_path: str, output_dir: Path) -> None:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement des modèles ────────────────────────────────────
    log.info("Chargement YOLOv11-pose …")
    pose_model = YOLO(YOLO_POSE_WEIGHTS)

    log.info("Chargement YOLOv11-face …")
    face_model = YOLO(YOLO_FACE_WEIGHTS)

    log.info("Initialisation WHENet …")
    whenet = WHENetEstimator(WHENET_ONNX)

    log.info("Initialisation py-feat …")
    pyfeat = PyFeatAUEstimator()

    log.info("Initialisation L2CS-Net …")
    l2cs = L2CSGazeEstimator()

    # Chargement paresseux — déclenché à la première frame
    # (évite le délai de démarrage visible à l'écran)

    # ── Ouverture vidéo ───────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Impossible d'ouvrir : %s", video_path)
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Dimensions upscalées (utilisées pour détection visage + crop)
    UW = W * UPSCALE_FACTOR
    UH = H * UPSCALE_FACTOR
    log.info("Vidéo : %dx%d @ %.1f fps — %d frames", W, H, fps, total_frames)
    log.info("Upscale détection : %dx%d → %dx%d (×%d)", W, H, UW, UH, UPSCALE_FACTOR)

    # ── Ouverture des CSV ─────────────────────────────────────────
    fh_body, w_body = open_csv_writer(output_dir / "raw_body_multi.csv",      BODY_COLS)
    fh_hp,   w_hp   = open_csv_writer(output_dir / "raw_head_pose_multi.csv", HEAD_POSE_COLS)
    fh_au,   w_au   = open_csv_writer(output_dir / "raw_action_units_multi.csv", AU_COLS)
    fh_gz,   w_gz   = open_csv_writer(output_dir / "raw_gaze_multi.csv",      GAZE_COLS)

    # ── État par track ────────────────────────────────────────────
    kalman_filters: dict[int, list[KalmanFilter2D]] = {}
    id_buffer = TrackIDBuffer()
    wrist_y_history: list[float] = []

    # ── Sélection device YOLO (MPS non supporté via str sur certaines versions) ──
    yolo_device = "mps" if DEVICE.type == "mps" else str(DEVICE)

    # ── Boucle frame par frame ────────────────────────────────────
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = frame_id / fps

        # ── Upscale frame pour améliorer la détection sur vidéo basse résolution ──
        # YOLO et py-feat fonctionnent mieux sur des visages ≥ 80px
        # On upscale avant détection puis on rescale les coordonnées en retour
        frame_up = cv2.resize(
            frame, (UW, UH), interpolation=cv2.INTER_CUBIC
        )
        scale_inv = 1.0 / UPSCALE_FACTOR   # pour ramener les coords à l'espace original

        # ════════════════════════════════════════════════════════
        # A. CORPS — YOLO Pose + Tracking (sur frame upscalée)
        # ════════════════════════════════════════════════════════
        pose_results = pose_model.track(
            frame_up,
            persist=True,
            verbose=False,
            device=yolo_device,
        )

        # Accumulation des Y de poignets pour global_desk_y
        # Les coords sont dans l'espace upscalé → rescale vers original
        if pose_results and pose_results[0].keypoints is not None:
            for kps in pose_results[0].keypoints.data.cpu().numpy():
                for wi in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
                    if kps[wi, 2] > 0.5:
                        # Rescale Y vers espace original
                        wrist_y_history.append(float(kps[wi, 1]) * scale_inv)

        global_desk_y = float(np.median(wrist_y_history)) if wrist_y_history else 0.0

        # Centroïdes des tracks pour le matching visage→personne (espace upscalé)
        track_centroids: dict[int, np.ndarray] = {}

        if (
            pose_results
            and pose_results[0].boxes is not None
            and pose_results[0].keypoints is not None
        ):
            boxes_data = pose_results[0].boxes
            kps_data   = pose_results[0].keypoints.data.cpu().numpy()

            ids  = (
                boxes_data.id.cpu().numpy().astype(int)
                if boxes_data.id is not None
                else np.arange(len(boxes_data))
            )
            xyxy = boxes_data.xyxy.cpu().numpy()

            for pi, raw_tid in enumerate(ids):
                track_id = int(id_buffer.update(int(raw_tid), frame_id))
                kps      = kps_data[pi]

                # Rescale kps et bbox vers espace original pour le CSV
                kps_orig = kps.copy()
                kps_orig[:, :2] *= scale_inv
                sw = shoulder_width(kps_orig)

                bx1, by1, bx2, by2 = [v * scale_inv for v in xyxy[pi].tolist()]

                # Centroïde dans l'espace UPSCALÉ (pour crop visage)
                track_centroids[track_id] = np.array(
                    [(xyxy[pi][0] + xyxy[pi][2]) / 2,
                     (xyxy[pi][1] + xyxy[pi][3]) / 2]
                )

                # Initialisation Kalman si nouveau track
                if track_id not in kalman_filters:
                    kalman_filters[track_id] = [KalmanFilter2D() for _ in range(17)]

                # Écriture keypoints (coords dans espace original, normalisées par SW)
                for lm in range(17):
                    rx, ry, vis = kps_orig[lm]
                    kx, ky = kalman_filters[track_id][lm].update(float(rx), float(ry))
                    w_body.writerow({
                        "frame_id":      frame_id,
                        "timestamp_sec": round(ts, 4),
                        "track_id":      track_id,
                        "landmark_idx":  lm,
                        "x":             round(kx / sw, 5),
                        "y":             round(ky / sw, 5),
                        "visibility":    round(float(vis), 4),
                        "bbox_x1":       round(bx1, 2),
                        "bbox_y1":       round(by1, 2),
                        "bbox_x2":       round(bx2, 2),
                        "bbox_y2":       round(by2, 2),
                        "global_desk_y": round(global_desk_y, 2),
                    })

        # ════════════════════════════════════════════════════════
        # B. VISAGE — YOLO Face sur frame_up → crop → WHENet + py-feat + L2CS
        # Toute la détection et le crop se font sur frame_up (upscalée)
        # ════════════════════════════════════════════════════════
        TARGET_FACE_PX = 224
        AU_FACE_PX     = 160

        face_results = face_model(frame_up, verbose=False, device=yolo_device)

        if face_results and face_results[0].boxes is not None:
            for fi, fb in enumerate(face_results[0].boxes.xyxy.cpu().numpy()):
                fx1, fy1, fx2, fy2 = fb.astype(int)
                fw = fx2 - fx1
                fh_px = fy2 - fy1

                # ── Filtre (dans l'espace upscalé) ──────────────────
                if fw < MIN_FACE_SIZE or fh_px < MIN_FACE_SIZE:
                    log.debug("Frame %d visage #%d ignoré %dx%d (upscalé)",
                              frame_id, fi, fw, fh_px)
                    continue

                log.debug("Frame %d visage #%d détecté %dx%d (upscalé)",
                          frame_id, fi, fw, fh_px)

                fc_x = (fx1 + fx2) / 2
                fc_y = (fy1 + fy2) / 2
                tid  = match_face_to_track(fc_x, fc_y, track_centroids)

                # ── Crop carré sur frame_up ──────────────────────────
                side = max(fw, fh_px)
                cx, cy = int(fc_x), int(fc_y)
                half = int(side * (1 + EXPAND_RATIO) / 2)
                sq_x1 = max(0, cx - half)
                sq_y1 = max(0, cy - half)
                sq_x2 = min(UW, cx + half)
                sq_y2 = min(UH, cy + half)

                square_crop = frame_up[sq_y1:sq_y2, sq_x1:sq_x2]
                if square_crop.size == 0:
                    continue

                face_224 = cv2.resize(
                    square_crop, (TARGET_FACE_PX, TARGET_FACE_PX),
                    interpolation=cv2.INTER_CUBIC
                )

                # ── Crop rect sur frame_up pour py-feat ──────────────
                ex1 = max(0,   int(fx1 - fw    * EXPAND_RATIO))
                ey1 = max(0,   int(fy1 - fh_px * EXPAND_RATIO))
                ex2 = min(UW,  int(fx2 + fw    * EXPAND_RATIO))
                ey2 = min(UH,  int(fy2 + fh_px * EXPAND_RATIO))
                face_rect = frame_up[ey1:ey2, ex1:ex2]
                if face_rect.size == 0:
                    face_rect = face_224

                rh, rw = face_rect.shape[:2]
                if rw < AU_FACE_PX or rh < AU_FACE_PX:
                    sc = AU_FACE_PX / min(rw, rh)
                    face_rect = cv2.resize(
                        face_rect, (int(rw * sc), int(rh * sc)),
                        interpolation=cv2.INTER_CUBIC
                    )

                log.debug("Frame %d tid=%d face_224=%dx%d face_rect=%dx%d",
                          frame_id, tid,
                          face_224.shape[1], face_224.shape[0],
                          face_rect.shape[1], face_rect.shape[0])

                # ── WHENet : Head Pose (crop carré 224×224) ──────────
                hp = whenet.predict(face_224)
                if hp is not None:
                    pitch, yaw, roll = hp
                    w_hp.writerow({
                        "frame_id":      frame_id,
                        "timestamp_sec": round(ts, 4),
                        "track_id":      tid,
                        "pitch":         round(pitch, 4),
                        "yaw":           round(yaw,   4),
                        "roll":          round(roll,  4),
                    })
                else:
                    log.debug("WHENet: pas de résultat frame %d", frame_id)

                # ── py-feat : Action Units (crop rect upscalé) ───────
                aus = pyfeat.predict(face_rect)
                au_rec: dict = {
                    "frame_id":   frame_id,
                    "track_id":   tid,
                    "confidence": 1.0 if aus is not None else 0.0,
                    "success":    1   if aus is not None else 0,
                }
                if aus is not None:
                    au_rec.update(aus)
                else:
                    for col in AU_INTENSITY + AU_BINARY:
                        au_rec[col] = 0.0
                    log.debug("py-feat: pas de résultat frame %d", frame_id)
                w_au.writerow(au_rec)

                # ── L2CS-Net : Gaze (crop carré 224×224) ─────────────
                gaze = l2cs.predict(face_224)
                gz_rec: dict = {
                    "frame_id":   frame_id,
                    "track_id":   tid,
                    "confidence": 1.0 if gaze is not None else 0.0,
                    "success":    1   if gaze is not None else 0,
                }
                if gaze is not None:
                    gz_rec.update(gaze)
                else:
                    for col in GAZE_COLS[4:]:
                        gz_rec[col] = 0.0
                w_gz.writerow(gz_rec)

        # ── Progression ───────────────────────────────────────
        frame_id += 1
        if frame_id % 50 == 0:
            pct = frame_id / total_frames * 100 if total_frames > 0 else 0
            log.info("  Frame %d/%d  (%.0f%%)", frame_id, total_frames, pct)

    # ── Fermeture ─────────────────────────────────────────────────
    cap.release()
    for fh in (fh_body, fh_hp, fh_au, fh_gz):
        fh.close()

    log.info("✅ Extraction terminée. Fichiers générés :")
    for f in sorted(output_dir.glob("*.csv")):
        size_kb = f.stat().st_size // 1024
        log.info("   %-42s  %d KB", f.name, size_kb)


# ──────────────────────────────────────────────────────────────────
# 13. LANGGRAPH NODE WRAPPER
# ──────────────────────────────────────────────────────────────────

def run_extraction_node(state: dict) -> dict:
    """
    LangGraph node: Unified video extraction.

    Takes a video file as input and produces all 4 raw CSV files
    in a single pass through the video:
      - raw_body_multi.csv      (YOLO pose tracking + keypoints)
      - raw_head_pose_multi.csv (WHENet head orientation)
      - raw_action_units_multi.csv (py-feat facial AUs)
      - raw_gaze_multi.csv      (L2CS-Net gaze vectors)

    State keys consumed:
        video_path       : str  — path to input video
        work_dir         : str  — output directory (optional, defaults to video parent)
        skip_extraction  : bool — if True, skip extraction and use existing CSVs

    State keys produced:
        raw_body_csv      : str
        raw_head_pose_csv : str
        raw_au_csv         : str
        raw_gaze_csv       : str
        face_crops_dir     : str  (empty — this extractor doesn't produce face crops)
        extraction_done    : bool
        error              : str | None
    """
    import os

    video_path = state.get("video_path", "")
    work_dir   = state.get("work_dir", str(Path(video_path).parent) if video_path else ".")

    print(f"\n{'='*60}")
    print(f"[Node: Extraction] Unified video extraction (v2)")
    print(f"  Video    : {video_path}")
    print(f"  Output   : {work_dir}")
    print(f"{'='*60}\n")

    output_dir = Path(work_dir)

    # ── Expected output paths ─────────────────────────────────────
    raw_body_csv = str(output_dir / "raw_body_multi.csv")
    raw_hp_csv   = str(output_dir / "raw_head_pose_multi.csv")
    raw_au_csv   = str(output_dir / "raw_action_units_multi.csv")
    raw_gaze_csv = str(output_dir / "raw_gaze_multi.csv")

    # ── Skip extraction if requested and CSVs exist ───────────────
    if state.get("skip_extraction"):
        all_exist = all(os.path.isfile(f) for f in [raw_body_csv, raw_hp_csv, raw_au_csv, raw_gaze_csv])
        if all_exist:
            print("[Extraction] ✓ Skipping — using existing raw CSVs.")
            return {
                "raw_body_csv":      raw_body_csv,
                "raw_head_pose_csv": raw_hp_csv,
                "raw_au_csv":        raw_au_csv,
                "raw_gaze_csv":      raw_gaze_csv,
                "face_crops_dir":    str(output_dir / "face_crops"),
                "extraction_done":   True,
                "error":             None,
            }
        else:
            print("[Extraction] --skip-extraction set but some CSVs missing. Running extraction.")

    # ── Validate video path ───────────────────────────────────────
    if not video_path or not os.path.isfile(video_path):
        msg = f"[Extraction] ERROR: video not found → {video_path}"
        print(msg)
        return {"extraction_done": False, "error": msg}

    # ── Run the unified extraction pipeline ───────────────────────
    try:
        run_extraction(video_path, output_dir)
    except Exception as exc:
        msg = f"[Extraction] Runtime error: {exc}"
        print(msg)
        import traceback; traceback.print_exc()
        return {"extraction_done": False, "error": msg}

    # ── Verify outputs ────────────────────────────────────────────
    missing = [f for f in [raw_body_csv, raw_hp_csv, raw_au_csv, raw_gaze_csv] if not os.path.isfile(f)]
    if missing:
        msg = f"[Extraction] WARNING: Missing output CSVs: {[os.path.basename(f) for f in missing]}"
        print(msg)

    print(f"\n[Node: Extraction] ✅ Done — 4 raw CSVs generated in {work_dir}")
    return {
        "raw_body_csv":      raw_body_csv,
        "raw_head_pose_csv": raw_hp_csv,
        "raw_au_csv":        raw_au_csv,
        "raw_gaze_csv":      raw_gaze_csv,
        "face_crops_dir":    str(output_dir / "face_crops"),
        "extraction_done":   True,
        "tracking_done":     True,
        "error":             None,
    }


# ──────────────────────────────────────────────────────────────────
# 14. POINT D'ENTRÉE CLI
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pixie v2 — Extraction comportementale en classe"
    )
    parser.add_argument("video", help="Chemin vers la vidéo source")
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR),
        help=f"Répertoire de sortie (défaut : {OUTPUT_DIR})",
    )
    args = parser.parse_args()
    run_extraction(args.video, Path(args.out))
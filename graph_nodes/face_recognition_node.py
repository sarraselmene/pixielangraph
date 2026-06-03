"""
graph_nodes/face_recognition_node.py
=====================================
Identifie l'élève visible dans la frame courante en comparant
l'encodage facial (face_recognition) à la base de données du CSV de mapping.

mapping_faces.csv attendu :
    student_name, image_path
    Alice,        /data/faces/alice.jpg
    Bob,          /data/faces/bob.jpg
    ...

Si face_recognition n'est pas installé → student_id = "student_unknown"

Clés State consommées :
    frame, mapping_csv, _face_db (cache)

Clés State produites :
    student_id, face_encoding, face_location, _face_db (mis à jour)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pixie_state import PixieState

# Tolérance de reconnaissance faciale (plus bas = plus strict)
TOLERANCE = 0.55
UNKNOWN   = "student_unknown"


def _build_face_db(mapping_csv: str) -> List[Tuple[Any, str]]:
    """
    Charge le mapping_faces.csv et construit la liste
    [(encoding_128d, student_name), ...].
    """
    try:
        import face_recognition as fr
        import pandas as pd
    except ImportError:
        print("[face_recog] WARN: face_recognition ou pandas non installé")
        return []

    if not os.path.isfile(mapping_csv):
        print(f"[face_recog] WARN: mapping CSV introuvable → {mapping_csv}")
        return []

    df  = pd.read_csv(mapping_csv)
    db  = []
    req = {"student_name", "image_path"}
    if not req.issubset(df.columns):
        print(f"[face_recog] WARN: colonnes attendues {req} — trouvées {set(df.columns)}")
        return []

    for _, row in df.iterrows():
        img_path = str(row["image_path"])
        name     = str(row["student_name"])
        if not os.path.isfile(img_path):
            print(f"[face_recog] WARN: image introuvable → {img_path}")
            continue
        img = fr.load_image_file(img_path)
        encs = fr.face_encodings(img)
        if encs:
            db.append((encs[0], name))
            print(f"[face_recog] Encodé : {name}")
        else:
            print(f"[face_recog] WARN: aucun visage dans {img_path}")

    print(f"[face_recog] Base de données : {len(db)} élève(s) chargé(s)")
    return db


def run_face_recognition_node(state: PixieState) -> PixieState:
    """
    Détecte et identifie le visage le plus proche dans la frame.

    Stratégie :
        1. Premier appel → charge la base de données depuis mapping_csv.
        2. Détecte les visages dans la frame (RGB).
        3. Compare chaque encodage à la base.
        4. Retourne l'identité avec la distance minimale si < TOLERANCE.
        5. Garde le student_id précédent si aucun visage détecté (robustesse).
    """
    frame      = state.get("frame")
    mapping_csv = state.get("mapping_csv", "")

    # ── Chargement initial de la base ─────────────────────────────────────────
    face_db: Optional[List] = state.get("_face_db")
    if face_db is None:
        face_db = _build_face_db(mapping_csv)

    if frame is None:
        return {**state, "_face_db": face_db,
                "student_id": state.get("student_id", UNKNOWN)}

    # ── Tentative d'identification ────────────────────────────────────────────
    student_id    = state.get("student_id", UNKNOWN)
    face_encoding = state.get("face_encoding")
    face_location = state.get("face_location")

    try:
        import face_recognition as fr

        # face_recognition travaille en RGB
        rgb = frame[:, :, ::-1].copy()

        locations = fr.face_locations(rgb, model="hog")   # "cnn" si GPU dispo
        if not locations:
            # Pas de visage → conserver l'identité précédente
            return {
                **state,
                "_face_db":     face_db,
                "student_id":   student_id,
                "face_location": None,
            }

        encodings = fr.face_encodings(rgb, locations)

        best_name     = UNKNOWN
        best_dist     = 1.0
        best_loc      = locations[0]
        best_encoding = encodings[0] if encodings else None

        for enc, loc in zip(encodings, locations):
            if not face_db:
                break
            db_encs = [e for e, _ in face_db]
            db_names = [n for _, n in face_db]
            dists    = fr.face_distance(db_encs, enc)
            idx      = int(np.argmin(dists))
            if dists[idx] < TOLERANCE and dists[idx] < best_dist:
                best_dist     = dists[idx]
                best_name     = db_names[idx]
                best_loc      = loc
                best_encoding = enc

        student_id    = best_name
        face_location = best_loc
        face_encoding = best_encoding

    except ImportError:
        # face_recognition non installé → identité fixe "student_unknown"
        pass
    except Exception as exc:
        print(f"[face_recog] Erreur : {exc}")

    return {
        **state,
        "_face_db":      face_db,
        "student_id":    student_id,
        "face_encoding": face_encoding,
        "face_location": face_location,
    }

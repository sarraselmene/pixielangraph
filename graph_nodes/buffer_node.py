"""
graph_nodes/sequence_buffer_node.py
=====================================
Gère le buffer glissant multi-élèves.

Chaque élève possède son propre deque de taille window_size.
Quand le buffer d'un élève atteint window_size frames,
le flag buffer_ready=True est mis dans le state pour déclencher l'inférence.

Clés State consommées :
    feature_vector, student_id, buffer_dict, window_size

Clés State produites :
    buffer_dict   (mis à jour)
    buffer_ready  bool
    X_window      np.ndarray (1, window_size, n_features) si buffer_ready
"""

from __future__ import annotations
from collections import deque
from typing import Any, Dict

import numpy as np

from pixie_state import PixieState, WINDOW_SIZE, FEATURE_COLS


def run_sequence_buffer_node(state: PixieState) -> PixieState:
    """
    Ajoute feature_vector au buffer de l'élève courant.
    Déclenche l'inférence si le buffer est plein.
    """
    feature_vector = state.get("feature_vector")
    student_id     = state.get("student_id", "unknown")
    buffer_dict    = dict(state.get("buffer_dict") or {})
    win_size       = state.get("window_size", WINDOW_SIZE)
    n_features     = len(FEATURE_COLS)

    if feature_vector is None:
        return {**state, "buffer_ready": False}

    # ── Initialiser le deque de l'élève si nécessaire ─────────────────────────
    if student_id not in buffer_dict:
        buffer_dict[student_id] = deque(maxlen=win_size)

    # ── Ajouter le vecteur courant (shape : (n_features,)) ───────────────────
    vec = feature_vector.flatten().astype(np.float32)
    buffer_dict[student_id].append(vec)

    buf = buffer_dict[student_id]

    # ── Vérifier si le buffer est plein ──────────────────────────────────────
    if len(buf) < win_size:
        # Buffer insuffisant → pas d'inférence
        return {
            **state,
            "buffer_dict":  buffer_dict,
            "buffer_ready": False,
        }

    # ── Construire le tenseur (1, window_size, n_features) ───────────────────
    X_window = np.stack(list(buf), axis=0)               # (win, n_feat)
    X_window = X_window.reshape(1, win_size, n_features)  # (1, win, n_feat)

    return {
        **state,
        "buffer_dict":  buffer_dict,
        "buffer_ready": True,
        "X_window":     X_window,
    }

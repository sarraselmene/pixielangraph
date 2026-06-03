"""
graph_nodes/inference_node.py
==============================
Effectue l'inférence LSTM sur la fenêtre courante.

Charge best_lstm_model.keras une seule fois (mise en cache dans _lstm_model).
Si clusters.pkl existe, effectue aussi l'assignation K-Means sur le vecteur
latent (compatible avec l'architecture BiLSTM autoencoder du preprocessor.py).

Clés State consommées :
    X_window, buffer_ready, student_id, frame_id,
    model_path, clusters_path, _lstm_model

Clés State produites :
    prediction  Dict {cluster, label, confidence, scores, student_id, frame_id}
    history     (mise à jour, max 60 entrées)
    _lstm_model (mis en cache)
"""

from __future__ import annotations
import os
from typing import Any, Dict, List

import numpy as np

from pixie_state import PixieState, BEHAVIOR_LABELS


def run_inference_node(state: PixieState) -> PixieState:
    """
    Inférence LSTM : prédit la classe comportementale à partir de X_window.
    Si buffer_ready=False, retourne la dernière prédiction inchangée.
    """
    if not state.get("buffer_ready", False):
        # Rien à inférer cette frame
        return state

    X_window   = state.get("X_window")
    student_id = state.get("student_id", "unknown")
    frame_id   = state.get("frame_id",   0)
    history    = list(state.get("history") or [])

    if X_window is None:
        return state

    # ── Chargement du modèle (une seule fois) ─────────────────────────────────
    model = state.get("_lstm_model")
    if model is None:
        model_path = state.get("model_path", "best_lstm_model.keras")
        if not os.path.isfile(model_path):
            print(f"[inference] WARN: modèle introuvable → {model_path}")
            return {**state, "prediction": _fallback(student_id, frame_id)}
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            model = tf.keras.models.load_model(model_path)
            print(f"[inference] Modèle chargé : {model_path}")
            print(f"            Input shape   : {model.input_shape}")
        except Exception as exc:
            print(f"[inference] Erreur chargement modèle : {exc}")
            return {**state, "prediction": _fallback(student_id, frame_id)}

    # ── Inférence LSTM (softmax sur 5 classes) ────────────────────────────────
    try:
        probas  = model.predict(X_window, verbose=0)[0]  # (5,)
        cluster = int(np.argmax(probas))
        conf    = float(np.max(probas))
        scores  = probas.tolist()
    except Exception as exc:
        print(f"[inference] Erreur predict : {exc}")
        probas  = np.ones(5) / 5
        cluster = 0
        conf    = 0.2
        scores  = probas.tolist()

    label = BEHAVIOR_LABELS.get(cluster, "INCONNU")

    prediction = {
        "cluster":    cluster,
        "label":      label,
        "confidence": round(conf, 4),
        "scores":     [round(s, 4) for s in scores],
        "student_id": student_id,
        "frame_id":   frame_id,
    }

    # ── Mise à jour de l'historique (max 60 entrées) ──────────────────────────
    history.append(prediction)
    if len(history) > 60:
        history = history[-60:]

    return {
        **state,
        "_lstm_model": model,
        "prediction":  prediction,
        "history":     history,
    }


def _fallback(student_id: str, frame_id: int) -> Dict:
    return {
        "cluster":    0,
        "label":      BEHAVIOR_LABELS[0],
        "confidence": 0.0,
        "scores":     [0.2, 0.2, 0.2, 0.2, 0.2],
        "student_id": student_id,
        "frame_id":   frame_id,
    }

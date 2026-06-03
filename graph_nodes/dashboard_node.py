"""
graph_nodes/dashboard_node.py
==============================
Crée le canvas OpenCV du dashboard Pixie et l'écrit dans le VideoWriter.

Layout du canvas (1440 × 810 px)
─────────────────────────────────
┌─────────────────────────┬──────────────────────────────┐
│                         │  NOM ÉLÈVE + CLASSE          │
│     VIDÉO SOURCE        │  ─────────────────────────── │
│     (720 × 810)         │  Jauge d'engagement           │
│                         │  ─────────────────────────── │
│                         │  Timeline comportement (60f)  │
│                         │  ─────────────────────────── │
│                         │  Radar des 5 scores           │
│                         │  ─────────────────────────── │
│                         │  Métriques temps réel         │
└─────────────────────────┴──────────────────────────────┘

Clés State consommées :
    frame, student_id, prediction, history, landmarks,
    frame_id, total_frames, _video_writer

Clés State produites :
    frame_id   (incrémenté de 1)
    _video_writer (inchangé)
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pixie_state import PixieState, BEHAVIOR_LABELS, BEHAVIOR_COLORS

# ─── Palette ──────────────────────────────────────────────────────────────────
BG_DARK      = (18,  18,  30)     # fond du panel analytique
BG_CARD      = (28,  28,  45)     # fond des cartes
ACCENT       = (80,  200, 120)    # vert accent (engagé)
TEXT_MAIN    = (230, 230, 245)
TEXT_SUB     = (140, 140, 160)
SEPARATOR    = (50,  50,  75)
WHITE        = (255, 255, 255)

CANVAS_W     = 1440
CANVAS_H     = 810
VIDEO_W      = 720
PANEL_W      = 720
PANEL_X      = VIDEO_W


def _put_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font=cv2.FONT_HERSHEY_DUPLEX,
    scale: float = 0.55,
    color: Tuple = TEXT_MAIN,
    thickness: int = 1,
    bold: bool = False,
) -> None:
    if bold:
        cv2.putText(img, text, pos, font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _draw_engagement_gauge(
    panel: np.ndarray,
    y0: int,
    score: float,          # 0.0 → 1.0
    label: str,
    color: Tuple,
) -> int:
    """
    Dessine une jauge horizontale d'engagement.
    Retourne le y final (après la jauge).
    """
    pad     = 20
    bar_w   = PANEL_W - 2 * pad
    bar_h   = 22
    fill_w  = int(bar_w * max(0.0, min(1.0, score)))

    # Fond de la barre
    cv2.rectangle(panel, (pad, y0), (pad + bar_w, y0 + bar_h), (40, 40, 60), -1)
    cv2.rectangle(panel, (pad, y0), (pad + bar_w, y0 + bar_h), SEPARATOR, 1)

    # Remplissage coloré (dégradé simulé via 5 rectangles)
    if fill_w > 0:
        for i in range(5):
            x1 = pad + int(fill_w * i / 5)
            x2 = pad + int(fill_w * (i+1) / 5)
            alpha = 0.5 + 0.5 * (i / 4)
            c = tuple(int(c * alpha) for c in color)
            cv2.rectangle(panel, (x1, y0), (x2, y0 + bar_h), c, -1)

    # Label + pourcentage
    pct_txt = f"{score*100:.0f}%"
    _put_text(panel, label,   (pad, y0 - 8), scale=0.45, color=TEXT_SUB)
    _put_text(panel, pct_txt, (pad + bar_w - 45, y0 + bar_h - 5),
              scale=0.5, color=TEXT_MAIN)

    return y0 + bar_h + 18


def _draw_timeline(
    panel: np.ndarray,
    y0: int,
    history: List[Dict],
    height: int = 70,
) -> int:
    """
    Trace une timeline horizontale colorée des 60 dernières prédictions.
    Chaque prédiction occupe une bande verticale colorée.
    """
    pad   = 20
    tl_w  = PANEL_W - 2 * pad
    n_h   = len(history)

    # Fond
    cv2.rectangle(panel, (pad, y0), (pad + tl_w, y0 + height), (30, 30, 48), -1)
    cv2.rectangle(panel, (pad, y0), (pad + tl_w, y0 + height), SEPARATOR, 1)

    if n_h == 0:
        _put_text(panel, "En attente de données…", (pad + 10, y0 + height//2 + 6),
                  scale=0.4, color=TEXT_SUB)
        return y0 + height + 15

    # Largeur de chaque bande
    step = tl_w / min(60, n_h)
    for i, pred in enumerate(history[-60:]):
        cls   = pred.get("cluster", 0)
        col   = BEHAVIOR_COLORS.get(cls, ACCENT)
        x1    = pad + int(i * step)
        x2    = pad + int((i + 1) * step)
        conf  = pred.get("confidence", 0.5)
        # Hauteur proportionnelle à la confiance
        bar_h = int(height * max(0.2, conf))
        y1    = y0 + height - bar_h
        cv2.rectangle(panel, (x1, y1), (x2, y0 + height), col, -1)

    # Légende
    _put_text(panel, "Timeline comportement (60 frames)",
              (pad, y0 - 8), scale=0.42, color=TEXT_SUB)

    # Curseur temporel (position courante = bord droit)
    cv2.line(panel, (pad + tl_w, y0), (pad + tl_w, y0 + height), WHITE, 2)

    return y0 + height + 20


def _draw_radar(
    panel: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    scores: List[float],
) -> None:
    """
    Dessine un radar (pentagone) des 5 scores comportementaux.
    """
    labels = ["ENG", "DIS", "AGI", "FAT", "SOC"]
    n      = 5
    angles = [math.radians(90 + 360 * i / n) for i in range(n)]

    # Cercles de référence
    for frac in [0.25, 0.5, 0.75, 1.0]:
        pts = [
            (int(cx + r * frac * math.cos(a)), int(cy - r * frac * math.sin(a)))
            for a in angles
        ]
        cv2.polylines(panel, [np.array(pts, dtype=np.int32)], True, SEPARATOR, 1)

    # Axes
    for a in angles:
        cv2.line(panel,
                 (cx, cy),
                 (int(cx + r * math.cos(a)), int(cy - r * math.sin(a))),
                 SEPARATOR, 1)

    # Aire des scores
    scores_n = [min(1.0, max(0.0, s)) for s in scores[:n]]
    pts_data = [
        (int(cx + r * scores_n[i] * math.cos(angles[i])),
         int(cy - r * scores_n[i] * math.sin(angles[i])))
        for i in range(n)
    ]
    overlay = panel.copy()
    cv2.fillPoly(overlay, [np.array(pts_data, dtype=np.int32)], (80, 200, 120, 100))
    cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
    cv2.polylines(panel, [np.array(pts_data, dtype=np.int32)], True, ACCENT, 2)

    # Points + labels
    for i, (px, py) in enumerate(pts_data):
        cv2.circle(panel, (px, py), 4, ACCENT, -1)
        lx = int(cx + (r + 18) * math.cos(angles[i]))
        ly = int(cy - (r + 18) * math.sin(angles[i]))
        _put_text(panel, labels[i], (lx - 12, ly + 5), scale=0.38, color=TEXT_SUB)


def _draw_metrics_grid(
    panel: np.ndarray,
    y0: int,
    landmarks: Dict,
    frame_id: int,
) -> None:
    """Affiche les métriques numériques en grille 2×3."""
    gaze = landmarks.get("gaze",  {})
    head = landmarks.get("head",  {})
    body = landmarks.get("body",  {})
    beh  = landmarks.get("behavior", {})

    metrics = [
        ("Gaze stab.",  f"{gaze.get('gaze_stability', 0):.2f}"),
        ("Divergence",  f"{gaze.get('eye_head_divergence', 0):.2f}"),
        ("Yaw",         f"{head.get('yaw', 0):+.1f}°"),
        ("Pitch",       f"{head.get('pitch', 0):+.1f}°"),
        ("Disp.",       f"{body.get('disp_score', 0):.3f}"),
        ("Frame",       f"#{frame_id}"),
    ]

    pad   = 20
    col_w = (PANEL_W - 2 * pad) // 3
    row_h = 38

    for i, (name, val) in enumerate(metrics):
        row = i // 3
        col = i % 3
        x   = pad + col * col_w
        y   = y0 + row * row_h
        # Carte de métrique
        cv2.rectangle(panel, (x, y), (x + col_w - 5, y + row_h - 5), BG_CARD, -1)
        cv2.rectangle(panel, (x, y), (x + col_w - 5, y + row_h - 5), SEPARATOR, 1)
        _put_text(panel, name, (x + 6,  y + 13), scale=0.35, color=TEXT_SUB)
        _put_text(panel, val,  (x + 6,  y + 28), scale=0.48, color=TEXT_MAIN)


def run_dashboard_node(state: PixieState) -> PixieState:
    """
    Compose le canvas et l'écrit dans le VideoWriter.
    Incrémente frame_id.
    """
    frame      = state.get("frame")
    student_id = state.get("student_id", "—")
    prediction = state.get("prediction") or {}
    history    = state.get("history")   or []
    landmarks  = state.get("landmarks") or {}
    frame_id   = state.get("frame_id",  0)
    total_fr   = state.get("total_frames", 1)
    writer     = state.get("_video_writer")

    cluster   = prediction.get("cluster",    0)
    label     = prediction.get("label",      BEHAVIOR_LABELS.get(0))
    conf      = prediction.get("confidence", 0.0)
    scores    = prediction.get("scores",     [0.2]*5)
    beh_color = BEHAVIOR_COLORS.get(cluster, ACCENT)

    # ── Canvas vierge ─────────────────────────────────────────────────────────
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_DARK, dtype=np.uint8)

    # ── Zone vidéo (moitié gauche) ────────────────────────────────────────────
    if frame is not None:
        vid = cv2.resize(frame, (VIDEO_W, CANVAS_H))
        # Overlay nom + classe en bas de la vidéo
        cv2.rectangle(vid, (0, CANVAS_H - 55), (VIDEO_W, CANVAS_H), (0,0,0), -1)
        cv2.addWeighted(vid[CANVAS_H-55:], 0.55,
                        np.zeros_like(vid[CANVAS_H-55:]), 0.45, 0,
                        vid[CANVAS_H-55:])
        # Boîte faciale
        loc = state.get("face_location")
        if loc:
            top, right, bottom, left = loc
            sx = VIDEO_W / (state.get("frame", np.zeros((1,1,3))).shape[1] or 1)
            sy = CANVAS_H / (state.get("frame", np.zeros((1,1,3))).shape[0] or 1)
            cv2.rectangle(vid,
                          (int(left*sx), int(top*sy)),
                          (int(right*sx), int(bottom*sy)),
                          beh_color, 2)
        cv2.putText(vid, student_id,
                    (12, CANVAS_H - 32), cv2.FONT_HERSHEY_DUPLEX,
                    0.65, WHITE, 1, cv2.LINE_AA)
        cv2.putText(vid, f"{label}  {conf*100:.0f}%",
                    (12, CANVAS_H - 10), cv2.FONT_HERSHEY_DUPLEX,
                    0.55, beh_color, 1, cv2.LINE_AA)
        canvas[:, :VIDEO_W] = vid
    else:
        cv2.putText(canvas, "Chargement…", (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_SUB, 1)

    # ── Séparateur vertical ───────────────────────────────────────────────────
    cv2.line(canvas, (VIDEO_W, 0), (VIDEO_W, CANVAS_H), SEPARATOR, 2)

    # ── Panel analytique (moitié droite) ──────────────────────────────────────
    panel = canvas[:, PANEL_X:].copy()   # vue locale pour dessiner

    y = 18

    # En-tête : nom de l'élève
    cv2.rectangle(panel, (0, 0), (PANEL_W, 52), BG_CARD, -1)
    name_disp = student_id[:28] if len(student_id) > 28 else student_id
    _put_text(panel, name_disp, (20, 34),
              scale=0.75, color=TEXT_MAIN, thickness=1, bold=True)
    # Badge de classe
    badge_txt = f"  {label}  "
    badge_w   = len(badge_txt) * 11
    cv2.rectangle(panel, (PANEL_W - badge_w - 20, 10),
                  (PANEL_W - 10, 42), beh_color, -1)
    _put_text(panel, badge_txt.strip(),
              (PANEL_W - badge_w - 14, 32), scale=0.48, color=(10,10,10))

    y = 62
    cv2.line(panel, (20, y), (PANEL_W - 20, y), SEPARATOR, 1)
    y += 12

    # ── Jauges des 5 scores ───────────────────────────────────────────────────
    _put_text(panel, "Scores comportementaux", (20, y), scale=0.45, color=TEXT_SUB)
    y += 16
    gauge_labels = ["Engagement", "Distraction", "Agitation", "Fatigue", "Anx. Sociale"]
    for i, (g_lbl, g_score) in enumerate(zip(gauge_labels, scores)):
        g_color = BEHAVIOR_COLORS.get(i, ACCENT)
        y = _draw_engagement_gauge(panel, y, g_score, g_lbl, g_color)

    cv2.line(panel, (20, y), (PANEL_W - 20, y), SEPARATOR, 1)
    y += 10

    # ── Timeline ──────────────────────────────────────────────────────────────
    y = _draw_timeline(panel, y, history, height=65)

    cv2.line(panel, (20, y), (PANEL_W - 20, y), SEPARATOR, 1)
    y += 12

    # ── Radar + métriques côte à côte ─────────────────────────────────────────
    radar_r  = 62
    radar_cx = 90
    radar_cy = y + radar_r + 10
    _draw_radar(panel, radar_cx, radar_cy, radar_r, scores)

    # Métriques à droite du radar
    _draw_metrics_grid(panel, y + 5, landmarks, frame_id)

    y = radar_cy + radar_r + 18

    # ── Barre de progression ──────────────────────────────────────────────────
    if total_fr > 0:
        prog_y = CANVAS_H - 20
        prog_w = PANEL_W - 40
        prog_fill = int(prog_w * (frame_id / max(1, total_fr)))
        cv2.rectangle(panel, (20, prog_y), (20 + prog_w, prog_y + 8), (40,40,60), -1)
        cv2.rectangle(panel, (20, prog_y), (20 + prog_fill, prog_y + 8), ACCENT, -1)
        pct_str = f"{100*frame_id//max(1,total_fr)}%  frame {frame_id}/{total_fr}"
        _put_text(panel, pct_str, (20, prog_y - 6), scale=0.38, color=TEXT_SUB)

    # ── Écrire panel dans le canvas ───────────────────────────────────────────
    canvas[:, PANEL_X:] = panel

    # ── Écrire dans le VideoWriter ────────────────────────────────────────────
    if writer is not None and writer.isOpened():
        writer.write(canvas)

    # ── Affichage temps réel (optionnel, désactivable) ────────────────────────
    try:
        cv2.imshow("Pixie Dashboard", canvas)
        cv2.waitKey(1)
    except Exception:
        pass   # environnement headless

    return {
        **state,
        "frame_id": frame_id + 1,
    }

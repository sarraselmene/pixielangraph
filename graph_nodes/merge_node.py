"""
graph_nodes/merge_node.py
=========================
Pixie — Merge node.

Reads the 5 labeled CSV files + face identity map and produces one row
per (frame_id, track_id):   full_analysis.csv

Source files (read from state keys, fallback to work_dir):
────────────────────────────────────────────────────────────
1. labeled_head_pose_multi.csv   → state["head_label_csv"]
2. labeled_gaze_multi.csv        → state["gaze_label_csv"]
3. labeled_action_units_multi.csv → state["au_label_csv"]
4. behaviour_raw_frames.csv      → state["body_raw_csv"]
5. face_identity_map.csv         → state["identity_map_csv"]

Optional:
6. behaviour_summary.csv         → state["body_label_csv"]

Output: full_analysis.csv
────────────────────────────
One row per (frame_id, track_id) — all columns from all sources merged.
Missing values filled with 0 (numerical) or "unknown" (categorical).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# DEFAULT FILE NAMES (fallback when state keys are missing)
# ═════════════════════════════════════════════════════════════════════════════

F_HEAD_POSE  = "labeled_head_pose_multi.csv"
F_GAZE       = "labeled_gaze_multi.csv"
F_AU         = "labeled_action_units_multi.csv"
F_BEHAVIOUR  = "behaviour_raw_frames.csv"
F_FACE_ID    = "face_identity_map.csv"
F_SUMMARY    = "behaviour_summary.csv"
F_OUTPUT     = "full_analysis.csv"


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _load(path: str, required: bool = True) -> Optional[pd.DataFrame]:
    """Load a CSV, printing status. Returns None if optional and missing."""
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(f"Required file missing: {path}")
        print(f"  [merge] OPTIONAL not found, skipping: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    print(f"  [merge] loaded {len(df):>7,} rows  ← {Path(path).name}")
    return df


def _rename_col(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    """Rename a column if it exists (silently skip if not)."""
    if old in df.columns:
        df = df.rename(columns={old: new})
    return df


def _ensure_key(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Assert frame_id and track_id exist; coerce to int."""
    for col in ("frame_id", "track_id"):
        if col not in df.columns:
            raise ValueError(f"[merge] {name} is missing column '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _resolve_csv(state: dict, state_key: str, work_dir: str, fallback_name: str) -> str:
    """Resolve a CSV path: state key → work_dir fallback."""
    path = state.get(state_key, "")
    if path and os.path.isfile(path):
        return path
    return os.path.join(work_dir, fallback_name)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN MERGE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def merge_csvs(
    head_path: str,
    gaze_path: str,
    au_path: str,
    body_path: str,
    face_id_path: str,
    summary_path: str,
    output: str,
    ffill: bool = True,
) -> pd.DataFrame:
    """
    Merge all source CSVs into full_analysis.csv.
    """
    print(f"\n[merge_node] Merging labeled CSVs …")

    # ── 1. HEAD POSE ──────────────────────────────────────────────────────────
    head = _load(head_path, required=True)
    head = _ensure_key(head, F_HEAD_POSE)
    head = _rename_col(head, "pitch_smooth", "pitch")
    head = _rename_col(head, "yaw_smooth",   "yaw")
    head = _rename_col(head, "roll_smooth",  "roll")

    # ── 2. GAZE ───────────────────────────────────────────────────────────────
    gaze = _load(gaze_path, required=True)
    gaze = _ensure_key(gaze, F_GAZE)
    gaze = _rename_col(gaze, "gaze_angle_x_smooth", "gaze_angle_x")
    gaze = _rename_col(gaze, "gaze_angle_y_smooth", "gaze_angle_y")

    # ── 3. ACTION UNITS ───────────────────────────────────────────────────────
    au = _load(au_path, required=True)
    au = _ensure_key(au, F_AU)

    # ── 4. BEHAVIOUR (body, frame-level) ──────────────────────────────────────
    body = _load(body_path, required=True)
    body = _ensure_key(body, F_BEHAVIOUR)
    body = _rename_col(body, "posture_confidence", "posture_conf")

    # ── 5. FACE IDENTITY (no frame_id — broadcast per track) ──────────────────
    face_id = _load(face_id_path, required=False)
    if face_id is not None:
        face_id["track_id"] = pd.to_numeric(
            face_id["track_id"], errors="coerce"
        ).fillna(0).astype(int)
        if "face_conf" in face_id.columns:
            face_id = (
                face_id.sort_values("face_conf", ascending=False)
                       .drop_duplicates(subset="track_id")
            )
        else:
            face_id = face_id.drop_duplicates(subset="track_id")
        face_id["person_present"] = 1

    # ── 6. BEHAVIOUR SUMMARY (optional episodic layer) ────────────────────────
    summary = _load(summary_path, required=False)
    if summary is not None and "track_id" in summary.columns:
        summary["track_id"] = pd.to_numeric(
            summary["track_id"], errors="coerce"
        ).fillna(0).astype(int)

    # ══════════════════════════════════════════════════════════════════════════
    # MERGE STRATEGY
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[merge_node] Joining on (frame_id, track_id) …")
    merged = head.copy()

    for right_df, name in [
        (gaze,  "gaze"),
        (au,    "action_units"),
        (body,  "behaviour"),
    ]:
        key_cols    = {"frame_id", "track_id"}
        shared_cols = (set(merged.columns) & set(right_df.columns)) - key_cols
        if shared_cols:
            right_df = right_df.rename(
                columns={c: f"{c}_{name}" for c in shared_cols}
            )
        merged = pd.merge(
            merged, right_df,
            on=["frame_id", "track_id"],
            how="left",
            validate="1:1",
        )

    print(f"  after frame-level joins: {len(merged):,} rows  {merged.shape[1]} cols")

    # Broadcast face identity per track
    if face_id is not None:
        merged = pd.merge(merged, face_id, on="track_id", how="left")
        merged["person_present"] = merged["person_present"].fillna(0).astype(int)

    # Attach episodic summary label
    if summary is not None and "start_frame" in summary.columns:
        merged = _attach_episode_label(merged, summary)

    # ══════════════════════════════════════════════════════════════════════════
    # POST-PROCESSING
    # ══════════════════════════════════════════════════════════════════════════

    merged = merged.sort_values(["track_id", "frame_id"]).reset_index(drop=True)

    if ffill:
        numeric_cols = merged.select_dtypes(include="number").columns.tolist()
        merged[numeric_cols] = (
            merged.groupby("track_id")[numeric_cols]
                  .transform(lambda g: g.ffill())
        )

    num_cols = merged.select_dtypes(include="number").columns
    str_cols = merged.select_dtypes(include="object").columns
    merged[num_cols] = merged[num_cols].fillna(0)
    merged[str_cols] = merged[str_cols].fillna("unknown")

    _ensure_lstm_columns(merged)

    merged.to_csv(output, index=False)
    size_kb = Path(output).stat().st_size / 1024
    print(f"\n[merge_node] Written: {output}  "
          f"({len(merged):,} rows × {merged.shape[1]} cols, {size_kb:.0f} KB)")
    _print_summary(merged)
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# EPISODE LABEL HELPER
# ═════════════════════════════════════════════════════════════════════════════

def _attach_episode_label(frames: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if "behaviour" not in summary.columns:
        return frames
    frames["episode_behaviour"]  = "unknown"
    frames["episode_confidence"] = 0.0
    frames["episode_duration"]   = 0
    for _, ep in summary.iterrows():
        tid   = int(ep["track_id"])
        start = int(ep.get("start_frame", 0))
        end   = int(ep.get("end_frame",   start))
        mask  = (
            (frames["track_id"] == tid)
            & (frames["frame_id"] >= start)
            & (frames["frame_id"] <= end)
        )
        frames.loc[mask, "episode_behaviour"]  = ep.get("behaviour",       "unknown")
        frames.loc[mask, "episode_confidence"] = ep.get("confidence_avg",  0.0)
        frames.loc[mask, "episode_duration"]   = ep.get("duration_frames", 0)
    return frames


# ═════════════════════════════════════════════════════════════════════════════
# COLUMN GUARD — ensure lstm_node required columns exist
# ═════════════════════════════════════════════════════════════════════════════

_LSTM_REQUIRED = {
    "posture_label":    ("str",   "unknown"),
    "posture_conf":     ("float", 0.0),
    "pitch":            ("float", 0.0),
    "yaw":              ("float", 0.0),
    "roll":             ("float", 0.0),
    "gaze_angle_x":     ("float", 0.0),
    "gaze_angle_y":     ("float", 0.0),
    "AU01_r":           ("float", 0.0),
    "AU04_r":           ("float", 0.0),
    "AU06_r":           ("float", 0.0),
    "AU12_r":           ("float", 0.0),
    "AU45_r":           ("float", 0.0),
    "AU01_c":           ("float", 0.0),
    "AU04_c":           ("float", 0.0),
    "AU45_c":           ("float", 0.0),
    "person_present":   ("float", 0.0),
    "face_conf":        ("float", 0.0),
    "gaze_stability":   ("float", 0.5),
    "expressiveness_score": ("float", 0.0),
    "eye_head_divergence":  ("float", 0.0),
    "fatigue_indicator":    ("float", 0.0),
    "yawning":              ("float", 0.0),
    "talking_flag":         ("float", 0.0),
    "disp_score":           ("float", 0.0),
    "genuine_smile":        ("float", 0.0),
}


def _ensure_lstm_columns(df: pd.DataFrame) -> None:
    for col, (dtype, fill) in _LSTM_REQUIRED.items():
        if col not in df.columns:
            print(f"  [merge] WARN: column '{col}' missing — filling with {fill}")
            df[col] = fill


def _print_summary(df: pd.DataFrame) -> None:
    print("\n── Merge summary ────────────────────────────────────────")
    print(f"  Tracks (track_ids) : {sorted(df['track_id'].unique())}")
    print(f"  Frame range        : {df['frame_id'].min()} – {df['frame_id'].max()}")
    print(f"  Total rows         : {len(df):,}")
    if "posture_label" in df.columns:
        print(f"  Posture dist       : {df['posture_label'].value_counts().to_dict()}")
    if "student_name" in df.columns:
        print(f"  Students           : {df['student_name'].unique().tolist()}")
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        print(f"  Remaining NaN cols : {nan_cols}")
    else:
        print(f"  NaN check          : all clear")


# ═════════════════════════════════════════════════════════════════════════════
# LangGraph node entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_merge_node(state: dict) -> dict:
    """
    LangGraph node wrapper.

    Reads CSV paths from state keys, with fallback to work_dir.

    State keys consumed:  head_label_csv, gaze_label_csv, au_label_csv,
                          body_raw_csv, identity_map_csv, body_label_csv, work_dir
    State keys produced:  full_analysis_csv, merge_done, error
    """
    work_dir = state.get("work_dir", ".")
    os.makedirs(work_dir, exist_ok=True)

    # Resolve all CSV paths from state with work_dir fallback
    head_path    = _resolve_csv(state, "head_label_csv",    work_dir, F_HEAD_POSE)
    gaze_path    = _resolve_csv(state, "gaze_label_csv",    work_dir, F_GAZE)
    au_path      = _resolve_csv(state, "au_label_csv",      work_dir, F_AU)
    body_path    = _resolve_csv(state, "body_raw_csv",      work_dir, F_BEHAVIOUR)
    face_id_path = _resolve_csv(state, "identity_map_csv",  work_dir, F_FACE_ID)
    summary_path = _resolve_csv(state, "body_label_csv",    work_dir, F_SUMMARY)

    output_path = os.path.join(work_dir, F_OUTPUT)

    print(f"\n[merge_node] CSV sources:")
    print(f"  head_pose : {head_path}")
    print(f"  gaze      : {gaze_path}")
    print(f"  au        : {au_path}")
    print(f"  body      : {body_path}")
    print(f"  face_id   : {face_id_path}")
    print(f"  summary   : {summary_path}")

    try:
        merge_csvs(
            head_path=head_path,
            gaze_path=gaze_path,
            au_path=au_path,
            body_path=body_path,
            face_id_path=face_id_path,
            summary_path=summary_path,
            output=output_path,
        )
    except Exception as exc:
        msg = f"[merge_node] ERROR: {exc}"
        print(msg)
        import traceback; traceback.print_exc()
        return {"merge_done": False, "full_analysis_csv": None, "error": msg}

    return {
        "merge_done":        True,
        "full_analysis_csv": output_path,
        "error":             None,
    }
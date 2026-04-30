"""
graph_nodes/gaze_node.py
========================
Description:
    LangGraph node wrapper for label_gaze.py. It processes raw gaze signals 
    and applies head-relative classification logic to distinguish between 
    eye-in-socket movements and head orientation. It also produces 
    room-relative focus labels for distractibility analysis.

Changes Effectuated:
    - Added comprehensive documentation header and change log.
    - Integrated head-relative gaze classification as the primary signal.
    - Standardized room-relative focus mapping for clinical reporting.
    - Improved dynamic config patching for multi-student track isolation.
    - Made heavy imports lazy to avoid import errors in skip-extraction mode.
"""

import os
import sys
from pathlib import Path

def run_gaze_labeling(state: dict) -> dict:
    raw_gaze_csv   = state.get("raw_gaze_csv", "")
    head_label_csv = state.get("head_label_csv", "")
    work_dir       = state.get("work_dir", ".")

    print(f"\n{'='*60}")
    print(f"[Node: Gaze Labeling] Input : {raw_gaze_csv}")
    print(f"{'='*60}\n")

    if not os.path.isfile(raw_gaze_csv):
        msg = f"[Gaze] ERROR: raw gaze CSV not found → {raw_gaze_csv}"
        print(msg)
        return {"gaze_labeling_done": False, "error": msg}

    os.chdir(work_dir)

    # Lazy import
    try:
        from label_gaze import main as run_g_labeling
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent))
        from label_gaze import main as run_g_labeling

    # Patch config paths at runtime via monkeypatching config module
    _patch_config(work_dir, gaze_input=raw_gaze_csv, head_label=head_label_csv)

    try:
        run_g_labeling()
    except SystemExit:
        pass
    except Exception as exc:
        msg = f"[Gaze] Runtime error: {exc}"
        print(msg)
        _restore_config()
        return {"gaze_labeling_done": False, "error": msg}
    finally:
        _restore_config()

    output_csv = os.path.join(work_dir, "labeled_gaze_multi.csv")
    if not os.path.isfile(output_csv):
        msg = f"[Gaze] Output not found: {output_csv}"
        print(msg)
        return {"gaze_labeling_done": False, "error": msg}

    print(f"[Node: Gaze Labeling] ✓ Done → {output_csv}")
    return {
        "gaze_label_csv":    output_csv,
        "gaze_labeling_done": True,
        "error": None,
    }



# ── Config patching helpers ──────────────────────────────────────────────────

_original_paths: dict = {}

def _patch_config(work_dir: str, gaze_input: str = "", head_label: str = ""):
    """Temporarily override Paths in config module."""
    import importlib
    try:
        import config as cfg_mod
        _original_paths["GAZE_INPUT_CSV"]    = cfg_mod.Paths.GAZE_INPUT_CSV
        _original_paths["GAZE_HEAD_POSE_CSV"] = cfg_mod.Paths.GAZE_HEAD_POSE_CSV
        _original_paths["GAZE_OUTPUT_CSV"]   = cfg_mod.Paths.GAZE_OUTPUT_CSV
        if gaze_input:
            cfg_mod.Paths.GAZE_INPUT_CSV = gaze_input
        if head_label:
            cfg_mod.Paths.GAZE_HEAD_POSE_CSV = head_label
        cfg_mod.Paths.GAZE_OUTPUT_CSV = os.path.join(work_dir, "labeled_gaze_multi.csv")
    except ImportError:
        pass


def _restore_config():
    """Restore Paths in config module after node execution."""
    try:
        import config as cfg_mod
        for attr, val in _original_paths.items():
            setattr(cfg_mod.Paths, attr, val)
        _original_paths.clear()
    except ImportError:
        pass

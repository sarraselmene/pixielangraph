"""
graph_nodes/action_units_node.py
=================================
Description:
    LangGraph node wrapper for label_action_units.py. It processes raw 
    facial action unit intensities to detect behavioral expressions 
    (smiles, fatigue, yawning) and aggregates collective social events 
    across multiple students.

Changes Effectuated:
    - Added comprehensive documentation header and change log.
    - Verified integration of collective smile event detection logic.
    - Standardized output synchronization for labeled AU data and event CSVs.
    - Made heavy imports lazy to avoid import errors in skip-extraction mode.
"""

import os
import sys
from pathlib import Path

def run_action_units_labeling(state: dict) -> dict:
    raw_au_csv     = state.get("raw_au_csv", "")
    head_label_csv = state.get("head_label_csv", "")
    work_dir       = state.get("work_dir", ".")

    print(f"\n{'='*60}")
    print(f"[Node: Action Units Labeling] Input : {raw_au_csv}")
    print(f"{'='*60}\n")

    if not os.path.isfile(raw_au_csv):
        msg = f"[AU] ERROR: raw action-units CSV not found → {raw_au_csv}"
        print(msg)
        return {"au_labeling_done": False, "error": msg}

    os.chdir(work_dir)
    _patch_config(work_dir, au_input=raw_au_csv, head_label=head_label_csv)

    # Lazy import
    try:
        from label_action_units import main as run_au_labeling
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent))
        from label_action_units import main as run_au_labeling

    try:
        run_au_labeling()
    except SystemExit:
        pass
    except Exception as exc:
        msg = f"[AU] Runtime error: {exc}"
        print(msg)
        _restore_config()
        return {"au_labeling_done": False, "error": msg}
    finally:
        _restore_config()

    au_csv     = os.path.join(work_dir, "labeled_action_units_multi.csv")
    events_csv = os.path.join(work_dir, "collective_events.csv")

    if not os.path.isfile(au_csv):
        msg = f"[AU] Output not found: {au_csv}"
        print(msg)
        return {"au_labeling_done": False, "error": msg}

    print(f"[Node: Action Units Labeling] ✓ Done → {au_csv}")
    return {
        "au_label_csv":    au_csv,
        "events_csv":      events_csv if os.path.isfile(events_csv) else None,
        "au_labeling_done": True,
        "error": None,
    }



_original_paths: dict = {}

def _patch_config(work_dir: str, au_input: str = "", head_label: str = ""):
    try:
        import config as cfg_mod
        _original_paths["AU_INPUT_CSV"]    = cfg_mod.Paths.AU_INPUT_CSV
        _original_paths["AU_HEAD_POSE_CSV"] = cfg_mod.Paths.AU_HEAD_POSE_CSV
        _original_paths["AU_OUTPUT_CSV"]   = cfg_mod.Paths.AU_OUTPUT_CSV
        _original_paths["AU_EVENTS_CSV"]   = cfg_mod.Paths.AU_EVENTS_CSV
        if au_input:
            cfg_mod.Paths.AU_INPUT_CSV = au_input
        if head_label:
            cfg_mod.Paths.AU_HEAD_POSE_CSV = head_label
        cfg_mod.Paths.AU_OUTPUT_CSV = os.path.join(work_dir, "labeled_action_units_multi.csv")
        cfg_mod.Paths.AU_EVENTS_CSV = os.path.join(work_dir, "collective_events.csv")
    except ImportError:
        pass


def _restore_config():
    try:
        import config as cfg_mod
        for attr, val in _original_paths.items():
            setattr(cfg_mod.Paths, attr, val)
        _original_paths.clear()
    except ImportError:
        pass

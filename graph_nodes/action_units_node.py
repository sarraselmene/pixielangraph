"""
graph_nodes/action_units_node.py
=================================
LangGraph node wrapper for label_action_units.py.
Reads raw_action_units_multi.csv and produces:
  labeled_action_units_multi.csv + collective_events.csv
"""

import os
import sys
import types
from pathlib import Path


def run_action_units_labeling(state: dict) -> dict:
    """
    LangGraph node: Label facial action units (smile, fatigue, yawning …).

    Expects state keys:
        raw_au_csv         (str): raw_action_units_multi.csv
        head_label_csv     (str): labeled_head_pose_multi.csv (optional)
        work_dir           (str)

    Produces state keys:
        au_label_csv       (str): labeled_action_units_multi.csv
        events_csv         (str): collective_events.csv
        au_labeling_done   (bool)
        error              (str | None)
    """
    raw_au_csv     = state.get("raw_au_csv", "")
    head_label_csv = state.get("head_label_csv", "")
    work_dir       = state.get("work_dir", ".")

    print(f"\n{'='*60}")
    print(f"[Node: Action Units Labeling] Input : {raw_au_csv}")
    print(f"{'='*60}\n")

    if not os.path.isfile(raw_au_csv):
        msg = f"[AU] ERROR: raw action-units CSV not found → {raw_au_csv}"
        print(msg)
        return {**state, "au_labeling_done": False, "error": msg}

    script_path = Path(__file__).parent.parent / "label_action_units.py"
    src = script_path.read_text(encoding="utf-8")

    os.chdir(work_dir)
    _patch_config(work_dir, au_input=raw_au_csv, head_label=head_label_csv)

    dummy = types.ModuleType("label_action_units")
    sys.modules["label_action_units"] = dummy
    try:
        exec_globals = {"__name__": "__main_exec__", "__file__": str(script_path)}
        exec(compile(src, str(script_path), "exec"), exec_globals)
        main_fn = exec_globals.get("main")
        if main_fn:
            main_fn()
    except SystemExit:
        pass
    except Exception as exc:
        msg = f"[AU] Runtime error: {exc}"
        print(msg)
        _restore_config()
        return {**state, "au_labeling_done": False, "error": msg}
    finally:
        sys.modules.pop("label_action_units", None)
        _restore_config()

    au_csv     = os.path.join(work_dir, "labeled_action_units_multi.csv")
    events_csv = os.path.join(work_dir, "collective_events.csv")

    if not os.path.isfile(au_csv):
        msg = f"[AU] Output not found: {au_csv}"
        print(msg)
        return {**state, "au_labeling_done": False, "error": msg}

    print(f"[Node: Action Units Labeling] ✓ Done → {au_csv}")
    return {
        **state,
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

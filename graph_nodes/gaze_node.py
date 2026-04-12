"""
graph_nodes/gaze_node.py
========================
LangGraph node wrapper for label_gaze.py.
Reads raw_gaze_multi.csv (+ labeled_head_pose_multi.csv if available)
and produces labeled_gaze_multi.csv.
"""

import os
import sys
import types
from pathlib import Path


def run_gaze_labeling(state: dict) -> dict:
    """
    LangGraph node: Label gaze direction & stability.

    Expects state keys:
        raw_gaze_csv       (str): raw_gaze_multi.csv
        head_label_csv     (str): labeled_head_pose_multi.csv (optional)
        work_dir           (str)

    Produces state keys:
        gaze_label_csv     (str): labeled_gaze_multi.csv
        gaze_labeling_done (bool)
        error              (str | None)
    """
    raw_gaze_csv   = state.get("raw_gaze_csv", "")
    head_label_csv = state.get("head_label_csv", "")
    work_dir       = state.get("work_dir", ".")

    print(f"\n{'='*60}")
    print(f"[Node: Gaze Labeling] Input : {raw_gaze_csv}")
    print(f"{'='*60}\n")

    if not os.path.isfile(raw_gaze_csv):
        msg = f"[Gaze] ERROR: raw gaze CSV not found → {raw_gaze_csv}"
        print(msg)
        return {**state, "gaze_labeling_done": False, "error": msg}

    script_path = Path(__file__).parent.parent / "label_gaze.py"
    src = script_path.read_text(encoding="utf-8")

    os.chdir(work_dir)

    # Patch config paths at runtime via monkeypatching config module
    _patch_config(work_dir, gaze_input=raw_gaze_csv, head_label=head_label_csv)

    dummy = types.ModuleType("label_gaze")
    sys.modules["label_gaze"] = dummy
    try:
        exec_globals = {"__name__": "__main_exec__", "__file__": str(script_path)}
        exec(compile(src, str(script_path), "exec"), exec_globals)
        main_fn = exec_globals.get("main")
        if main_fn:
            main_fn()
    except SystemExit:
        pass
    except Exception as exc:
        msg = f"[Gaze] Runtime error: {exc}"
        print(msg)
        _restore_config()
        return {**state, "gaze_labeling_done": False, "error": msg}
    finally:
        sys.modules.pop("label_gaze", None)
        _restore_config()

    output_csv = os.path.join(work_dir, "labeled_gaze_multi.csv")
    if not os.path.isfile(output_csv):
        msg = f"[Gaze] Output not found: {output_csv}"
        print(msg)
        return {**state, "gaze_labeling_done": False, "error": msg}

    print(f"[Node: Gaze Labeling] ✓ Done → {output_csv}")
    return {
        **state,
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

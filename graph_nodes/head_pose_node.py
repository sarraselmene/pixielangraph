"""
graph_nodes/head_pose_node.py
=============================
LangGraph node wrapper for label_head_pose.py.
Reads raw_head_pose_multi.csv and produces labeled_head_pose_multi.csv.
"""

import os
import sys
import types
from pathlib import Path


def run_head_pose_labeling(state: dict) -> dict:
    """
    LangGraph node: Label head pose (pitch / yaw / roll → discrete labels).

    Expects state keys:
        raw_head_pose_csv      (str): raw_head_pose_multi.csv
        work_dir               (str)

    Produces state keys:
        head_label_csv         (str): labeled_head_pose_multi.csv
        head_labeling_done     (bool)
        error                  (str | None)
    """
    raw_head_pose_csv = state.get("raw_head_pose_csv", "")
    work_dir          = state.get("work_dir", ".")

    print(f"\n{'='*60}")
    print(f"[Node: Head Pose Labeling] Input : {raw_head_pose_csv}")
    print(f"{'='*60}\n")

    if not os.path.isfile(raw_head_pose_csv):
        msg = f"[HeadPose] ERROR: raw head-pose CSV not found → {raw_head_pose_csv}"
        print(msg)
        return {**state, "head_labeling_done": False, "error": msg}

    script_path = Path(__file__).parent.parent / "label_head_pose.py"
    src = script_path.read_text(encoding="utf-8")

    os.chdir(work_dir)
    _patch_config(work_dir, head_input=raw_head_pose_csv)

    dummy = types.ModuleType("label_head_pose")
    sys.modules["label_head_pose"] = dummy
    try:
        exec_globals = {"__name__": "__main_exec__", "__file__": str(script_path)}
        exec(compile(src, str(script_path), "exec"), exec_globals)
        main_fn = exec_globals.get("main")
        if main_fn:
            main_fn()
    except SystemExit:
        pass
    except Exception as exc:
        msg = f"[HeadPose] Runtime error: {exc}"
        print(msg)
        _restore_config()
        return {**state, "head_labeling_done": False, "error": msg}
    finally:
        sys.modules.pop("label_head_pose", None)
        _restore_config()

    output_csv = os.path.join(work_dir, "labeled_head_pose_multi.csv")
    if not os.path.isfile(output_csv):
        msg = f"[HeadPose] Output not found: {output_csv}"
        print(msg)
        return {**state, "head_labeling_done": False, "error": msg}

    print(f"[Node: Head Pose Labeling] ✓ Done → {output_csv}")
    return {
        **state,
        "head_label_csv":    output_csv,
        "head_labeling_done": True,
        "error": None,
    }


_original_paths: dict = {}

def _patch_config(work_dir: str, head_input: str = ""):
    try:
        import config as cfg_mod
        _original_paths["HEAD_POSE_INPUT_CSV"]  = cfg_mod.Paths.HEAD_POSE_INPUT_CSV
        _original_paths["HEAD_POSE_OUTPUT_CSV"] = cfg_mod.Paths.HEAD_POSE_OUTPUT_CSV
        if head_input:
            cfg_mod.Paths.HEAD_POSE_INPUT_CSV = head_input
        cfg_mod.Paths.HEAD_POSE_OUTPUT_CSV = os.path.join(work_dir, "labeled_head_pose_multi.csv")
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

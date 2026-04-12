"""
graph_nodes/body_node.py
========================
LangGraph node wrapper for anchor2.py (body behaviour classifier).
Reads raw_body_multi.csv and produces bodyLabeling.csv
(behaviour_summary.csv + behaviour_raw_frames.csv).
"""

import os
import sys
import types
import importlib.util
from pathlib import Path


def run_body_labeling(state: dict) -> dict:
    """
    LangGraph node: Label body posture & behaviour.

    Expects state keys:
        raw_body_csv (str): Path to raw_body_multi.csv
        work_dir     (str): Working directory

    Produces state keys:
        body_label_csv     (str): Path to behaviour_summary.csv
        body_raw_csv       (str): Path to behaviour_raw_frames.csv
        body_labeling_done (bool)
        error              (str | None)
    """
    raw_body_csv = state.get("raw_body_csv", "")
    work_dir     = state.get("work_dir", ".")
    video_path   = state.get("video_path", "")

    print(f"\n{'='*60}")
    print(f"[Node: Body Labeling] Input : {raw_body_csv}")
    print(f"{'='*60}\n")

    if not os.path.isfile(raw_body_csv):
        msg = f"[Body] ERROR: raw body CSV not found → {raw_body_csv}"
        print(msg)
        return {**state, "body_labeling_done": False, "error": msg}

    script_path = Path(__file__).parent.parent / "anchor2.py"
    src = script_path.read_text(encoding="utf-8")

    # Patch CFG defaults to point at the correct files
    src = src.replace(
        'VIDEO_PATH:  str = "aya2.mov"',
        f'VIDEO_PATH:  str = r"{video_path}"',
    )
    src = src.replace(
        'BODY_CSV:    str = "raw_body_multi.csv"',
        f'BODY_CSV:    str = r"{raw_body_csv}"',
    )

    os.chdir(work_dir)

    dummy = types.ModuleType("anchor2")
    sys.modules["anchor2"] = dummy
    try:
        exec_globals = {"__name__": "__main_exec__", "__file__": str(script_path)}
        exec(compile(src, str(script_path), "exec"), exec_globals)
        main_fn = exec_globals.get("main") or exec_globals.get("run")
        if main_fn:
            main_fn()
        else:
            # anchor2 may run at module level via argparse — try via argv
            import argparse
            sys.argv = [
                str(script_path),
                "--video",  video_path,
                "--body",   raw_body_csv,
            ]
            # re-exec if there's a guarded main
            if "__name__" in exec_globals:
                pass  # already ran at module load
    except SystemExit:
        pass  # argparse / sys.exit(0) — normal exit
    except Exception as exc:
        msg = f"[Body] Runtime error: {exc}"
        print(msg)
        return {**state, "body_labeling_done": False, "error": msg}
    finally:
        sys.modules.pop("anchor2", None)

    summary_csv = os.path.join(work_dir, "behaviour_summary.csv")
    raw_csv     = os.path.join(work_dir, "behaviour_raw_frames.csv")

    # Accept partial output (raw frames might not exist)
    body_label_csv = summary_csv if os.path.isfile(summary_csv) else None
    body_raw_csv   = raw_csv if os.path.isfile(raw_csv) else None

    if not body_label_csv and not body_raw_csv:
        msg = "[Body] No output CSVs were produced."
        print(msg)
        return {**state, "body_labeling_done": False, "error": msg}

    print(f"[Node: Body Labeling] ✓ Done. summary={summary_csv}")
    return {
        **state,
        "body_label_csv":    body_label_csv or summary_csv,
        "body_raw_csv":      body_raw_csv or raw_csv,
        "body_labeling_done": True,
        "error": None,
    }

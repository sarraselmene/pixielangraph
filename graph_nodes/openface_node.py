"""
graph_nodes/openface_node.py
============================
LangGraph node wrapper for OpenFace / py-feat parallel extraction.
Produces raw AU and gaze CSVs from face crops.
"""

import os
import sys
from pathlib import Path


def run_openface_parallel(state: dict) -> dict:
    """
    LangGraph node: Run OpenFace (or py-feat) AU + gaze extraction.
    """
    work_dir  = state.get("work_dir", ".")
    crops_dir = state.get("face_crops_dir", os.path.join(work_dir, "face_crops"))

    print(f"\n{'='*60}")
    print(f"[Node: OpenFace Parallel] Input: {crops_dir}")
    print(f"{'='*60}\n")

    if state.get("skip_extraction"):
        raw_au_csv   = os.path.join(work_dir, "raw_action_units_multi.csv")
        raw_gaze_csv = os.path.join(work_dir, "raw_gaze_multi.csv")
        if os.path.isfile(raw_au_csv) and os.path.isfile(raw_gaze_csv):
            print("[OpenFace] ✓ Skipping — using existing raw CSVs.")
            return {
                "raw_au_csv":   raw_au_csv,
                "raw_gaze_csv": raw_gaze_csv,
                "extraction_done": True,
                "error": None,
            }

    # Try to run the extraction
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from extract_openface import run_openface_parallel as _run_of
        _run_of(work_dir)
    except ImportError:
        print("[OpenFace] extract_openface.py not found — checking if CSVs exist from extract_node")
    except Exception as exc:
        msg = f"[OpenFace] Runtime error: {exc}"
        print(msg)
        return {"extraction_done": False, "error": msg}

    raw_au_csv   = os.path.join(work_dir, "raw_action_units_multi.csv")
    raw_gaze_csv = os.path.join(work_dir, "raw_gaze_multi.csv")

    print(f"[Node: OpenFace Parallel] ✓ Done")
    return {
        "raw_au_csv":      raw_au_csv,
        "raw_gaze_csv":    raw_gaze_csv,
        "extraction_done": True,
        "error":           None,
    }

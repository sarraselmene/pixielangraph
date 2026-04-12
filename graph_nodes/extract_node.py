"""
graph_nodes/extract_node.py
===========================
LangGraph node wrapper for extract_raw_data_multi.py.
Runs the full YOLO + 6DRepNet + OpenFace extraction pipeline on a given
video and returns the four raw CSV paths expected by downstream nodes.

Output files (all written to work_dir):
  raw_body_multi.csv           → anchor2.py (body labeling)
  raw_head_pose_multi.csv      → label_head_pose.py
  raw_action_units_multi.csv   → label_action_units.py  (empty if no OpenFace)
  raw_gaze_multi.csv           → label_gaze.py          (empty if no OpenFace)
"""

import csv
import os
import sys
import types
from pathlib import Path


# ── AU / Gaze column headers (mirrors extract_raw_data_multi.py) ──────────
_AU_INTENSITY_COLS = [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r",
    "AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r",
    "AU20_r","AU23_r","AU25_r","AU26_r","AU45_r",
]
_AU_BINARY_COLS = [
    "AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c",
    "AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c",
    "AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c",
]
_GAZE_COLS = [
    "gaze_0_x","gaze_0_y","gaze_0_z",
    "gaze_1_x","gaze_1_y","gaze_1_z",
    "gaze_angle_x","gaze_angle_y",
]


def _write_stub(path: str, extra_cols: list) -> None:
    """Create a header-only CSV so downstream nodes never crash."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "track_id", "confidence", "success"] + extra_cols)
    print(f"[Extraction] Stub CSV created (headers only): {path}")


def run_extraction(state: dict) -> dict:
    """
    LangGraph node: Extract raw multimodal data using
    extract_raw_data_multi.py (YOLO + 6DRepNet + OpenFace).

    Expects state keys:
        video_path     (str): Absolute path to the input video.
        work_dir       (str): Directory for all output files.
        openface_dir   (str, optional): Path to the OpenFace build/bin dir.

    Produces state keys:
        raw_body_csv      (str): raw_body_multi.csv
        raw_head_pose_csv (str): raw_head_pose_multi.csv
        raw_au_csv        (str): raw_action_units_multi.csv
        raw_gaze_csv      (str): raw_gaze_multi.csv
        extraction_done   (bool)
        error             (str | None)
    """
    video_path   = state.get("video_path", "")
    work_dir     = state.get("work_dir", str(Path(video_path).parent))
    openface_dir = state.get(
        "openface_dir",
        "/Users/sarahselmene/OpenFace/build/bin"   # same default as extract_raw_data2.py
    )

    print(f"\n{'='*60}")
    print(f"[Node: Extraction] Starting for video: {video_path}")
    print(f"[Node: Extraction] Script  : extract_raw_data_multi.py")
    print(f"[Node: Extraction] Work dir: {work_dir}")
    print(f"[Node: Extraction] OpenFace: {openface_dir}")
    print(f"{'='*60}\n")

    if not os.path.isfile(video_path):
        msg = f"[Extraction] ERROR: video not found → {video_path}"
        print(msg)
        return {**state, "extraction_done": False, "error": msg}

    script_path = Path(__file__).parent.parent / "extract_raw_data_multi.py"
    if not script_path.exists():
        msg = f"[Extraction] ERROR: extract_raw_data_multi.py not found at {script_path}"
        print(msg)
        return {**state, "extraction_done": False, "error": msg}

    os.chdir(work_dir)

    # ── Patch the script source at runtime ──────────────────────────
    src = script_path.read_text(encoding="utf-8")

    # 1. Redirect input video
    src = src.replace(
        'INPUT_SOURCE = "testing_vid/own_vid(gaze direction)1.mp4"',
        f'INPUT_SOURCE = r"{video_path}"',
    )

    # 2. Redirect output CSVs and temp dirs into work_dir
    src = src.replace(
        'BODY_OUTPUT      = "raw_body_multi.csv"',
        f'BODY_OUTPUT      = r"{os.path.join(work_dir, "raw_body_multi.csv")}"',
    )
    src = src.replace(
        'HEAD_POSE_OUTPUT = "raw_head_pose_multi.csv"',
        f'HEAD_POSE_OUTPUT = r"{os.path.join(work_dir, "raw_head_pose_multi.csv")}"',
    )
    src = src.replace(
        'AU_OUTPUT        = "raw_action_units_multi.csv"',
        f'AU_OUTPUT        = r"{os.path.join(work_dir, "raw_action_units_multi.csv")}"',
    )
    src = src.replace(
        'GAZE_OUTPUT      = "raw_gaze_multi.csv"',
        f'GAZE_OUTPUT      = r"{os.path.join(work_dir, "raw_gaze_multi.csv")}"',
    )
    src = src.replace(
        'FACE_CROPS_DIR   = "face_crops"',
        f'FACE_CROPS_DIR   = r"{os.path.join(work_dir, "face_crops")}"',
    )
    src = src.replace(
        'OPENFACE_OUT_DIR = "openface_output"',
        f'OPENFACE_OUT_DIR = r"{os.path.join(work_dir, "openface_output")}"',
    )

    # 3. Patch the Windows OpenFace path → Mac path
    openface_exe = os.path.join(openface_dir, "FaceLandmarkImg")
    src = src.replace(
        r'OPENFACE_DIR = r"C:\Users\mouss\Documents\OpenFace_2.2.0_win_x86"',
        f'OPENFACE_DIR = r"{openface_dir}"',
    )
    src = src.replace(
        'OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FaceLandmarkImg.exe")',
        f'OPENFACE_EXE = r"{openface_exe}"',
    )

    # ── Execute the patched script ───────────────────────────────────
    dummy = types.ModuleType("extract_raw_data_multi")
    sys.modules["extract_raw_data_multi"] = dummy
    try:
        exec_globals = {"__name__": "__main_exec__", "__file__": str(script_path)}
        exec(compile(src, str(script_path), "exec"), exec_globals)
        main_fn = exec_globals.get("main")
        if main_fn:
            main_fn()
    except SystemExit:
        pass  # normal script exit
    except Exception as exc:
        msg = f"[Extraction] Runtime error: {exc}"
        print(msg)
        return {**state, "extraction_done": False, "error": msg}
    finally:
        sys.modules.pop("extract_raw_data_multi", None)

    # ── Canonical output paths ────────────────────────────────────────
    body_csv      = os.path.join(work_dir, "raw_body_multi.csv")
    head_pose_csv = os.path.join(work_dir, "raw_head_pose_multi.csv")
    au_csv        = os.path.join(work_dir, "raw_action_units_multi.csv")
    gaze_csv      = os.path.join(work_dir, "raw_gaze_multi.csv")

    # ── Validate critical outputs ─────────────────────────────────────
    missing_critical = [p for p in [body_csv, head_pose_csv] if not os.path.isfile(p)]
    if missing_critical:
        msg = f"[Extraction] Critical CSVs missing: {missing_critical}"
        print(msg)
        return {**state,
                "raw_body_csv": body_csv, "raw_head_pose_csv": head_pose_csv,
                "raw_au_csv": au_csv,     "raw_gaze_csv": gaze_csv,
                "extraction_done": False, "error": msg}

    # ── Stub AU / Gaze if OpenFace was unavailable ────────────────────
    # merge_openface_outputs() writes these when OpenFace completes.
    # If OpenFace was skipped, create header-only stubs so the labeling
    # nodes degrade gracefully instead of crashing with FileNotFoundError.
    if not os.path.isfile(au_csv):
        _write_stub(au_csv,   _AU_INTENSITY_COLS + _AU_BINARY_COLS)
    if not os.path.isfile(gaze_csv):
        _write_stub(gaze_csv, _GAZE_COLS)

    print(f"\n[Node: Extraction] ✅ Extraction complete.")
    print(f"   body      → {body_csv}")
    print(f"   head_pose → {head_pose_csv}")
    print(f"   au        → {au_csv}")
    print(f"   gaze      → {gaze_csv}")

    return {
        **state,
        "raw_body_csv":      body_csv,
        "raw_head_pose_csv": head_pose_csv,
        "raw_au_csv":        au_csv,
        "raw_gaze_csv":      gaze_csv,
        "extraction_done":   True,
        "error":             None,
    }

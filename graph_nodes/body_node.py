import os
import sys
from pathlib import Path

# Add parent directory to path so we can import our new script
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def run_body_labeling(state: dict) -> dict:
    """
    LangGraph node: Computes upper-body behaviors based on raw_body_multi.csv.
    """
    try:
        from body_labeling_upper import process_body_data
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from body_labeling_upper import process_body_data
    raw_body_csv = state.get("raw_body_csv", "")
    work_dir     = state.get("work_dir", ".")

    print(f"\n{'='*60}")
    print(f"[Node: Body Labeling] Input : {raw_body_csv}")
    print(f"{'='*60}\n")

    if not os.path.isfile(raw_body_csv):
        msg = f"[Body] ERROR: raw body CSV not found → {raw_body_csv}"
        print(msg)
        return {"body_labeling_done": False, "error": msg}

    summary_csv = os.path.join(work_dir, "behaviour_summary.csv")
    raw_csv     = os.path.join(work_dir, "behaviour_raw_frames.csv")

    try:
        # We process it at 30 fps
        process_body_data(input_csv=raw_body_csv, summary_out=summary_csv, raw_out=raw_csv, fps=30.0)
    except Exception as exc:
        msg = f"[Body] Runtime error: {exc}"
        print(msg)
        return {"body_labeling_done": False, "error": msg}

    # Verify if CSVs were successfully created
    body_label_csv = summary_csv if os.path.isfile(summary_csv) else None
    body_raw_csv   = raw_csv if os.path.isfile(raw_csv) else None

    if not body_label_csv and not body_raw_csv:
        msg = "[Body] No output CSVs were produced."
        print(msg)
        return {"body_labeling_done": False, "error": msg}

    print(f"[Node: Body Labeling] ✓ Done. summary={summary_csv}")
    return {
        "body_label_csv":     body_label_csv,
        "body_raw_csv":       body_raw_csv,
        "body_labeling_done": True,
        "error":              None,
    }



"""
main_graph.py
=============
Description:
    The central orchestrator for the Pixie Behavioral Analysis Pipeline. 
    It leverages LangGraph to manage a directed acyclic graph (DAG) of 
    extraction, labeling, merging, preprocessing, LSTM prediction, 
    LLM analysis, and dashboard output nodes.
    
    The pipeline processes video data to generate:
      - full_analysis.csv (merged sensor data)
      - processed_features.npy (LSTM-ready features)
      - lstm_predictions.csv (engagement scores & risk levels)
      - Clinical behavioral report (via Groq LLM)
      - Interactive HTML dashboard with Flask API

Pipeline Topology (v3 — with preprocessor, LSTM, and Flask dashboard):

    tracking → openface_raw → headpose_raw → head_pose_labeling
             → gaze_labeling → au_labeling → body_labeling
             → face_recognition → merge_node → preprocessor_node
             → lstm_node → llm_analysis → output_node → save_report → END

Data Flow:
    [5 labeled CSVs] → merge_node → full_analysis.csv
                     → preprocessor_node → processed_features.npy
                     → lstm_node → lstm_predictions.csv (scores)
                     → llm_analysis → report_text (uses scores)
                     → output_node → Dashboard + Flask API + HTML
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Load .env (GROQ_API_KEY, GROQ_MODEL, etc.) before anything else ──────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
        print(f"[.env] Loaded → {_env_path}")
except ImportError:
    pass

# ── Make sure graph_nodes is importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import operator
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Annotated, Union

# ── Node imports ──────────────────────────────────────────────────────────────
from graph_nodes.extract_node            import run_extraction_node
from graph_nodes.head_pose_labeling_node import run_head_pose_labeling
from graph_nodes.body_node               import run_body_labeling
from graph_nodes.gaze_node               import run_gaze_labeling
from graph_nodes.action_units_node       import run_action_units_labeling
from graph_nodes.face_recognition_node   import run_face_recognition
from graph_nodes.merge_node              import run_merge_node
from graph_nodes.preprocessor_node       import run_preprocessor_node
from graph_nodes.lstm_node               import run_lstm_node
from graph_nodes.llm_analysis_node       import run_llm_analysis
from graph_nodes.output_node             import run_output_node


# ──────────────────────────────────────────────────────────────────────────────
# STATE SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

def add_errors(left: Optional[list[str]], right: Union[Optional[list[str]], str, None]) -> list[str]:
    res = list(left) if left is not None else []
    if right is not None:
        if isinstance(right, str):
            res.append(right)
        elif isinstance(right, list):
            res.extend(right)
    return res

class PipelineState(TypedDict, total=False):
    # ── Inputs ──
    video_path:      str
    work_dir:        str
    groq_api_key:    str
    groq_model:      str
    skip_extraction: bool

    # ── Synchronization ──
    tracking_done:    bool
    face_crops_dir:   str

    # ── Raw CSVs (extraction output) ──
    raw_body_csv:      str
    raw_head_pose_csv: str
    raw_au_csv:        str
    raw_gaze_csv:      str
    extraction_done:   bool
    
    # ── Face Recognition ──
    identity_map:     dict
    identity_map_csv: str

    # ── Labeled CSVs ──
    body_label_csv:    str
    body_raw_csv:      str
    head_label_csv:    str
    gaze_label_csv:    str
    au_label_csv:      str
    events_csv:        Optional[str]

    # ── Merge output ──
    full_analysis_csv:  str
    merge_done:         bool

    # ── Preprocessor output ──
    processed_features_npy:  str
    processed_metadata_csv:  str
    preprocessor_done:       bool

    # ── LSTM output ──
    lstm_predictions_csv: str
    lstm_done:            bool

    # ── Status flags ──
    body_labeling_done:    bool
    head_labeling_done:    bool
    gaze_labeling_done:    bool
    au_labeling_done:      bool
    llm_done:              bool
    output_done:           bool

    # ── LLM output ──
    report_text: str

    # ── Output node ──
    session_report_html:  str
    session_summary_json: str
    n_alerts:             int

    # ── Context ──
    teacher_context: str

    # ── Error ──
    error: Annotated[list[str], add_errors]


def save_report(state: PipelineState) -> PipelineState:
    """Final node: persist the clinical report to disk."""
    work_dir    = state.get("work_dir", ".")
    report_text = state.get("report_text", "")
    video_name  = Path(state.get("video_path", "video")).stem
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(work_dir, f"behavioral_report_{video_name}_{timestamp}.md")

    os.makedirs(work_dir, exist_ok=True)

    header = f"""# Pixie Behavioral Analysis Report
**Video:** {state.get("video_path", "N/A")}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {state.get("groq_model", "llama-3.3-70b-versatile")}
**Pipeline:** LangGraph + BiLSTM + Groq LLM (v3)

---

"""
    full_report = header + report_text

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    # ── Print session summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ Report saved → {report_path}")

    lstm_csv = state.get("lstm_predictions_csv", "")
    if lstm_csv and os.path.isfile(lstm_csv):
        print(f"  📊 LSTM predictions → {lstm_csv}")

    html_report = state.get("session_report_html", "")
    if html_report and os.path.isfile(html_report):
        print(f"  📄 Dashboard → {html_report}")

    n_alerts = state.get("n_alerts", 0)
    print(f"  🚨 Alerts: {n_alerts}")
    print(f"  🌐 Flask API: http://localhost:5050")
    print(f"{'='*60}\n")

    return {"report_path": report_path}


# ──────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ──────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # ── Register nodes ──
    graph.add_node("extraction",         run_extraction_node)
    
    graph.add_node("face_recognition",   run_face_recognition)
    graph.add_node("head_pose_labeling", run_head_pose_labeling)
    graph.add_node("gaze_labeling",      run_gaze_labeling)
    graph.add_node("body_labeling",      run_body_labeling)
    graph.add_node("au_labeling",        run_action_units_labeling)

    graph.add_node("merge",              run_merge_node)
    graph.add_node("preprocessor",       run_preprocessor_node)
    graph.add_node("lstm",               run_lstm_node)
    graph.add_node("llm_analysis",       run_llm_analysis)
    graph.add_node("output",             run_output_node)
    graph.add_node("save_report",        save_report)

    # ── Entry point ──
    graph.set_entry_point("extraction")

    # ── Sequential Pipeline ───────────────────────────────────────────────────
    # Extraction
    graph.add_edge("extraction",         "head_pose_labeling")
    
    # Labeling
    graph.add_edge("head_pose_labeling", "gaze_labeling")
    graph.add_edge("gaze_labeling",      "au_labeling")
    graph.add_edge("au_labeling",        "body_labeling")
    graph.add_edge("body_labeling",      "face_recognition")

    # Synthesis: merge → preprocess → lstm → llm → output
    graph.add_edge("face_recognition",   "merge")
    graph.add_edge("merge",              "preprocessor")
    graph.add_edge("preprocessor",       "lstm")
    graph.add_edge("lstm",               "llm_analysis")
    
    # Output
    graph.add_edge("llm_analysis",       "output")
    graph.add_edge("output",             "save_report")
    graph.add_edge("save_report",        END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pixie — Multimodal Behavioral Analysis Pipeline (LangGraph + BiLSTM + Groq)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video", "-v",
        default="/Users/sarahselmene/Desktop/langtarak/Pixie/aya2.mov",
        help="Path to the input video file",
    )
    parser.add_argument("--api-key", "-k", default=None, help="Groq API key")
    parser.add_argument("--model", "-m", default="llama-3.3-70b-versatile", help="Groq model ID")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction, use existing CSVs")
    parser.add_argument("--teacher-context", "-c", default="", help="Optional teacher context for LLM")
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = os.path.abspath(args.video)
    work_dir   = os.path.abspath(args.output_dir) if args.output_dir else str(Path(video_path).parent)
    api_key    = args.api_key or os.environ.get("GROQ_API_KEY", "")
    model      = args.model   or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    os.makedirs(work_dir, exist_ok=True)

    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          PIXIE — Behavioral Analysis Pipeline  v3            ║
║          LangGraph + BiLSTM + Groq LLM + Flask               ║
╚══════════════════════════════════════════════════════════════╝
  Video      : {video_path}
  Work dir   : {work_dir}
  Model      : {model}
  Skip extrac: {args.skip_extraction}
  API key    : {'✓ set' if api_key else '✗ MISSING'}

  Pipeline topology (v3):
    tracking → openface → headpose → labeling (head/gaze/au/body)
    → face_recognition → merge → preprocessor → lstm
    → llm_analysis → output (dashboard + Flask) → save_report → END
""")

    if not api_key:
        print("⚠️  WARNING: No Groq API key found. LLM analysis will fail.")

    initial_state: PipelineState = {
        "video_path":      video_path,
        "work_dir":        work_dir,
        "groq_api_key":    api_key,
        "groq_model":      model,
        "skip_extraction": args.skip_extraction,
        "teacher_context": args.teacher_context,
        "error":           None,
    }

    pipeline = build_graph()

    t0 = time.time()
    final_state = pipeline.invoke(initial_state)
    elapsed = time.time() - t0

    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)
    print(f"\n[Pipeline] Total elapsed: {int(h):02d}:{int(m):02d}:{s:05.2f}")

    if final_state.get("error"):
        print(f"\n[Pipeline] ⚠️  Completed with warnings/errors: {final_state['error']}")
        if not final_state.get("output_done"):
            sys.exit(1)
    
    print(f"\n[Pipeline] ✅ Done!")
    print(f"  📝 Clinical Report : {final_state.get('report_path', 'N/A')}")
    print(f"  📄 Dashboard       : {final_state.get('session_report_html', 'N/A')}")
    print(f"  📊 LSTM Predictions: {final_state.get('lstm_predictions_csv', 'N/A')}")
    print(f"  🚨 Alerts          : {final_state.get('n_alerts', 0)}")
    print(f"  🌐 Flask API       : http://localhost:5050")

    # Keep the process alive so Flask can serve the dashboard
    if final_state.get("output_done"):
        print(f"\n  💡 Dashboard is running at http://localhost:5050")
        print(f"     Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Pipeline] Shutting down.")


if __name__ == "__main__":
    main()

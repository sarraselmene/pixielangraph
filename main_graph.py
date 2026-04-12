#!/usr/bin/env python3
"""
main_graph.py
=============
Pixie Behavioral Analysis Pipeline — LangGraph Orchestrator
============================================================

Graph topology:
                         ┌─────────────────┐
                         │   extraction    │  (extract_raw_data2.py)
                         └────────┬────────┘
                                  │ raw CSVs
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
     │ head_pose    │   │ body_label   │   │  action_units    │
     └──────┬───────┘   └──────────────┘   └──────────────────┘
            │ labeled_head_pose_multi.csv
            ▼
     ┌──────────────┐
     │ gaze_label   │   (needs head pose for reliability check)
     └──────────────┘
              │
              └──────────── (all 4 labeled CSVs) ──────────────┐
                                                               ▼
                                                      ┌──────────────┐
                                                      │ llm_analysis │
                                                      └──────┬───────┘
                                                             │ report_text
                                                             ▼
                                                      ┌──────────────┐
                                                      │  save_report │
                                                      └──────────────┘

Usage
─────
    python main_graph.py --video /path/to/video.mov [--api-key YOUR_GROQ_KEY]

    # or set GROQ_API_KEY environment variable
    export GROQ_API_KEY=gsk_...
    python main_graph.py --video aya2.mov

    # Skip extraction (use existing raw CSVs):
    python main_graph.py --video aya2.mov --skip-extraction

    # Specify output directory:
    python main_graph.py --video aya2.mov --output-dir ./results
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
        load_dotenv(dotenv_path=_env_path, override=False)  # override=False: CLI args win
        print(f"[.env] Loaded → {_env_path}")
except ImportError:
    pass  # python-dotenv not installed — fall back to shell env vars

# ── Make sure graph_nodes is importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# ── Node imports ──────────────────────────────────────────────────────────────
from graph_nodes.extract_node       import run_extraction
from graph_nodes.body_node          import run_body_labeling
from graph_nodes.head_pose_node     import run_head_pose_labeling
from graph_nodes.gaze_node          import run_gaze_labeling
from graph_nodes.action_units_node  import run_action_units_labeling
from graph_nodes.llm_analysis_node  import run_llm_analysis


# ──────────────────────────────────────────────────────────────────────────────
# STATE SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    # ── Inputs ──
    video_path:      str
    work_dir:        str
    groq_api_key:    str
    groq_model:      str
    skip_extraction: bool

    # ── Raw CSVs (extraction output) ──
    raw_body_csv:      str
    raw_head_pose_csv: str
    raw_au_csv:        str
    raw_gaze_csv:      str
    extraction_done:   bool

    # ── Labeled CSVs ──
    body_label_csv:    str
    body_raw_csv:      str
    head_label_csv:    str
    gaze_label_csv:    str
    au_label_csv:      str
    events_csv:        Optional[str]

    # ── Status flags ──
    body_labeling_done:    bool
    head_labeling_done:    bool
    gaze_labeling_done:    bool
    au_labeling_done:      bool
    llm_done:              bool

    # ── LLM output ──
    report_text: str

    # ── Error ──
    error: Optional[str]


# ──────────────────────────────────────────────────────────────────────────────
# CONDITIONAL SKIP-EXTRACTION NODE
# ──────────────────────────────────────────────────────────────────────────────

def maybe_extract(state: PipelineState) -> PipelineState:
    """
    If --skip-extraction was passed (and raw CSVs already exist),
    populate the raw CSV paths from the work_dir and skip the heavy extraction.
    Otherwise, delegate to run_extraction.
    """
    if state.get("skip_extraction"):
        work_dir = state.get("work_dir", ".")
        body_csv      = os.path.join(work_dir, "raw_body_multi.csv")
        head_pose_csv = os.path.join(work_dir, "raw_head_pose_multi.csv")
        au_csv        = os.path.join(work_dir, "raw_action_units_multi.csv")
        gaze_csv      = os.path.join(work_dir, "raw_gaze_multi.csv")
        missing = [p for p in [body_csv, head_pose_csv, au_csv, gaze_csv]
                   if not os.path.isfile(p)]
        if missing:
            msg = f"[maybe_extract] --skip-extraction set but CSVs missing: {missing}"
            print(msg)
            return {**state,
                    "raw_body_csv": body_csv, "raw_head_pose_csv": head_pose_csv,
                    "raw_au_csv": au_csv, "raw_gaze_csv": gaze_csv,
                    "extraction_done": False, "error": msg}
        print("[maybe_extract] ✓ Skipping extraction — using existing raw CSVs.")
        return {**state,
                "raw_body_csv": body_csv, "raw_head_pose_csv": head_pose_csv,
                "raw_au_csv": au_csv, "raw_gaze_csv": gaze_csv,
                "extraction_done": True, "error": None}
    else:
        return run_extraction(state)


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
**Pipeline:** LangGraph Multimodal Behavioral Analysis

---

"""
    full_report = header + report_text

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\n{'='*60}")
    print(f"  ✅ Report saved → {report_path}")
    print(f"{'='*60}\n")
    print(full_report)

    return {**state, "report_path": report_path}


# ──────────────────────────────────────────────────────────────────────────────
# ROUTING HELPERS (conditional edges)
# ──────────────────────────────────────────────────────────────────────────────

def route_after_extraction(state: PipelineState) -> str:
    if not state.get("extraction_done"):
        print("[Router] Extraction failed — aborting pipeline.")
        return END
    return "head_pose_labeling"


def route_after_head_pose(state: PipelineState) -> str:
    if not state.get("head_labeling_done"):
        print("[Router] Head pose labeling failed — continuing without it.")
    # Always continue — gaze and AU can degrade gracefully without head pose
    return "gaze_labeling"


def route_after_gaze(state: PipelineState) -> str:
    # Always continue to body labeling
    return "body_labeling"


def route_after_body(state: PipelineState) -> str:
    # Always continue to AU labeling
    return "au_labeling"


def route_after_au(state: PipelineState) -> str:
    # Proceed to LLM — requires at least one labeled CSV
    any_data = any([
        state.get("body_label_csv"),
        state.get("head_label_csv"),
        state.get("gaze_label_csv"),
        state.get("au_label_csv"),
    ])
    if not any_data:
        print("[Router] No labeled data available — aborting LLM step.")
        return END
    return "llm_analysis"


def route_after_llm(state: PipelineState) -> str:
    if not state.get("llm_done"):
        print("[Router] LLM analysis failed.")
        return END
    return "save_report"


# ──────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ──────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # ── Register nodes ──
    graph.add_node("extraction",       maybe_extract)
    graph.add_node("head_pose_labeling", run_head_pose_labeling)
    graph.add_node("gaze_labeling",    run_gaze_labeling)
    graph.add_node("body_labeling",    run_body_labeling)
    graph.add_node("au_labeling",      run_action_units_labeling)
    graph.add_node("llm_analysis",     run_llm_analysis)
    graph.add_node("save_report",      save_report)

    # ── Entry point ──
    graph.set_entry_point("extraction")

    # ── Edges (sequential with conditional routing) ──
    graph.add_conditional_edges("extraction",       route_after_extraction,
                                {"head_pose_labeling": "head_pose_labeling", END: END})
    graph.add_conditional_edges("head_pose_labeling", route_after_head_pose,
                                {"gaze_labeling": "gaze_labeling"})
    graph.add_conditional_edges("gaze_labeling",    route_after_gaze,
                                {"body_labeling": "body_labeling"})
    graph.add_conditional_edges("body_labeling",    route_after_body,
                                {"au_labeling": "au_labeling"})
    graph.add_conditional_edges("au_labeling",      route_after_au,
                                {"llm_analysis": "llm_analysis", END: END})
    graph.add_conditional_edges("llm_analysis",     route_after_llm,
                                {"save_report": "save_report", END: END})
    graph.add_edge("save_report", END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pixie — Multimodal Behavioral Analysis Pipeline (LangGraph + Groq)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video", "-v",
        default="/Users/sarahselmene/Desktop/langtarak/Pixie/aya2.mov",
        help="Path to the input video file (default: aya2.mov in project dir)",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--model", "-m",
        default="llama-3.3-70b-versatile",
        help="Groq model ID (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to save all output CSVs and the report (default: same as video)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction step and use existing raw CSVs in --output-dir",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = os.path.abspath(args.video)
    work_dir   = os.path.abspath(args.output_dir) if args.output_dir else str(Path(video_path).parent)
    api_key    = args.api_key or os.environ.get("GROQ_API_KEY", "")
    model      = args.model   or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    os.makedirs(work_dir, exist_ok=True)

    # Add work_dir to sys.path so config.py is importable from there
    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)
    # Also add project root (where config.py lives)
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          PIXIE — Behavioral Analysis Pipeline                 ║
║          Powered by LangGraph + Groq LLM                      ║
╚══════════════════════════════════════════════════════════════╝
  Video      : {video_path}
  Work dir   : {work_dir}
  Model      : {args.model}
  Skip extrac: {args.skip_extraction}
  API key    : {'✓ set' if api_key else '✗ MISSING — LLM step will fail'}
""")

    if not api_key:
        print("⚠️  WARNING: No Groq API key found. Set GROQ_API_KEY or pass --api-key.")
        print("   The pipeline will run all extraction/labeling steps but the LLM")
        print("   analysis will fail. To test without a key run with --skip-extraction\n")

    initial_state: PipelineState = {
        "video_path":      video_path,
        "work_dir":        work_dir,
        "groq_api_key":    api_key,
        "groq_model":      model,
        "skip_extraction": args.skip_extraction,
        "error":           None,
    }

    pipeline = build_graph()

    t0 = time.time()
    final_state = pipeline.invoke(initial_state)
    elapsed = time.time() - t0

    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)
    print(f"\n[Pipeline] Total elapsed: {int(h):02d}:{int(m):02d}:{s:05.2f}")

    if final_state.get("error") and not final_state.get("llm_done"):
        print(f"\n[Pipeline] ⚠️  Completed with errors: {final_state['error']}")
        sys.exit(1)
    else:
        report_path = final_state.get("report_path", "N/A")
        print(f"\n[Pipeline] ✅ Done! Report → {report_path}")


if __name__ == "__main__":
    main()

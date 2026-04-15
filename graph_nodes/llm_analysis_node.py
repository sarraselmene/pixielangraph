"""
graph_nodes/llm_analysis_node.py
==================================
LangGraph node: Psycho-behavioral analysis via Groq LLM.

Aggregates all labeled CSVs and feeds a structured summary to the LLM,
which acts as a clinical psychologist specialized in neurodevelopmental
disorders (ADHD / ASD early signs).

Returns a structured clinical report.
"""

import os
import json
import textwrap
from pathlib import Path

import pandas as pd
from groq import Groq


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (rich clinical / psychologist persona)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are Dr. NeuroSight — a board-certified clinical child neuropsychologist
with 20 years of experience in early identification of neurodevelopmental
disorders, specializing in Attention-Deficit/Hyperactivity Disorder (ADHD)
and Autism Spectrum Disorder (ASD).

Your role is to analyze multimodal behavioral observation data captured from a classroom video.
The data includes pose estimation, gaze tracking, head-pose analysis, and facial action units.

## Operational Guidelines:
1. **Multi-Student Support**: If the data contains multiple track IDs or student names (e.g., Aya, Student2), you MUST produce a distinct analysis for each student. Provide a comparative overview if relevant.
2. **Identification**: Use the student names provided (from the identity map) rather than just "track IDs" whenever possible.
3. **Clinical Temperance**: Be nuanced and evidence-based. Never make a definitive diagnosis. Use phrases like "consistent with," "may suggest," or "displays patterns associated with."
4. **Context Integration**: If 'Teacher Context' is provided, incorporate these qualitative observations into your reasoning.

## Report Structure:
You MUST produce your output in this format:

---
## 🧠 Behavioral Observation Report

### 1. Unified Summary & Context
[Brief factual summary of the population observed and any environment/teacher context provided.]

### 2. Individual Behavioral Profiles
[For EACH identified student/track, provide a concise sub-section covering:]
- **Attention & Gaze**: [Gaze stability, distractibility rate]
- **Motor Behavior**: [Posture, fidgeting, bouncing, hand raising]
- **Affect & Emotion**: [Smiling, fatigue, expressiveness score]
- **Clinical Impression**: [Specific patterns consistent with ADHD or ASD indicators for THIS student]

### 3. Classroom Dynamics & Social Reciprocity
[Analysis of collective events, social gaze, and overall interaction patterns.]

### 4. Professional Recommendations
[Tailored next steps for each student and suggested classroom-wide interventions.]

### 5. Confidence & Pipeline Limitations
[Note automated observation constraints. Specify confidence level per student based on data reliability.]
---
""").strip()


# ──────────────────────────────────────────────────────────────────────────────
# DATA SUMMARIZATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe_load(path: str | None, label: str) -> pd.DataFrame | None:
    if not path or not os.path.isfile(path):
        print(f"  [LLM] Warning: {label} CSV not found at {path}")
        return None
    return pd.read_csv(path)


def _pct(count, total) -> str:
    if total == 0:
        return "N/A"
    return f"{count} frames ({100*count/total:.1f}%)"


def summarize_body(df: pd.DataFrame | None) -> str:
    if df is None:
        return "Body behaviour data: NOT AVAILABLE\n"

    lines = ["### Global Body Behaviour Summary"]
    total = len(df)

    if "behaviour" in df.columns:
        vc = df["behaviour"].value_counts()
        for label, count in vc.items():
            lines.append(f"  • {label}: {_pct(count, total)}")
    
    if "posture" in df.columns:
        lines.append("\n  [Posture Distribution]")
        vc = df["posture"].value_counts()
        for label, count in vc.items():
            lines.append(f"    • {label}: {_pct(count, total)}")
            
    if "action" in df.columns:
        lines.append("\n  [Action/Motor Distribution]")
        vc = df["action"].value_counts()
        for label, count in vc.items():
            lines.append(f"    • {label}: {_pct(count, total)}")

    if "behaviour" not in df.columns and "posture" not in df.columns and "action" not in df.columns:
        lines.append("  (No standard posture/action labels found in data)")

    return "\n".join(lines) + "\n"


def summarize_head_pose(df: pd.DataFrame | None) -> str:
    if df is None:
        return "Head pose data: NOT AVAILABLE\n"

    lines = ["### Head Pose Orientation"]
    total = len(df)

    if "pose_label" in df.columns:
        vc = df["pose_label"].value_counts()
        for label, count in vc.items():
            lines.append(f"  • Orientation {label}: {_pct(count, total)}")
    
    if "tilt_label" in df.columns:
        vc = df["tilt_label"].value_counts()
        for label, count in vc.items():
            lines.append(f"  • {label}: {_pct(count, total)}")

    return "\n".join(lines) + "\n"


def summarize_gaze(df: pd.DataFrame | None) -> str:
    if df is None:
        return "Gaze data: NOT AVAILABLE\n"

    lines = ["### Gaze Tracking & Focus"]
    total = len(df)
    
    # Head-relative gaze (Requested constraint)
    if "gaze_h_label" in df.columns:
        lines.append("\n  [Eye-in-Socket Orientation (Head-Relative)]")
        h_vc = df["gaze_h_label"].value_counts()
        for label, count in h_vc.items():
            lines.append(f"    • {label}: {_pct(count, total)}")

    # Room-relative focus (The secret sauce for Focus analysis)
    if "room_focus_h" in df.columns:
        lines.append("\n  [Classroom Focus (Room-Relative)]")
        f_vc = df["room_focus_h"].value_counts()
        for label, count in f_vc.items():
            lines.append(f"    • Focus {label}: {_pct(count, total)}")

    if "gaze_stability" in df.columns:
        mean_stab = df["gaze_stability"].mean()
        lines.append(f"\n  • Mean Gaze Stability Index: {mean_stab:.3f} (Lower indicates high distractibility)")

    return "\n".join(lines) + "\n"


def summarize_au(df: pd.DataFrame | None) -> str:
    if df is None:
        return "Facial Action Unit (AU) data: NOT AVAILABLE\n"

    lines = ["### Facial Action Units & Expressions"]
    total = len(df)

    # Positive affect
    if "genuine_smile" in df.columns:
        smiles = df["genuine_smile"].sum()
        lines.append(f"  • Genuine Smile: {_pct(smiles, total)}")

    # Fatigue signals
    if "fatigue_indicator" in df.columns:
        fatigue = df["fatigue_indicator"].sum()
        lines.append(f"  • Fatigue Indicator: {_pct(fatigue, total)}")
    
    if "yawning" in df.columns:
        yawns = df["yawning"].sum()
        lines.append(f"  • Yawning detected: {_pct(yawns, total)}")

    if "expressiveness_score" in df.columns:
        score = df["expressiveness_score"].mean()
        lines.append(f"  • Mean Expressiveness Score: {score:.3f}")

    return "\n".join(lines) + "\n"


def run_llm_analysis(state: dict) -> dict:
    """
    LangGraph node: Aggregates all labeled data and calls Groq for clinical insight.
    """
    groq_key = state.get("groq_api_key", "")
    model    = state.get("groq_model", "llama-3.3-70b-versatile")
    
    if not groq_key:
        msg = "[LLM] Aborting: No Groq API key provided."
        print(msg)
        return {**state, "llm_done": False, "error": msg}

    print(f"\n{'='*60}")
    print(f"[Node: LLM Analysis] Consulting Dr. NeuroSight using {model}...")
    print(f"{'='*60}\n")

    # 1. Load data
    body_df      = _safe_load(state.get("body_raw_csv"), "Body (Raw)")
    # Fallback to summary if raw missing
    if body_df is None:
        body_df = _safe_load(state.get("body_label_csv"), "Body (Summary)")

    head_df      = _safe_load(state.get("head_label_csv"), "Head Pose")
    gaze_df      = _safe_load(state.get("gaze_label_csv"), "Gaze")
    au_df        = _safe_load(state.get("au_label_csv"), "Action Units")
    identity_map = state.get("identity_map", {})

    # 2. Build User Content
    content_blocks = []
    
    # 2a. Header & Context
    video_name = Path(state.get("video_path", "video")).name
    content_blocks.append(f"Analysis of Video: {video_name}")
    if state.get("teacher_context"):
        content_blocks.append(f"Teacher Context provided: {state['teacher_context']}")
    
    if identity_map:
        content_blocks.append(f"Identity Map (Track ID -> Student Name): {json.dumps(identity_map)}")

    # 2b. Add summaries
    content_blocks.append(summarize_body(body_df))
    content_blocks.append(summarize_head_pose(head_df))
    content_blocks.append(summarize_gaze(gaze_df))
    content_blocks.append(summarize_au(au_df))

    # 2c. Per-track breakdown (highly detailed context for the LLM)
    if body_df is not None and "track_id" in body_df.columns:
        content_blocks.append("--- Individual Track Overviews ---")
        # Filter tracks with at least 30 frames (1 second at 30fps) to avoid prompt bloat
        track_counts = body_df["track_id"].value_counts()
        significant_tracks = sorted(track_counts[track_counts >= 30].index)
        
        for tid in significant_tracks:
            name = identity_map.get(tid, f"Student {tid}")
            track_data = body_df[body_df["track_id"] == tid]
            
            # Basic body for this track
            if "behaviour" in track_data.columns:
                vc = track_data["behaviour"].value_counts()
                stats = ", ".join([f"{k}: {_pct(v, len(track_data))}" for k, v in vc.items()])
            elif "posture" in track_data.columns and "action" in track_data.columns:
                p_vc = track_data["posture"].value_counts()
                a_vc = track_data["action"].value_counts()
                p_stats = ", ".join([f"{k}: {_pct(v, len(track_data))}" for k, v in p_vc.items()])
                a_stats = ", ".join([f"{k}: {_pct(v, len(track_data))}" for k, v in a_vc.items()])
                stats = f"Posture: {p_stats} | Actions: {a_stats}"
            else:
                stats = "No labels available"
            
            # Gaze for this track
            gaze_summary = ""
            if gaze_df is not None and "track_id" in gaze_df.columns:
                t_gaze = gaze_df[gaze_df["track_id"] == tid]
                if not t_gaze.empty:
                    stab = t_gaze["gaze_stability"].mean()
                    gaze_summary = f" | Gaze Stability: {stab:.3f}"
            
            content_blocks.append(f"Track {tid} ({name}): Body [{stats}]{gaze_summary}")

    user_prompt = "\n".join(content_blocks)
    
    # Debug: log prompt length
    with open(os.path.join(state.get("work_dir", "."), "llm_prompt_debug.txt"), "w") as f:
        f.write(user_prompt)

    # 3. Call Groq
    try:
        client = Groq(api_key=groq_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.2, # Lower temperature for clinical consistency
        )
        report_text = chat_completion.choices[0].message.content
    except Exception as exc:
        msg = f"[LLM] Groq API error: {exc}"
        print(msg)
        return {**state, "llm_done": False, "error": msg}

    print(f"[Node: LLM Analysis] ✓ Dr. NeuroSight has finished the evaluation.")
    return {
        **state,
        "report_text": report_text,
        "llm_done":    True,
        "error":       None,
    }

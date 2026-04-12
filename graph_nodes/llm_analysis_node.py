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

Your role in this session is to analyze multimodal behavioral observation data
captured from a classroom video of a student. The data was extracted using
computer-vision pipelines (pose estimation, gaze tracking, head-pose analysis,
and facial action unit recognition). You will receive a structured behavioral
summary and must produce a rigorous clinical-style behavioral report.

## Your analytical framework includes:

### ADHD Early Indicators:
- Sustained attention difficulties (gaze instability, frequent off-task gaze)
- Hyperactivity markers: excessive fidgeting, leg shaking, body bouncing,
  bounding movements, hand/arm restlessness
- Impulsivity cues: abrupt posture changes, frequent hand-raising without
  settling, high bounding frequency
- Distractibility: frequent head turns (left/right yaw), gaze wandering

### ASD Early Indicators:
- Reduced social gaze (persistent downward / avoidant gaze)
- Low or absent smiling / emotional expressiveness
- Stereotyped repetitive motor behaviors (rhythmic rocking, leg shaking)
- Rigid posturing (sustained slouching without variation)
- Reduced reciprocal social signals (low collective smile participation)
- Flat affective expression (low expressiveness score)

### Fatigue / Emotional State Signals:
- Drowsiness: high yawning frequency, fatigue indicator activation,
  downward gaze sustained
- Disengagement: low expressiveness, persistent slouching, downward gaze
- Stress / anxiety: elevated AU04 (brow lowerer), AU07 (lid tightener),
  frequent AU20 (lip stretcher)

## Report Structure:
You MUST produce your output in this exact format:

---
## 🧠 Behavioral Observation Report

### 1. Observation Summary
[Brief factual summary of the behavioral data provided]

### 2. Attention & Gaze Analysis
[Analysis of gaze patterns, stability, head orientation — link to ADHD/ASD attention markers]

### 3. Motor Behavior Analysis
[Analysis of posture labels, fidgeting, bouncing, bounding, hand raising — link to ADHD hyperactivity/impulsivity]

### 4. Facial Expression & Affective Analysis
[Analysis of AUs: smiling, fatigue, yawning, expressiveness score — link to ASD social affect and emotional state]

### 5. Clinical Impressions
[Synthesized clinical interpretation — what patterns, if any, are consistent with early ADHD or ASD indicators. Be nuanced and evidence-based. Avoid definitive diagnosis.]

### 6. Recommendations
[Practical next steps for further evaluation, classroom support strategies, or caregiver guidance]

### 7. Confidence & Limitations
[Note the limitations of automated behavioral observation vs. clinical interview. Describe confidence level in your impressions.]
---

IMPORTANT: You must always be nuanced, evidence-based, and respectful. Never
make a definitive diagnosis. Frame impressions as "consistent with" or
"may suggest" rather than absolute claims. Treat this as one data point
in a larger clinical picture.
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

    lines = ["### Body Behaviour (Posture & Motor)"]

    if "behaviour" in df.columns:
        # Raw frames CSV
        total = len(df)
        vc = df["behaviour"].value_counts()
        lines.append(f"Total body-frame records: {total}")
        for label, count in vc.items():
            lines.append(f"  • {label}: {_pct(count, total)}")
    elif "behaviour_label" in df.columns or "label" in df.columns:
        col = "behaviour_label" if "behaviour_label" in df.columns else "label"
        total = len(df)
        vc = df[col].value_counts()
        lines.append(f"Total body-frame records: {total}")
        for label, count in vc.items():
            lines.append(f"  • {label}: {_pct(count, total)}")
    else:
        # summary CSV — just show all rows as key-value
        for _, row in df.iterrows():
            lines.append(f"  • {dict(row)}")

    return "\n".join(lines) + "\n"


def summarize_head_pose(df: pd.DataFrame | None) -> str:
    if df is None:
        return "Head pose data: NOT AVAILABLE\n"

    lines = ["### Head Pose Orientation"]
    total = len(df)
    lines.append(f"Total head-pose records: {total}")

    if "pose_label" in df.columns:
        vc = df["pose_label"].value_counts()
        for label, count in vc.items():
            lines.append(f"  • {label}: {_pct(count, total)}")

    if "tilt_label" in df.columns:
        vc2 = df["tilt_label"].value_counts()
        lines.append("Head Tilt:")
        for label, count in vc2.items():
            lines.append(f"  • {label}: {_pct(count, total)}")

    if "confidence" in df.columns:
        mean_conf = df["confidence"].mean()
        lines.append(f"Mean pose confidence: {mean_conf:.3f}")

    # Per-track breakdown
    if "track_id" in df.columns:
        for tid in sorted(df["track_id"].unique()):
            t = df[df["track_id"] == tid]
            n = len(t)
            if "pose_label" in t.columns:
                dominant = t["pose_label"].value_counts().idxmax()
                lines.append(f"  Track {tid} dominant pose: {dominant} (n={n})")

    return "\n".join(lines) + "\n"


def summarize_gaze(df: pd.DataFrame | None) -> str:
    if df is None:
        return "Gaze data: NOT AVAILABLE\n"

    lines = ["### Gaze Direction & Stability"]
    total = len(df)
    reliable = df["openface_reliable"].sum() if "openface_reliable" in df.columns else total
    lines.append(f"Total gaze records: {total} | Reliable: {reliable}")

    for col, label in [("gaze_h_label", "Horizontal Gaze"), ("gaze_v_label", "Vertical Gaze")]:
        if col in df.columns:
            vc = df[col].value_counts()
            lines.append(f"{label}:")
            for lbl, count in vc.items():
                lines.append(f"  • {lbl}: {_pct(count, total)}")

    if "gaze_stability" in df.columns:
        mean_stab = df["gaze_stability"].mean()
        lines.append(f"Mean gaze stability: {mean_stab:.3f} (1.0=very stable, 0.0=unstable)")

    if "eye_head_divergence" in df.columns:
        mean_div = df["eye_head_divergence"].dropna().mean()
        lines.append(f"Mean eye-head divergence: {mean_div:.3f} rad")

    # Per-track
    if "track_id" in df.columns:
        for tid in sorted(df["track_id"].unique()):
            t = df[df["track_id"] == tid]
            parts = []
            if "gaze_h_label" in t.columns:
                h_dom = t["gaze_h_label"].value_counts().idxmax()
                parts.append(f"h_dominant={h_dom}")
            if "gaze_v_label" in t.columns:
                v_dom = t["gaze_v_label"].value_counts().idxmax()
                parts.append(f"v_dominant={v_dom}")
            if "gaze_stability" in t.columns:
                parts.append(f"stability={t['gaze_stability'].mean():.3f}")
            lines.append(f"  Track {tid}: {', '.join(parts)}")

    return "\n".join(lines) + "\n"


def summarize_action_units(df: pd.DataFrame | None, events_csv: str | None) -> str:
    if df is None:
        return "Action unit data: NOT AVAILABLE\n"

    lines = ["### Facial Action Units & Expressions"]
    total = len(df)
    reliable = df["openface_reliable"].sum() if "openface_reliable" in df.columns else total
    lines.append(f"Total AU records: {total} | Reliable: {reliable}")

    bool_cols = {
        "genuine_smile":     "Genuine Smile",
        "fatigue_indicator": "Fatigue Indicator",
        "yawning":           "Yawning",
        "talking_flag":      "Talking (speech filter)",
    }
    for col, label in bool_cols.items():
        if col in df.columns:
            count = df[col].sum()
            lines.append(f"  • {label}: {_pct(count, total)}")

    if "expressiveness_score" in df.columns:
        rel_df = df[df["openface_reliable"]] if "openface_reliable" in df.columns else df
        mean_expr = rel_df["expressiveness_score"].mean()
        lines.append(f"Mean expressiveness score: {mean_expr:.3f} (0=flat, 1=very expressive)")

    # Per-track
    if "track_id" in df.columns:
        for tid in sorted(df["track_id"].unique()):
            t = df[df["track_id"] == tid]
            t_rel = t[t["openface_reliable"]] if "openface_reliable" in t.columns else t
            parts = []
            for col, label in bool_cols.items():
                if col in t.columns:
                    parts.append(f"{col}={int(t[col].sum())}")
            if "expressiveness_score" in t_rel.columns:
                parts.append(f"expr_mean={t_rel['expressiveness_score'].mean():.3f}")
            lines.append(f"  Track {tid}: {', '.join(parts)}")

    # Collective events
    if events_csv and os.path.isfile(events_csv):
        ev_df = pd.read_csv(events_csv)
        lines.append(f"Collective smile events: {len(ev_df)}")

    return "\n".join(lines) + "\n"


def build_user_prompt(state: dict) -> str:
    """Build the structured behavioral summary to send to the LLM."""

    body_df      = _safe_load(state.get("body_label_csv")  or state.get("body_raw_csv"), "Body")
    head_df      = _safe_load(state.get("head_label_csv"), "Head Pose")
    gaze_df      = _safe_load(state.get("gaze_label_csv"), "Gaze")
    au_df        = _safe_load(state.get("au_label_csv"), "Action Units")
    events_csv   = state.get("events_csv")

    video_name = Path(state.get("video_path", "unknown.mov")).name

    prompt = textwrap.dedent(f"""
    ## Behavioral Observation Data
    **Video source:** {video_name}
    **Analysis pipeline:** LangGraph multimodal pipeline (YOLO + OpenFace + 6DRepNet)

    Please analyze the following behavioral signals observed in this video and
    produce a clinical behavioral report as instructed.

    ---

    {summarize_body(body_df)}
    {summarize_head_pose(head_df)}
    {summarize_gaze(gaze_df)}
    {summarize_action_units(au_df, events_csv)}

    ---
    Please provide your structured clinical behavioral report now.
    """).strip()

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def run_llm_analysis(state: dict) -> dict:
    """
    LangGraph node: Analyze labeled behavioral data with a Groq LLM.

    Expects state keys:
        groq_api_key    (str): Groq API key
        groq_model      (str): Model ID (default: llama-3.3-70b-versatile)
        body_label_csv  (str | None)
        head_label_csv  (str | None)
        gaze_label_csv  (str | None)
        au_label_csv    (str | None)
        events_csv      (str | None)
        video_path      (str)

    Produces state keys:
        report_text     (str): Full markdown clinical report
        llm_done        (bool)
        error           (str | None)
    """
    api_key   = state.get("groq_api_key") or os.environ.get("GROQ_API_KEY", "")
    model_id  = state.get("groq_model", "llama-3.3-70b-versatile")

    print(f"\n{'='*60}")
    print(f"[Node: LLM Analysis] Model: {model_id}")
    print(f"{'='*60}\n")

    if not api_key:
        msg = "[LLM] ERROR: No GROQ_API_KEY found in state or environment."
        print(msg)
        return {**state, "llm_done": False, "error": msg}

    user_prompt = build_user_prompt(state)
    print("[Node: LLM Analysis] User prompt built. Sending to Groq...")
    print("─" * 40)
    print(user_prompt[:800] + "...[truncated for display]")
    print("─" * 40)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        report_text = response.choices[0].message.content
    except Exception as exc:
        msg = f"[LLM] Groq API error: {exc}"
        print(msg)
        return {**state, "llm_done": False, "error": msg}

    print("\n[Node: LLM Analysis] ✓ Report received.")
    return {
        **state,
        "report_text": report_text,
        "llm_done":    True,
        "error":       None,
    }

import os
import glob
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --------------------------------------------------------------------------------
# PAGE CONFIG FOR PREMIUM AESTHETICS (Must be first Streamlit command)
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Pixie | Clinical Behavior Analytics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# CACHED DATA LOADING
# --------------------------------------------------------------------------------
@st.cache_data
def load_data(work_dir="."):
    """Loads behavior summary, face map, and gaze labels."""
    summary_path = os.path.join(work_dir, "behaviour_summary.csv")
    map_path = os.path.join(work_dir, "face_identity_map.csv")
    # Corrected path to match LangGraph output
    gaze_path = os.path.join(work_dir, "labeled_gaze_multi.csv")
    au_path = os.path.join(work_dir, "labeled_action_units_multi.csv")
    
    # 1. Load behavior summary
    df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    if not df.empty:
        df["duration_sec"] = df["duration_frames"] / 30.0

    # 2. Add Student Names via Identity Map
    identity_map = {}
    if os.path.exists(map_path):
        idf = pd.read_csv(map_path)
        identity_map = dict(zip(idf["track_id"], idf["student_name"]))
    
    if not df.empty:
        df["Student"] = df["track_id"].map(lambda tid: identity_map.get(tid, f"Student {tid}"))
        
    # 3. Load Gaze Labels
    gaze_df = pd.read_csv(gaze_path) if os.path.exists(gaze_path) else pd.DataFrame()
    if not gaze_df.empty:
        gaze_df["Student"] = gaze_df["track_id"].map(lambda tid: identity_map.get(tid, f"Student {tid}"))

    # 4. Load AU Labels
    au_df = pd.read_csv(au_path) if os.path.exists(au_path) else pd.DataFrame()
    if not au_df.empty:
        au_df["Student"] = au_df["track_id"].map(lambda tid: identity_map.get(tid, f"Student {tid}"))

    return df, gaze_df, au_df, identity_map

def get_latest_report(work_dir="."):
    files = glob.glob(os.path.join(work_dir, "behavioral_report_*.md"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# --------------------------------------------------------------------------------
# CUSTOM CSS / PREMIUM STYLING (Glassmorphism + HSL)
# --------------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --highlight: #7b2cbf;
        }

        html, body, [class*="css"] { 
            font-family: 'Outfit', sans-serif; 
            color: #f8f9fa;
        }
        
        /* Glassmorphism containers */
        div[data-testid="metric-container"] {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            padding: 1.5rem;
            border-radius: 1.2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(8px);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: var(--highlight);
            background: rgba(123, 44, 191, 0.07);
        }
        
        .premium-title {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #9d4edd, #c77dff, #e0aaff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #b197fc;
            margin-bottom: 2.5rem;
            font-weight: 300;
        }

        .stForm {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 1rem;
            padding: 2rem;
        }

        /* Gauge custom labels */
        .gauge-label {
            font-size: 0.9rem;
            color: #adb5bd;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# MAIN DASHBOARD APP
# --------------------------------------------------------------------------------
def main():
    inject_custom_css()
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8636/8636254.png", width=80)
    st.sidebar.markdown("## PIXIE INSIGHTS")
    st.sidebar.caption("Multimodal Clinical Observation")
    
    work_dir = st.sidebar.text_input("Work Directory", value=".")
    
    df, gaze_df, au_df, identity_map = load_data(work_dir)
    
    # Header
    st.markdown('<div class="premium-title">Pixie Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evidence-based behavioral insights for neurodevelopmental support</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning(f"No behavior data found in `{work_dir}`. Please run the tracking pipeline.")
        return

    # --- ROW 1 : CLASSROOM OVERVIEW GAUGES ---
    st.subheader("📊 Classroom Dynamics")
    c1, c2, c3 = st.columns(3)
    
    # 1. Collective Attention (Gaze Center Rate)
    if not gaze_df.empty:
        reliable_gaze = gaze_df[gaze_df["openface_reliable"]]
        center_rate = (reliable_gaze["room_focus_h"] == "Center").mean() if not reliable_gaze.empty else 0.0
        
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = center_rate * 100,
            title = {'text': "Collective Focus", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#7b2cbf"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(255, 0, 0, 0.2)"},
                    {'range': [40, 70], 'color': "rgba(255, 255, 0, 0.1)"},
                    {'range': [70, 100], 'color': "rgba(0, 255, 0, 0.1)"}
                ],
            }
        ))
        fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#ececec"})
        c1.plotly_chart(fig_g, width="stretch")

    # 2. Collective Affect (Smile Rate)
    if not au_df.empty:
        smile_rate = au_df["genuine_smile"].mean() if "genuine_smile" in au_df.columns else 0.0
        fig_s = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = smile_rate * 100,
            title = {'text': "Positive Affect", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 20], 'tickwidth': 1},
                'bar': {'color': "#ffca3a"},
            }
        ))
        fig_s.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#ececec"})
        c2.plotly_chart(fig_s, width="stretch")

    # 3. Kinetic Activity (Fidgeting/Bouncing rate)
    kinetic_rate = df[df["behaviour"].isin(["fidgeting", "bouncing"])]["duration_sec"].sum() / df["duration_sec"].sum()
    fig_k = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = kinetic_rate * 100,
        title = {'text': "Classroom Kineticity", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 50], 'tickwidth': 1},
            'bar': {'color': "#ff9e00"},
        }
    ))
    fig_k.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#ececec"})
    c3.plotly_chart(fig_k, width="stretch")

    # --- ROW 2 : INDIVIDUAL ATTENTION FLAGS ---
    st.divider()
    st.subheader("🔍 Individual Attention Scan")
    if gaze_df.empty:
        st.info("No gaze tracking data available.")
    else:
        students = gaze_df["Student"].unique()
        flags_cols = st.columns(len(students))
        for idx, student in enumerate(students):
            s_df = gaze_df[(gaze_df["Student"] == student) & (gaze_df["openface_reliable"] == True)]
            with flags_cols[idx]:
                st.markdown(f"**{student}**")
                if s_df.empty:
                    st.caption("No reliable data.")
                else:
                    gaze_stability = s_df["gaze_stability"].mean()
                    off_task_rate = (s_df["room_focus_h"] != "Center").mean()
                    
                    color = "normal"
                    if gaze_stability < 0.4: color = "inverse"
                    
                    st.metric("Focus Stability", f"{gaze_stability:.2f}", delta=f"{off_task_rate*100:.0f}% Off-task", delta_color=color)
                    
                    if gaze_stability < 0.5 or off_task_rate > 0.4:
                        st.warning("⚠️ Attention Alert")
                    else:
                        st.success("✅ Stable Focus")

    # --- ROW 3 : TIMELINE ---
    st.divider()
    st.subheader("🕰️ Behavioral Timeline")
    df["Start_Time"] = pd.to_datetime(df["start_frame"] * (1000/30), unit="ms")
    df["End_Time"] = pd.to_datetime(df["end_frame"] * (1000/30), unit="ms")
    
    # Updated color map including slouching
    color_discrete_map = {
        "sitting": "#4cc9f0", 
        "standing": "#f72585", 
        "slouching": "#3a0ca3", 
        "bouncing": "#ffca3a", 
        "hand_raised": "#4f772d",
        "fidgeting": "#fb8500"
    }
            
    fig_timeline = px.timeline(
        df, x_start="Start_Time", x_end="End_Time", y="Student", 
        color="behaviour", color_discrete_map=color_discrete_map, 
        hover_name="behaviour", template="plotly_dark"
    )
    fig_timeline.update_yaxes(autorange="reversed")
    fig_timeline.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
        font=dict(family="Outfit", color="#ececec"), 
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_timeline, width="stretch")

    # --- ROW 4 : TEACHER CONTEXT & REPORT GENERATION ---
    st.sidebar.divider()
    st.sidebar.subheader("📝 Context Integration")
    teacher_ctx = st.sidebar.text_area("Observer Notes", height=150, placeholder="e.g. 'Aya seems tired today', 'Noise level was high'")
    
    if st.sidebar.button("✨ Run Clinical Analysis"):
        with st.spinner("Dr. NeuroSight is reviewing the findings..."):
            try:
                cmd = ["python3", "main_graph.py", "--skip-extraction"]
                if teacher_ctx.strip():
                    cmd.extend(["--teacher-context", teacher_ctx.strip()])
                
                res = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
                
                if res.returncode != 0:
                    st.error(f"Analysis failed!\n{res.stderr or res.stdout}")
                else:
                    st.success("New report generated!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # Show latest report
    st.divider()
    latest_md = get_latest_report(work_dir)
    if latest_md:
        st.subheader("📄 Clinical Observation Report")
        with open(latest_md, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(f'<div style="background:rgba(255,255,255,0.02); padding:2rem; border-radius:1rem; border:1px solid rgba(255,255,255,0.05);">{content}</div>', unsafe_allow_html=True)
            

if __name__ == "__main__":
    main()

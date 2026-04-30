"""
graph_nodes/output_node.py
==========================
Final LangGraph node: Generates the HTML dashboard + launches a Flask
API server for interactive session browsing.

Produces:
  1. Static HTML session report (always)
  2. Session summary JSON (always)
  3. Flask dashboard server (optional, auto-starts)

The Flask server exposes:
  GET  /                    → Interactive dashboard
  GET  /api/session         → Full session summary JSON
  GET  /api/predictions     → LSTM predictions as JSON
  GET  /api/tracks          → Per-track engagement data
  GET  /api/report          → Clinical report text
"""

import os
import json
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


# ── Alert thresholds ──────────────────────────────────────────────────────────
CONSECUTIVE_HIGH_RISK_FRAMES = 10
ALERT_ENGAGEMENT_THRESHOLD = 0.3


def _load_csv_safe(path: str) -> pd.DataFrame:
    if path and os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def _detect_alerts(predictions_df: pd.DataFrame) -> list[dict]:
    """Scan LSTM predictions for sustained high-risk episodes."""
    alerts = []
    if predictions_df.empty:
        return alerts

    for tid in predictions_df["track_id"].unique():
        track_df = predictions_df[predictions_df["track_id"] == tid].sort_values("frame_id").reset_index(drop=True)

        consecutive_high = 0
        alert_start = None
        run_engagements = []

        for idx, row in track_df.iterrows():
            if row.get("risk_level", "low") == "high":
                if consecutive_high == 0:
                    alert_start = int(row["frame_id"])
                    run_engagements = []
                consecutive_high += 1
                if "engagement_score" in track_df.columns:
                    run_engagements.append(float(row["engagement_score"]))

                if consecutive_high >= CONSECUTIVE_HIGH_RISK_FRAMES:
                    avg_eng = float(np.nanmean(run_engagements)) if run_engagements else 0.0
                    alerts.append({
                        "track_id": int(tid),
                        "type": "sustained_high_risk",
                        "start_frame": alert_start,
                        "end_frame": int(row["frame_id"]),
                        "duration_windows": consecutive_high,
                        "avg_engagement": round(avg_eng, 4),
                        "severity": "critical" if consecutive_high > 20 else "warning",
                    })
                    consecutive_high = 0
                    run_engagements = []
            else:
                consecutive_high = 0
                alert_start = None
                run_engagements = []

        if not track_df.empty and "engagement_score" in track_df.columns:
            avg_eng = track_df["engagement_score"].mean()
            if avg_eng < ALERT_ENGAGEMENT_THRESHOLD:
                alerts.append({
                    "track_id": int(tid),
                    "type": "low_overall_engagement",
                    "avg_engagement": round(float(avg_eng), 3),
                    "severity": "warning",
                })

    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DASHBOARD HTML (with embedded JS for API calls)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_dashboard_html(
    predictions_df: pd.DataFrame,
    alerts: list[dict],
    report_text: str,
    identity_map: dict,
    video_path: str,
    api_port: int = 5050,
) -> str:
    """Generate a rich interactive dashboard HTML with Chart.js visualizations."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_name = Path(video_path).stem if video_path else "Unknown"

    # Build per-track data for charts
    tracks_data = {}
    if not predictions_df.empty:
        for tid in sorted(predictions_df["track_id"].unique()):
            t_df = predictions_df[predictions_df["track_id"] == tid]
            name = identity_map.get(int(tid), identity_map.get(str(tid), f"Student {tid}"))
            avg_eng = float(t_df["engagement_score"].mean()) if "engagement_score" in t_df.columns else 0
            risk_dist = t_df["risk_level"].value_counts().to_dict() if "risk_level" in t_df.columns else {}
            track_alerts = [a for a in alerts if a.get("track_id") == int(tid)]

            # Engagement timeline (downsample for chart)
            timeline = []
            if "engagement_score" in t_df.columns:
                step = max(1, len(t_df) // 200)
                for _, row in t_df.iloc[::step].iterrows():
                    timeline.append({
                        "frame": int(row["frame_id"]),
                        "score": round(float(row["engagement_score"]), 3),
                    })

            tracks_data[str(tid)] = {
                "name": name, "avg_engagement": round(avg_eng, 3),
                "risk_dist": risk_dist, "n_alerts": len(track_alerts),
                "timeline": timeline,
            }

    tracks_json = json.dumps(tracks_data, default=str)

    n_critical = sum(1 for a in alerts if a.get("severity") == "critical")
    n_warning  = sum(1 for a in alerts if a.get("severity") == "warning")
    n_tracks   = len(tracks_data)
    alerts_json = json.dumps(alerts, default=str)

    report_escaped = (report_text or "No clinical report generated").replace("`", "\\`").replace("${", "\\${")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixie Dashboard — {video_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0; min-height: 100vh; padding: 1.5rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            font-size: 2.2rem;
            background: linear-gradient(135deg, #9d4edd, #c77dff, #e0aaff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }}
        .subtitle {{ color: #b197fc; font-size: 1rem; margin-bottom: 1.5rem; }}
        .meta {{ color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }}

        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}

        .card {{
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 1rem; padding: 1.2rem; backdrop-filter: blur(8px);
            transition: transform 0.3s, border-color 0.3s;
        }}
        .card:hover {{ transform: translateY(-3px); border-color: #9d4edd; }}
        .card h2 {{ font-size: 1.2rem; color: #c77dff; margin-bottom: 0.8rem; }}
        .card h3 {{ font-size: 1.1rem; color: #e0aaff; margin-bottom: 0.6rem; }}

        .kpi-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
        .kpi {{
            flex: 1; min-width: 180px; padding: 1rem; border-radius: 0.8rem;
            text-align: center; background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .kpi .value {{ font-size: 2rem; font-weight: 700; }}
        .kpi .label {{ font-size: 0.8rem; color: #999; margin-top: 0.3rem; }}
        .kpi.critical .value {{ color: #f87171; }}
        .kpi.warning .value {{ color: #facc15; }}
        .kpi.ok .value {{ color: #4ade80; }}

        .engagement-bar-container {{
            background: rgba(255,255,255,0.08); border-radius: 0.5rem;
            height: 28px; overflow: hidden; margin-bottom: 0.5rem;
        }}
        .engagement-bar {{
            height: 100%; border-radius: 0.5rem; display: flex;
            align-items: center; justify-content: center;
            font-size: 0.85rem; font-weight: 600; color: #000;
            transition: width 0.8s ease;
        }}

        .risk-pills {{ display: flex; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap; }}
        .pill {{
            padding: 0.2rem 0.6rem; border-radius: 0.3rem; font-size: 0.78rem;
        }}
        .pill-low {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
        .pill-medium {{ background: rgba(250,204,21,0.2); color: #facc15; }}
        .pill-high {{ background: rgba(248,113,113,0.2); color: #f87171; }}

        .chart-container {{ height: 200px; margin-top: 0.8rem; }}

        .section {{
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem;
            backdrop-filter: blur(8px);
        }}
        .section h2 {{ font-size: 1.3rem; color: #c77dff; margin-bottom: 1rem; }}

        .report-content {{ line-height: 1.8; font-size: 0.92rem; white-space: pre-wrap; }}

        .tab-bar {{ display: flex; gap: 0; margin-bottom: 1rem; border-bottom: 2px solid rgba(255,255,255,0.1); }}
        .tab {{
            padding: 0.6rem 1.2rem; cursor: pointer; font-weight: 600;
            color: #888; border-bottom: 2px solid transparent; margin-bottom: -2px;
            transition: all 0.3s;
        }}
        .tab:hover {{ color: #c77dff; }}
        .tab.active {{ color: #e0aaff; border-bottom-color: #9d4edd; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        .alert-item {{
            padding: 0.6rem 0.8rem; border-radius: 0.5rem; margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }}
        .alert-critical {{ background: rgba(248,113,113,0.15); border-left: 3px solid #f87171; }}
        .alert-warning {{ background: rgba(250,204,21,0.15); border-left: 3px solid #facc15; }}

        .footer {{
            text-align: center; color: #555; font-size: 0.75rem;
            margin-top: 2rem; padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.05);
        }}

        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .card, .section, .kpi {{ animation: fadeIn 0.5s ease forwards; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>✨ Pixie Dashboard</h1>
        <p class="subtitle">Multimodal Behavioral Analysis — Engagement & Risk Assessment</p>
        <p class="meta">Video: {video_name} &nbsp;|&nbsp; Generated: {timestamp} &nbsp;|&nbsp; Pipeline: LangGraph + BiLSTM + Groq LLM</p>

        <!-- KPIs -->
        <div class="kpi-row">
            <div class="kpi {'critical' if n_critical > 0 else 'ok'}">
                <div class="value">🔴 {n_critical}</div>
                <div class="label">Critical Alerts</div>
            </div>
            <div class="kpi {'warning' if n_warning > 0 else 'ok'}">
                <div class="value">🟡 {n_warning}</div>
                <div class="label">Warnings</div>
            </div>
            <div class="kpi ok">
                <div class="value">👥 {n_tracks}</div>
                <div class="label">Tracks Analyzed</div>
            </div>
            <div class="kpi ok">
                <div class="value">📊 {len(predictions_df) if not predictions_df.empty else 0}</div>
                <div class="label">LSTM Predictions</div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="tab-bar">
            <div class="tab active" onclick="switchTab('students')">📊 Students</div>
            <div class="tab" onclick="switchTab('timeline')">📈 Timeline</div>
            <div class="tab" onclick="switchTab('alerts')">🚨 Alerts</div>
            <div class="tab" onclick="switchTab('report')">🧠 Report</div>
        </div>

        <!-- Tab: Students -->
        <div id="tab-students" class="tab-content active">
            <div class="grid" id="students-grid"></div>
        </div>

        <!-- Tab: Timeline -->
        <div id="tab-timeline" class="tab-content">
            <div class="section">
                <h2>📈 Engagement Timeline (All Tracks)</h2>
                <div style="height: 350px;">
                    <canvas id="globalTimeline"></canvas>
                </div>
            </div>
        </div>

        <!-- Tab: Alerts -->
        <div id="tab-alerts" class="tab-content">
            <div class="section">
                <h2>🚨 Alert Log</h2>
                <div id="alerts-container"></div>
            </div>
        </div>

        <!-- Tab: Report -->
        <div id="tab-report" class="tab-content">
            <div class="section">
                <h2>🧠 Clinical Observation Report (Dr. NeuroSight)</h2>
                <div class="report-content" id="report-content"></div>
            </div>
        </div>

        <div class="footer">
            Pixie Behavioral Analysis System — Powered by LangGraph, BiLSTM, OpenFace & Groq
        </div>
    </div>

    <script>
        const tracksData = {tracks_json};
        const alertsData = {alerts_json};
        const reportText = `{report_escaped}`;

        // ── Tab switching ────────────────────────────────────────
        function switchTab(name) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + name).classList.add('active');
            event.target.classList.add('active');
        }}

        // ── Render student cards ─────────────────────────────────
        function renderStudents() {{
            const grid = document.getElementById('students-grid');
            grid.innerHTML = '';
            for (const [tid, data] of Object.entries(tracksData)) {{
                const pct = Math.round(data.avg_engagement * 100);
                const color = data.avg_engagement > 0.6 ? '#4ade80' :
                              data.avg_engagement > 0.35 ? '#facc15' : '#f87171';
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <h3>${{data.name}}</h3>
                    <div class="engagement-bar-container">
                        <div class="engagement-bar" style="width: ${{pct}}%; background: ${{color}};">
                            ${{data.avg_engagement.toFixed(2)}}
                        </div>
                    </div>
                    <div class="risk-pills">
                        <span class="pill pill-low">🟢 Low: ${{data.risk_dist.low || 0}}</span>
                        <span class="pill pill-medium">🟡 Med: ${{data.risk_dist.medium || 0}}</span>
                        <span class="pill pill-high">🔴 High: ${{data.risk_dist.high || 0}}</span>
                    </div>
                    <div class="chart-container">
                        <canvas id="chart-${{tid}}"></canvas>
                    </div>
                `;
                grid.appendChild(card);

                // Mini timeline chart
                if (data.timeline && data.timeline.length > 0) {{
                    const ctx = document.getElementById('chart-' + tid).getContext('2d');
                    new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: data.timeline.map(p => p.frame),
                            datasets: [{{
                                data: data.timeline.map(p => p.score),
                                borderColor: color, borderWidth: 1.5,
                                fill: true, backgroundColor: color + '20',
                                pointRadius: 0, tension: 0.3,
                            }}],
                        }},
                        options: {{
                            responsive: true, maintainAspectRatio: false,
                            plugins: {{ legend: {{ display: false }} }},
                            scales: {{
                                x: {{ display: false }},
                                y: {{ min: 0, max: 1, ticks: {{ color: '#666', font: {{ size: 10 }} }} }},
                            }},
                        }},
                    }});
                }}
            }}
        }}

        // ── Render global timeline ───────────────────────────────
        function renderGlobalTimeline() {{
            const ctx = document.getElementById('globalTimeline').getContext('2d');
            const datasets = [];
            const colors = ['#9d4edd', '#4ade80', '#facc15', '#f87171', '#38bdf8', '#fb923c'];
            let i = 0;
            for (const [tid, data] of Object.entries(tracksData)) {{
                if (data.timeline && data.timeline.length > 0) {{
                    datasets.push({{
                        label: data.name,
                        data: data.timeline.map(p => ({{ x: p.frame, y: p.score }})),
                        borderColor: colors[i % colors.length],
                        borderWidth: 2, pointRadius: 0, tension: 0.3,
                        fill: false,
                    }});
                    i++;
                }}
            }}
            new Chart(ctx, {{
                type: 'line',
                data: {{ datasets }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }} }},
                    scales: {{
                        x: {{ type: 'linear', title: {{ display: true, text: 'Frame', color: '#999' }}, ticks: {{ color: '#666' }} }},
                        y: {{ min: 0, max: 1, title: {{ display: true, text: 'Engagement', color: '#999' }}, ticks: {{ color: '#666' }} }},
                    }},
                }},
            }});
        }}

        // ── Render alerts ────────────────────────────────────────
        function renderAlerts() {{
            const container = document.getElementById('alerts-container');
            if (alertsData.length === 0) {{
                container.innerHTML = '<p style="color:#4ade80;">✅ No alerts — all students within normal engagement range</p>';
                return;
            }}
            container.innerHTML = alertsData.map(a => {{
                const cls = a.severity === 'critical' ? 'alert-critical' : 'alert-warning';
                const icon = a.severity === 'critical' ? '🔴' : '🟡';
                const name = tracksData[a.track_id]?.name || 'Track ' + a.track_id;
                return `<div class="alert-item ${{cls}}">${{icon}} [${{a.severity.toUpperCase()}}] ${{name}}: ${{a.type.replace(/_/g, ' ')}} (engagement=${{a.avg_engagement || 'N/A'}})</div>`;
            }}).join('');
        }}

        // ── Render report ────────────────────────────────────────
        function renderReport() {{
            document.getElementById('report-content').innerText = reportText;
        }}

        // ── Init ─────────────────────────────────────────────────
        renderStudents();
        renderGlobalTimeline();
        renderAlerts();
        renderReport();
    </script>
</body>
</html>"""


def _generate_summary_json(
    predictions_df: pd.DataFrame,
    alerts: list[dict],
    identity_map: dict,
) -> dict:
    """Generate a machine-readable summary of the session."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_tracks": 0, "n_predictions": 0,
        "n_alerts": len(alerts), "alerts": alerts, "tracks": {},
    }
    if predictions_df.empty:
        return summary
    summary["n_tracks"] = int(predictions_df["track_id"].nunique())
    summary["n_predictions"] = len(predictions_df)
    for tid in predictions_df["track_id"].unique():
        t_df = predictions_df[predictions_df["track_id"] == tid]
        name = identity_map.get(int(tid), identity_map.get(str(tid), f"Student_{tid}"))
        summary["tracks"][str(tid)] = {
            "name": name,
            "avg_engagement": round(float(t_df["engagement_score"].mean()), 4) if "engagement_score" in t_df.columns else None,
            "risk_distribution": t_df["risk_level"].value_counts().to_dict() if "risk_level" in t_df.columns else {},
            "n_windows": len(t_df),
        }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# FLASK DASHBOARD SERVER
# ══════════════════════════════════════════════════════════════════════════════

def _start_flask_server(
    work_dir: str,
    html_path: str,
    json_path: str,
    lstm_csv: str,
    report_text: str,
    port: int = 5050,
) -> None:
    """Start a Flask API server in a background thread."""
    try:
        from flask import Flask, jsonify, send_file, Response
    except ImportError:
        print("  [Output] Flask not installed — skipping dashboard server. pip install flask")
        return

    app = Flask(__name__)

    @app.route("/")
    def dashboard():
        if os.path.isfile(html_path):
            return send_file(html_path)
        return "<h1>Dashboard not generated yet</h1>", 404

    @app.route("/api/session")
    def api_session():
        if os.path.isfile(json_path):
            with open(json_path) as f:
                return jsonify(json.load(f))
        return jsonify({"error": "Session summary not found"}), 404

    @app.route("/api/predictions")
    def api_predictions():
        if os.path.isfile(lstm_csv):
            df = pd.read_csv(lstm_csv)
            return jsonify(df.to_dict(orient="records"))
        return jsonify([])

    @app.route("/api/report")
    def api_report():
        return jsonify({"report": report_text})

    @app.route("/api/tracks")
    def api_tracks():
        if os.path.isfile(json_path):
            with open(json_path) as f:
                data = json.load(f)
            return jsonify(data.get("tracks", {}))
        return jsonify({})

    # Run in a daemon thread so it doesn't block the pipeline
    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    print(f"  [Output] 🌐 Flask dashboard running at http://localhost:{port}")


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE
# ══════════════════════════════════════════════════════════════════════════════

def run_output_node(state: dict) -> dict:
    """
    LangGraph node: Generate final output — interactive dashboard, alerts, JSON.
    """
    work_dir     = state.get("work_dir", ".")
    video_path   = state.get("video_path", "")
    report_text  = state.get("report_text", "")
    identity_map = state.get("identity_map", {})

    print(f"\n{'='*60}")
    print(f"[Node: Output] Generating dashboard & alerts")
    print(f"{'='*60}")

    # Load LSTM predictions
    lstm_csv = state.get("lstm_predictions_csv", os.path.join(work_dir, "lstm_predictions.csv"))
    predictions_df = _load_csv_safe(lstm_csv)
    print(f"  [Output] LSTM predictions: {len(predictions_df)} rows")

    # Detect alerts
    alerts = _detect_alerts(predictions_df)

    if alerts:
        print(f"\n  🚨 {'='*50}")
        print(f"  🚨 {len(alerts)} ALERT(S) DETECTED")
        print(f"  🚨 {'='*50}")
        for a in alerts:
            icon = "🔴" if a.get("severity") == "critical" else "🟡"
            name = identity_map.get(a.get("track_id", -1), f"Track {a.get('track_id', '?')}")
            print(f"  {icon} [{a['severity'].upper()}] {name}: "
                  f"{a['type'].replace('_', ' ')} "
                  f"(engagement={a.get('avg_engagement', 'N/A')})")
    else:
        print("  ✅ No alerts — all students within normal engagement range")

    # Generate dashboard HTML
    video_name = Path(video_path).stem if video_path else "session"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_content = _generate_dashboard_html(
        predictions_df=predictions_df,
        alerts=alerts,
        report_text=report_text,
        identity_map=identity_map,
        video_path=video_path,
    )

    html_path = os.path.join(work_dir, f"session_report_{video_name}_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Generate summary JSON
    summary = _generate_summary_json(predictions_df, alerts, identity_map)
    json_path = os.path.join(work_dir, f"session_summary_{video_name}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Start Flask server
    _start_flask_server(
        work_dir=work_dir,
        html_path=html_path,
        json_path=json_path,
        lstm_csv=lstm_csv,
        report_text=report_text,
    )

    print(f"\n[Node: Output] ✅ Done")
    print(f"  📄 Dashboard    : {html_path}")
    print(f"  📊 Summary JSON : {json_path}")
    print(f"  🚨 Alerts       : {len(alerts)}")
    print(f"  🌐 Server       : http://localhost:5050")

    return {
        "session_report_html":  html_path,
        "session_summary_json": json_path,
        "n_alerts":             len(alerts),
        "output_done":          True,
        "error":                None,
    }

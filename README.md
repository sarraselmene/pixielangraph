#  PIXIE — Classroom Behavioral Analysis via LangGraph

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-8B5CF6?style=flat-square" />
  <img src="https://img.shields.io/badge/YOLO-Detection-FF6B35?style=flat-square" />
  <img src="https://img.shields.io/badge/BiLSTM-Inference-16A34A?style=flat-square" />
  <img src="https://img.shields.io/badge/LLaMA-via%20Groq-F59E0B?style=flat-square" />
</p>

> **PIXIE** is an AI-powered system that analyzes student behavioral states in classroom videos using computer vision, deep learning, and a **LangGraph-orchestrated multi-node pipeline**. It produces per-student behavioral labels and generates clinical PDF/CSV reports via an LLM.

---

##  System Architecture

The full pipeline runs across **5 phases**, each mapped to a set of LangGraph nodes:

```
Phase 0 — Input
    └── Video input → creates working directories

Phase 1 — Extraction
    ├── Pose estimation     (BoTSORT + 17 keypoints)
    ├── Face detection      (per-frame crop)
    ├── Identity resolver   (InsightFace + voting)
    ├── Head orientation    (6DRepNet → Pitch/Yaw/Roll)
    └── Facial analysis     (OpenFace → AU + Gaze)

Phase 2 — Sync & Preprocessing
    ├── ID synchroniser     (injects persistent_id)
    ├── Head preproc        (Conf gate + Savitzky-Golay)
    ├── Body preproc        (Vis gate + SG)
    ├── AU preproc          (Median + SG)
    └── Gaze preproc        (Median + SG)

Phase 3 — Labelling
    ├── Head pose label     (Up / Down / Left / Right)
    ├── Gaze label          (H / V + stability)
    ├── AU label            (Smiles, fatigue)
    └── Body label          (Sit / Stand / Slouch)

Phase 4 — Inference
    ├── Merge node          (join on frame_id × pid)
    ├── Preprocessor        (Scaler + OneHot)
    └── Sequence model      (Sliding-window BiLSTM)

Phase 5 — Output
    ├── Database            (predictions storage)
    ├── Dashboard           (live monitoring canvas)
    └── Report              (LLaMA via Groq → PDF/CSV)
```

---

##  LangGraph Pipeline — Node Map

PIXIE's entire inference flow is orchestrated by **LangGraph**, a stateful graph framework for building multi-agent and multi-step AI pipelines.

### Shared State

All nodes communicate through a single typed state object:

```python
# graph_nodes/pixie_state.py
class PixieState(TypedDict):
    video_path: str
    persistent_ids: dict
    raw_features: dict
    preprocessed: dict
    labels: dict
    predictions: list
    report: str
```

### Node Descriptions

| Node | File | Role |
|------|------|------|
| `extract_node` | `extract_node.py` | Runs YOLO + BoTSORT, extracts pose keypoints and crops faces per frame |
| `face_recognition_node` | `face_recognition_node.py` | Identifies students using InsightFace with majority-vote assignment |
| `body_node` | `body_node.py` | Computes body posture labels (sit / stand / slouch) from keypoints |
| `gaze_node` | `gaze_node.py` | Estimates horizontal/vertical gaze direction and stability |
| `head_pose_node` | `head_pose_node.py` | Extracts 6DoF head orientation from 6DRepNet |
| `action_units_node` | `action_units_node.py` | Computes OpenFace AUs (fatigue, smile detection) |
| `inference_node` | `inference_node.py` | Runs sliding-window BiLSTM over merged feature vectors |
| `merge_node` | `merge_node.py` | Joins all label streams on `(frame_id, persistent_id)` |
| `buffer_node` | `buffer_node.py` | Maintains sliding feature buffer for real-time sequence building |
| `llm_analysis_node` | `llm_analysis_node.py` | Calls LLaMA 3 via Groq API to generate clinical behavioral report |
| `dashboard_node` | `dashboard_node.py` | Renders live per-student behavioral metrics on canvas dashboard |

### Graph Construction

```python
# main_graph.py
from langgraph.graph import StateGraph
from graph_nodes.pixie_state import PixieState

graph = StateGraph(PixieState)

graph.add_node("extract",          extract_node)
graph.add_node("face_recognition", face_recognition_node)
graph.add_node("body",             body_node)
graph.add_node("gaze",             gaze_node)
graph.add_node("head_pose",        head_pose_node)
graph.add_node("action_units",     action_units_node)
graph.add_node("merge",            merge_node)
graph.add_node("buffer",           buffer_node)
graph.add_node("inference",        inference_node)
graph.add_node("llm_analysis",     llm_analysis_node)
graph.add_node("dashboard",        dashboard_node)

graph.set_entry_point("extract")
graph.add_edge("extract",          "face_recognition")
graph.add_edge("face_recognition", "body")
graph.add_edge("face_recognition", "gaze")
graph.add_edge("face_recognition", "head_pose")
graph.add_edge("face_recognition", "action_units")
graph.add_edge("body",             "merge")
graph.add_edge("gaze",             "merge")
graph.add_edge("head_pose",        "merge")
graph.add_edge("action_units",     "merge")
graph.add_edge("merge",            "buffer")
graph.add_edge("buffer",           "inference")
graph.add_edge("inference",        "llm_analysis")
graph.add_edge("inference",        "dashboard")
graph.set_finish_point("llm_analysis")

app = graph.compile()
```

---

## Repository Structure

```
pixielangraph/
│
├── graph_nodes/
│   ├── __init__.py
│   ├── pixie_state.py              # Shared TypedDict state
│   ├── extract_node.py             # Phase 1 — Pose + detection
│   ├── face_recognition_node.py    # Phase 1 — Identity
│   ├── body_node.py                # Phase 3 — Body labelling
│   ├── gaze_node.py                # Phase 3 — Gaze labelling
│   ├── head_pose_node.py           # Phase 3 — Head orientation
│   ├── action_units_node.py        # Phase 3 — AU labelling
│   ├── merge_node.py               # Phase 4 — Feature join
│   ├── buffer_node.py              # Phase 4 — Sliding buffer
│   ├── inference_node.py           # Phase 4 — BiLSTM inference
│   ├── llm_analysis_node.py        # Phase 5 — LLM report
│   └── dashboard_node.py           # Phase 5 — Live dashboard
│
├── main_graph.py                   # LangGraph compilation & entry point
├── config.py                       # Thresholds & configuration
├── body_gestures_labeling.py       # Body gesture rules
├── label_head_pose.py              # Head pose labeling logic
├── label_gaze.py                   # Gaze labeling logic
├── label_action_units.py           # AU labeling logic
├── extract_raw_data_multi.py       # Multi-student raw extraction
├── anchor2.py                      # Anchor student detection
├── requirements.txt
└── .gitignore
```

---

##  Installation

```bash
git clone https://github.com/sarraselmene/pixielangraph.git
cd pixielangraph
pip install -r requirements.txt
```

> For macOS OpenFace setup, run: `bash install_openface_macos.sh`

---

##  Usage

```bash
python main_graph.py --video path/to/classroom_video.mp4
```

The pipeline will:
1. Extract poses and faces from each frame
2. Identify students using InsightFace
3. Label behavioral signals (body, gaze, head, AU)
4. Run BiLSTM inference on sliding windows
5. Generate a clinical report via LLaMA 3 (Groq)
6. Display live metrics on the dashboard

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | **LangGraph** |
| Detection & Tracking | **YOLOv8 + BoTSORT** |
| Face Recognition | **InsightFace** |
| Facial Analysis | **OpenFace** (AU + Gaze) |
| Head Pose | **6DRepNet** |
| Sequence Model | **BiLSTM (PyTorch)** |
| LLM Report | **LLaMA 3 via Groq API** |
| Language | **Python 3.10+** |

---

##  Author

**Sarra Selmene** — Engineering Student, ENSI Tunisia  
Final Year Project · 2026

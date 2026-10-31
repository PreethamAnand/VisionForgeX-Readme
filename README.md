# VisionForgeX

> **Autonomous Closed-Loop Industrial Quality Assurance & Root-Cause Intelligence**  
> Powered by Self-Supervised Foundation Vision (DINOv2), Multimodal Telemetry Fusion, and Qualcomm Cloud AI 100 Ultra / Qualcomm AI Hub.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Qualcomm Cloud AI](https://img.shields.io/badge/Qualcomm-Cloud%20AI%20100%20Ultra-0066CC.svg)](https://aisuite.cirrascale.com/)
[![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm-AI%20Hub%20(qai--hub)-purple.svg)](https://aihub.qualcomm.com/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Streamlit%20%7C%20NetworkX-FF4B4B.svg)]()
[![Tests](https://img.shields.io/badge/Tests-105%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Key Capabilities & Innovations](#-key-capabilities--innovations)
3. [The 5-Stage Closed-Loop Pipeline](#-the-5-stage-closed-loop-pipeline)
4. [System Architecture](#-system-architecture)
5. [Qualcomm AI Ecosystem Integration](#-qualcomm-ai-ecosystem-integration)
6. [Interactive Web Dashboard Guide](#-interactive-web-dashboard-guide)
7. [Quick Start & Installation](#-quick-start--installation)
8. [CLI Execution & Scenarios](#-cli-execution--scenarios)
9. [Automated Test Suite](#-automated-test-suite)
10. [Repository Structure](#-repository-structure)
11. [Hardware & Software Specifications](#-hardware--software-specifications)
12. [Full Documentation Reference](#-full-documentation-reference)

---

## 🏭 Executive Summary

Modern high-precision CNC manufacturing facilities (producing automotive, aerospace, and precision tooling components) operate under extreme quality tolerances. However, conventional quality control suffers from severe **data silos**:

1. **Optical Inspection is Passive & Delayed**: Vision cameras inspect components *after* machining operations are finished. By the time a crack, scratch, or surface deformation is flagged, the part is already scrap.
2. **Telemetry Data is Disconnected from Defects**: Machine sensors (accelerometers, thermocouples, current transducers) detect load surges and vibration chatter without contextual understanding of whether a physical defect occurred.
3. **Cascading Batch Failures**: Unnoticed tool wear, spindle bearing play, or thermal drift cascade across an entire production batch of 200–500 units before manual detection, resulting in **8% to 15% scrap rates** and costing manufacturers **\$50,000 to \$250,000 per incident**.

**VisionForgeX** bridges this gap by unifying foundation computer vision, machine sensor analytics, predictive failure recurrence modeling, and cloud/edge Qualcomm AI reasoning into a single **closed-loop industrial intelligence workstation**.

---

## 🌟 Key Capabilities & Innovations

- **Self-Supervised Visual Metrology (DINOv2 ViT-S/14)**: Extracts rich 384-dimensional spatial patch embeddings without requiring large task-specific labeled defect datasets.
- **PatchCore Anomaly Segmentation**: Sub-samples normal patch distributions into a coreset memory bank for sub-pixel surface defect segmentation and precise bounding box localization.
- **Open-Set Anomaly Awareness**: Recognizes novel, unseen defect geometries (such as inverted/reversed parts in Scenario S4) as open-set anomalies instead of misclassifying them.
- **Sub-Second Multimodal Temporal Fusion**: Synchronizes optical defect timestamps with CNC machining intervals within an exact $[-10\text{ s}, +2\text{ s}]$ engagement window.
- **Causal Evidence Graph**: Employs NetworkX to construct directed causal graphs linking machine physical distress (e.g., $+3.8\sigma$ vibration spike) to surface flaws.
- **Calibrated Failure Recurrence Modeling**: Evaluates batch failure recurrence probability using a machine learning model trained on 2,100 production workpieces across 42 historical batches.
- **Qualcomm Cloud AI 100 Ultra Reasoning**: Executes natural-language root-cause diagnosis via Cirrascale's Qualcomm Cloud AI API (`Llama-3.1-8B`) in **under 2.5 seconds** per inspection.
- **Dual-Tier Offline Resilience**: Built-in deterministic industrial diagnostic safety fallback engine ensuring 100% uptime with zero factory crashes during network interruptions.
- **Certified ISO Audit Reports**: Generates formal, audit-ready PDF and Markdown inspection certificates with human-in-the-loop engineer digital sign-off.

---

## 🔄 The 5-Stage Closed-Loop Pipeline

VisionForgeX follows the **SEE → UNDERSTAND → CORRELATE → PREDICT → ACT** paradigm:

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌──────────────┐     ┌─────────────┐
│   1. SEE    │ ──▶ │  2. UNDERSTAND   │ ──▶ │  3. CORRELATE  │ ──▶ │  4. PREDICT  │ ──▶ │   5. ACT    │
│  (Optical)  │     │   (Telemetry)    │     │    (Fusion)    │     │    (Risk)    │     │ (Qualcomm)  │
└─────────────┘     └──────────────────┘     └────────────────┘     └──────────────┘     └─────────────┘
  DINOv2 Patch       4-Channel CNC Sensor     [-10s, +2s] Window     Logistic Regress     Qualcomm AI 100
  Memory Bank        Z-Score Baseline         NetworkX Causal        2,100 Units DB       Root Cause & ISO
  Bounding Box       Anomaly Detection        Evidence Graph         Recurrence Dial      Audit PDF Report
```

| Stage | Name | Technical Implementation | Output / Artifact |
| :--- | :--- | :--- | :--- |
| **1** | **SEE** | PyTorch DINOv2 ViT-S/14 (384-dim patch features) + PatchCore k-NN memory bank + Few-shot prototypes | Anomaly heatmap, bounding box coordinates, anomaly score, prototype confidence |
| **2** | **UNDERSTAND** | 4-channel telemetry ingestion (10 Hz: Vibration RMS, Spindle Temp, Current, Coolant) + Past-only rolling baseline | Per-signal $z$-score deviation, ISO 10816 threshold alerts |
| **3** | **CORRELATE** | Temporal alignment window ($[-10\text{ s}, +2\text{ s}]$) correlating visual flaws with machining telemetry spikes | Synchronized multi-modal timeline & NetworkX directed evidence graph |
| **4** | **PREDICT** | Calibrated Logistic Regression trained on 2,100 historical workpieces in SQLite (`history.db`) | Batch failure recurrence probability & ranked machine risk drivers |
| **5** | **ACT & EXPLAIN** | Qualcomm Cloud AI 100 Ultra (`Llama-3.1-8B`) + Deterministic industrial rule fallback engine | Root-cause category, physical mechanism hypothesis, corrective actions, ISO PDF certificate |

---

## 🏗️ System Architecture

```
[ Edge Acquisition Layer ]
  ├── High-Resolution Optical Inspection Camera (MVTec AD Industrial Dataset)
  └── 4-Channel CNC Machine Sensors (Vibration RMS, Spindle Temp, Motor Load, Coolant Flow)
            │
            ▼
[ VisionForgeX Pipeline Core (orchestrator/pipeline.py) ]
  ├── Vision Pipeline (Part A)
  │     ├── Backbone: DINOv2 ViT-S/14 Patch Transformer (384-dim)
  │     ├── Coreset Memory: Non-parametric PatchCore k-NN bank
  │     └── Open-Set Detector: Density-based unknown anomaly flagging
  ├── Telemetry Pipeline (Part B)
  │     ├── Signal Ingestion: Rolling past-only statistics (mean, variance)
  │     ├── Anomaly Engine: Normalized z-score thresholding
  │     └── Causal Fusion: [-10s, +2s] machining temporal alignment
  ├── Risk & History Engine (Part B)
  │     ├── SQLite Production DB: 2,100 historical workpieces, 42 batches
  │     └── Recurrence Predictor: Calibrated Logistic Regression model
  └── Diagnostic Reasoning Engine (Part C)
        ├── Primary Provider: Qualcomm Cloud AI 100 Ultra (Cirrascale API)
        ├── Low-Latency Model: Llama-3.1-8B (~2.2s inference)
        └── Offline Fallback: Deterministic industrial rule engine
            │
            ▼
[ Industrial Interfaces & Output ]
  ├── Streamlit 8-Module Web Dashboard (http://localhost:8501)
  ├── Automated ISO-Compliant PDF/Markdown Audit Reports
  └── Qualcomm AI Hub Toolkit (Snapdragon X Elite NPU compilation & profiling)
```

---

## ⚡ Qualcomm AI Ecosystem Integration

VisionForgeX is natively architected for Qualcomm's cloud-to-edge acceleration ecosystem:

### 1. Qualcomm Cloud AI 100 Ultra (Cirrascale REST API)
- **Endpoint**: `https://aisuite.cirrascale.com/apis/v2`
- **Hosted Silicon**: Qualcomm Cloud AI 100 Ultra accelerator cards.
- **Model Selected**: `Llama-3.1-8B`
- **Performance**: High-throughput structured diagnostic generation in **~2.2 seconds** per workpiece.
- **Supported Alternative**: `DeepSeek-R1-Distill-Llama-70B` (supported for deep-chain reasoning).

### 2. Qualcomm AI Hub (`qai-hub` 0.56.0 SDK)
- **Target Silicon Profiled**: `Snapdragon X Elite CRD`, `Snapdragon 8 Gen 3`, and `Cloud AI 100 Ultra`.
- **Target Runtime**: `qnn_context_binary` (compiled for Hexagon NPU execution).
- **Capabilities**: Compiles vision models into quantized INT8/FP16 context binaries for ultra-low latency (<10ms) edge deployment directly beside factory CNC machines.

### 3. Industrial Offline Resilience
If factory Internet disconnects or cloud latency exceeds tolerance, VisionForgeX automatically activates its calibrated rule engine ([`reasoning/fallback.py`](file:///c:/Users/preet/Downloads/Inf%20Hack-AI-Thon/reasoning/fallback.py)) without dropping a single inspection cycle.

---

## 🖥️ Interactive Web Dashboard Guide

The web dashboard is powered by Streamlit and runs on port **8501**:

👉 **`http://localhost:8501`**

```
┌────────────────────────────────────────────────────────────────────────┐
│ VisionForgeX — Production Inspection & Diagnostic Station              │
├────────────────────────────────────────────────────────────────────────┤
│ Unit ID: U-DEMO-01  |  Machine: CNC-02  |  Mode: Live  |  Latency: 2s  │
├────────────────────────────────────────────────────────────────────────┤
│ [Tab 1: Visual]     [Tab 2: Telemetry]    [Tab 3: Evidence Graph]      │
│ [Tab 4: Reasoning]  [Tab 5: Recurrence]   [Tab 6: Engineer Review]     │
│ [Tab 7: Export PDF] [Tab 8: Qualcomm Hub] [Tab 9: System Evaluation]   │
└────────────────────────────────────────────────────────────────────────┘
```

### Module Descriptions
- **🔍 Tab 1: Visual Inspection**: Displays real optical photos of the workpiece, DINOv2 patch anomaly heatmaps, bounding box coordinates, and defect classification score.
- **📊 Tab 2: Sensor Telemetry**: Interactive time-series charts showing Vibration RMS, Spindle Bearing Temperature, Motor Current, and Coolant Flow synchronized to the cutting interval.
- **🕸️ Tab 3: Evidence Graph**: Interactive NetworkX causal graph mapping sensor spikes directly to optical defect nodes.
- **🧠 Tab 4: Root-Cause Reasoning**: Displays live diagnostics executed on Qualcomm Cloud AI 100 Ultra, including category (`tool_wear_chatter`), physical mechanism, and prioritized machine adjustments.
- **📈 Tab 5: Recurrence Risk**: Interactive radial gauge displaying batch failure recurrence probability (0% to 100%) and ranked contributing risk drivers.
- **✍️ Tab 6: Engineer Review**: Human-in-the-loop review station with engineer notes, digital badge ID sign-off, and immutable SQLite logging.
- **📄 Tab 7: Export Report**: One-click generation and download of formal ISO 9001/IATF 16949 inspection PDF audit reports.
- **⚡ Tab 8: Qualcomm Toolkits & NPU**: Device selection card for Snapdragon X Elite and Cloud AI 100 with model compilation and profiling triggers.
- **📊 Tab 9: System Evaluation**: Real-time latency benchmarks, confusion metrics, and memory bank health monitors.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: 3.10, 3.11, or 3.12
- **Memory**: 8 GB RAM minimum (16 GB recommended)

### 2. Clone Repository
```bash
git clone https://github.com/PreethamAnand/VisionForgeX.git
cd VisionForgeX
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Credentials
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

Open `.env` and verify your credentials:
```env
# Qualcomm Cloud AI 100 Ultra (Cirrascale REST API)
QAIC_API_KEY=your_cirrascale_api_key_here
QAIC_BASE_URL=https://aisuite.cirrascale.com/apis/v2
QAIC_LLM_MODEL=Llama-3.1-8B
QAIC_VLM_MODEL=Llama-3.1-8B

# Qualcomm AI Hub Toolkit (qai-hub)
QAI_HUB_API_TOKEN=your_cirrascale_api_key_here
QAI_HUB_DEVICE=Snapdragon X Elite CRD
QAI_HUB_RUNTIME=qnn_context_binary
```

### 5. Launch the Web Application
```bash
streamlit run dashboard/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 💻 CLI Execution & Scenarios

VisionForgeX includes a powerful command-line interface for batch processing, automated testing, and CI/CD pipelines:

### Built-in Factory Scenarios (S0 to S4)
- **Scenario S0**: Nominal good workpiece (normal vibration, baseline temperature).
- **Scenario S1**: Surface thread scratch (correlated with $+3.8\sigma$ vibration spike).
- **Scenario S2**: Surface discoloration (correlated with $+3.2\sigma$ spindle overheating).
- **Scenario S3**: Bent/deformed flange (correlated with transient cutting load current spike).
- **Scenario S4**: Open-set unseen defect (inverted/flipped geometry triggering open-set review).

### Run Commands

```powershell
# Run Scenario S1 with automated PDF audit report generation:
python run_pipeline.py -s S1 --report

# Run Scenario S0 (Nominal normal unit):
python run_pipeline.py -s S0 --report

# Run all scenarios sequentially (S0 through S4):
python run_pipeline.py -s all --report

# Inspect any custom image file:
python run_pipeline.py -i PART_A/mvtec_ad/metal_nut/test/scratch/001.png --report
```

---

## 🧪 Automated Test Suite

VisionForgeX is backed by an automated test suite with **105 passing tests** covering every component of the pipeline:

```powershell
pytest tests/ PART_A/tests/ PART_B/tests/ -v
```

### Test Coverage Summary
| Test Directory | Focus Area | Status |
| :--- | :--- | :--- |
| `tests/` | End-to-end integration, Qualcomm Cloud API, `qai-hub` toolkits, reports | **31 Passed, 2 Skipped** |
| `PART_A/tests/` | DINOv2 feature extraction, PatchCore memory bank, bounding box localization | **37 Passed (100%)** |
| `PART_B/tests/` | Telemetry ingestion, baseline z-scores, multimodal fusion, recurrence risk ML | **37 Passed (100%)** |
| **Total** | **Full System Verification** | **105 Passed** |

---

## 📁 Repository Structure

```
VisionForgeX/
├── .env.example                       # Template for environment configuration
├── .gitignore                         # Strict exclusion for secrets, logs & large datasets
├── README.md                          # Comprehensive project overview & quick start
├── DOCUMENTATION.md                   # Full 360-degree technical architectural handbook
├── requirements.txt                   # Production Python dependencies
├── conftest.py                        # Pytest global fixtures & headless Matplotlib configuration
├── run_pipeline.py                    # Unified CLI inspection runner
├── generate_presentation_pdf.py       # Script to compile 7-slide Hackathon presentation PDF
├── generate_master_guide_pdf.py       # Script to compile comprehensive Master Guide PDF
│
├── contracts/                         # Pydantic V2 Contract Schemas
│   ├── __init__.py
│   └── schemas.py                     # Typed schemas: VisionResult, SignalResult, FusedEvent, etc.
│
├── dashboard/                         # Streamlit Interactive Web Application
│   ├── __init__.py
│   ├── app.py                         # 8-Module Streamlit station (live at port 8501)
│   ├── charts.py                      # Matplotlib/Plotly telemetry & risk charts
│   └── report.py                      # Industrial PDF & Markdown report generation engine
│
├── orchestrator/                      # Pipeline Orchestration Core
│   ├── __init__.py
│   └── pipeline.py                    # Unified end-to-end inspection coordinator
│
├── qcloud/                            # Qualcomm Ecosystem Wrappers
│   ├── __init__.py
│   └── hub_toolkit.py                 # Qualcomm AI Hub (qai-hub) SDK & CLI integration
│
├── reasoning/                         # Diagnostic Reasoning Engine
│   ├── __init__.py
│   ├── engine.py                      # Reasoning engine coordinator
│   ├── fallback.py                    # Deterministic industrial diagnostic rule engine
│   ├── prompts.py                     # Structured industrial prompt templates
│   ├── providers.py                   # Qualcomm Cloud AI 100 Ultra REST provider
│   └── validator.py                   # Output validation, JSON & Key-Value parser
│
├── PART_A/                            # Computer Vision Subsystem (SEE)
│   ├── vision/                        # DINOv2 ViT-S/14, PatchCore memory bank, heatmaps
│   ├── mvtec_ad/metal_nut/            # Industrial workpiece image dataset
│   └── tests/                         # 37 Unit & integration tests for computer vision
│
├── PART_B/                            # Telemetry, Fusion & Risk Prediction (UNDERSTAND, CORRELATE, PREDICT)
│   ├── telemetry/                     # 4-Channel signal simulator, rolling baseline, detector
│   ├── fusion/                        # Temporal alignment [-10s, +2s] & NetworkX evidence graph
│   ├── prediction/                    # Calibrated Logistic Regression recurrence model
│   ├── data/history.db                # SQLite database (2,100 units across 42 batches)
│   ├── models/risk_logreg.joblib      # Trained serialized failure recurrence model
│   └── tests/                         # 37 Unit & integration tests for signals & fusion
│
├── runs/                              # Inspection Outputs & Reports
│   ├── report_S1.pdf                  # Generated ISO inspection PDF certificate
│   └── report_S1.md                   # Generated Markdown audit report
│
└── tests/                             # Unified Integration Test Suite
    ├── test_dashboard_and_reports.py
    ├── test_orchestrator.py
    ├── test_qualcomm_integration.py
    ├── test_qualcomm_toolkits.py
    ├── test_reasoning.py
    └── test_scenarios_integration.py
```

---

## ⚙️ Hardware & Software Specifications

| Parameter | Edge / Inspection Station Specification |
| :--- | :--- |
| **CPU Architecture** | x86_64 or ARM64 (Qualcomm Snapdragon X Elite) |
| **Edge Accelerator** | Qualcomm Hexagon NPU / Snapdragon X Elite CRD |
| **Cloud Accelerator** | Qualcomm Cloud AI 100 Ultra (Cirrascale Infrastructure) |
| **Vision Backbone** | DINOv2 ViT-S/14 ($224 \times 224$ input, 384-dim patch features) |
| **Telemetry Ingestion** | 4 channels at 10 Hz (Vibration RMS, Spindle Temp, Current, Flow) |
| **Database Engine** | SQLite 3 (WAL journal mode enabled) |
| **Web Server** | Streamlit 1.32+ running on Uvicorn |
| **Report Standards** | ISO 9001 / IATF 16949 compliant PDF format via FPDF2 |

---

## 📖 Full Documentation Reference

For the comprehensive mathematical formulation, component-by-component deep dive, database ER diagrams, API schema contracts, and factory operational runbooks, refer to:

👉 **[DOCUMENTATION.md](DOCUMENTATION.md)**

---

## 👥 Authors & Team VisionForge

- **Lead AI & Computer Vision Engineer**: Self-supervised DINOv2 patch representation, PatchCore memory bank, few-shot prototypes & open-set anomaly detection.
- **Edge & Systems Specialist**: 4-channel CNC telemetry ingestion, rolling baseline anomaly scoring, and Qualcomm AI Hub (`qai-hub`) Snapdragon NPU profiling.
- **Pipeline & Full-Stack Engineer**: Multimodal temporal fusion engine, SQLite production history, Streamlit 8-module dashboard, and certified ISO report engine.

---

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

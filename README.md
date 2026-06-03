# 🛡️ GraphShield AI

### Enterprise Heterogeneous GraphSAGE Transaction Risk & Explainability Platform

GraphShield AI is a production-grade, state-of-the-art financial crime detection platform designed for modern banking and fintech operations. By moving beyond legacy rules-based engines, GraphShield AI models transactions, accounts, customers, devices, and merchants as a unified heterogeneous network. It leverages a **Heterogeneous Graph Neural Network (GraphSAGE)** to detect structural fraud rings, layering chains, and account takeover patterns in real-time.

---

## 🧭 Navigating the System

The application is structured into **8 Dedicated Analyst Workspaces** accessible via the Streamlit frontend navigation sidebar:

1. **Executive Dashboard**: High-level metrics summaries, fraud volumes, recall metrics, alert trends, and GNN risk score distributions.
2. **Fraud Alerts**: SOC-style alerts triage queue displaying transaction amounts, GNN probabilities, case assignments, priority tiers, and interactive resolution forms.
3. **Investigation Center**: The investigator's "single pane of glass" compiling transaction lookups, risk scores, SHAP explanations, connected entities, forensic narratives, and downloadable case reports.
4. **Network Explorer**: Interconnected node visualization canvas supporting entity category selections, 1/2-hop neighborhood expansions, path tracing between accounts (NetworkX shortest path), and structural anomaly detection.
5. **Explainable AI**: Diagnostics console separating local SHAP feature contributions from global feature importance metrics.
6. **Model Performance**: MLOps analysis panel displaying ROC curves, confusion matrices, model drift charts, and performance analytics.
7. **Transaction Simulator**: Real-time payload injector sandbox displaying prediction outcomes, recommendations, and raw JSON logs instantly.
8. **System Administration**: Admin dashboard managing FastAPI connections, model retraining operations, and batch CSV uploader validation.

---

## 🏗️ Platform System Architecture

```text
[Core Transaction Database]
   └───> [Validation & Ingestion Pipeline]
           └───> [Behavioral Feature Engineering]
                   └───> [Heterogeneous Graph Construction (PyG)]
                           └───> [GraphSAGE GNN Layers]
                                   └───> [Edge Classification Network]
                                           ├───> [Explainable AI Layer (SHAP)]
                                           └───> [FastAPI REST API Services]
                                                   └───> [Streamlit Operations Dashboard]
```

---

## ⚡ Quick Start: Running Locally

### 1. Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend

Run the backend service on port `8000`:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The REST API Docs will be available at:

```text
http://127.0.0.1:8000/docs
```

### 3. Launch Streamlit Frontend

Run the analyst dashboard on port `8501`:

```bash
python -m streamlit run visualization/dashboard_app.py --server.port 8501
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## 🔌 API Reference Routes

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | `/` | Checks service connectivity and health status |
| GET | `/api/v1/dashboard/summary` | Fetches aggregate totals, alert rates, and recent alerts |
| POST | `/api/v1/prediction/predict` | Scores a manual transaction payload |
| POST | `/api/v1/explainability/explain` | Computes SHAP attributions and returns explanation artifacts |
| GET | `/api/v1/graph/neighbors/{type}/{id}` | Queries immediate graph connections |
| GET | `/api/v1/graph/stats` | Fetches graph statistics and density metrics |
| POST | `/api/v1/prediction/train` | Triggers model retraining processes |

### Sample Prediction Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/prediction/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sender_account_id": "ACC_1001250",
  "merchant_id": "M_1005",
  "amount": 8800.0,
  "device_id": "DEV_H_99999",
  "txns_last_1h": 7,
  "txns_last_24h": 18,
  "geo_anomaly": 1
}'
```

---

## 🧠 Core Technologies

### Machine Learning
- PyTorch
- PyTorch Geometric (PyG)
- GraphSAGE
- SHAP Explainability

### Backend
- FastAPI
- Uvicorn
- Pydantic

### Frontend
- Streamlit
- Plotly
- NetworkX

### Data Processing
- Pandas
- NumPy
- Scikit-Learn

---

## 🎨 Visual Design System (Obsidian Slate Theme)

The UI overrides Streamlit default styles utilizing `styles.css` and configuration files.

### Theme Colors

- **Background:** Obsidian (`#060913`)
- **Containers & Cards:** Slate (`#0F172A`)
- **Primary Accents:** Neon Cyan (`#00E5FF`) and Emerald Green (`#10B981`)
- **High Risk:** Crimson (`#FF1744`)
- **Warning:** Amber (`#F59E11`)
- **Low Risk:** Mint (`#34D399`)

### Typography

- **Outfit** — Display headings and metric cards
- **Inter** — Tables, forms, and interface text

---

## 📊 Key Capabilities

- Real-time Fraud Detection
- Graph-based Risk Analysis
- Fraud Ring Discovery
- Account Takeover Detection
- Explainable AI Workflows
- Network Relationship Analysis
- Behavioral Anomaly Detection
- Investigator Case Management
- Model Monitoring & Drift Analysis
- Transaction Simulation Environment

---

## 🔒 Security & Compliance

- Input validation pipelines
- Secure API request handling
- Audit-ready investigation workflows
- Explainable prediction outputs
- Data integrity checks
- Traceable transaction analysis

---

## 🚀 Future Enhancements

- Graph Transformer Networks
- Neo4j Graph Database Integration
- Kafka Real-Time Streaming
- LLM-Powered Investigation Assistant
- Automated Suspicious Activity Report Generation
- Multi-Tenant Enterprise Deployment
- Advanced Case Management System


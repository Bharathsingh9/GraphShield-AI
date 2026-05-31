# 🛡️ GraphShield AI

### Enterprise Heterogeneous GraphSAGE Transaction Risk & Explainability Platform

GraphShield AI is a production-grade, state-of-the-art financial crime detection platform designed for modern banking operations (styled in accordance with Lloyds Banking Group guidelines). By moving beyond legacy rules-based engines, GraphShield AI models transactions, accounts, customers, devices, and merchants as a unified heterogeneous network. It leverages a **Heterogeneous Graph Neural Network (GraphSAGE)** to detect structural fraud rings, layering chains, and account takeover patterns in real-time.

---

## 🧭 Navigating the System

The application is structured into **8 Dedicated Analyst Workspaces** accessible via the Streamlit frontend navigation sidebar:

1. **Executive Dashboard**: High-level metrics summaries, fraud volumes, recall constants, and GNN risk score population distributions.
2. **Fraud Alerts**: SOC-style alerts triage queue displaying transaction amounts, GNN probabilities, case assignments, priority tiers, and interactive resolution forms.
3. **Investigation Center**: The investigator's "single pane of glass" compiling transaction lookups, risk scores, local SHAP waterfalls, 1-hop neighbor listings, forensic narratives, and regulatory download files.
4. **Network Explorer**: Interconnected node visualization canvas supporting entity category selections, 1/2-hop neighborhood expansions, path tracing between accounts (NetworkX shortest path), and structural anomaly warnings.
5. **Explainable AI**: Diagnostics console separating local SHAP marginal perturbations from global feature importance metrics.
6. **Model Performance**: MLOps analysis panel displaying ROC curves, confusion heatmaps, data drift charts, and a math integrity check verify calculations.
7. **Transaction Simulator**: Real-time payload injector sandbox displaying prediction outcomes, recommendations, and raw JSON logs instantly.
8. **System Administration**: Admin dashboard managing FastAPI connections, PyTorch optimizer retraining runs, and batch CSV uploader validators.

---

## 🏗️ Platform System Architecture

```text
[Core Ledger DB] 
   └───> [Validation Ingestion Pipes]
           └───> [Behavioral Feature Engineering]
                   └───> [Heterogeneous Graph Construction (PyG)]
                           └───> [GraphSAGE GNN Convolutions]
                                   └───> [Edge Classifier MLP Head]
                                           ├───> [Explainable AI Layer (SHAP)]
                                           └───> [FastAPI REST API Services]
                                                   └───> [Streamlit Ops Dashboard]
```

---

## ⚡ Quick Start: Running Locally

### 1. Installation
Clone the registry and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Daemon
Run the backend uvicorn service on port `8000`:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*The REST API Docs will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

### 3. Launch Streamlit Frontend
Run the analyst operations center on port `8501`:
```bash
python -m streamlit run visualization/dashboard_app.py --server.port 8501
```
*The Analyst Dashboard will be available at [http://localhost:8501](http://localhost:8501)*

---

## 🔌 API Reference Routes

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/` | Checks service connectivity and health status. |
| **GET** | `/api/v1/dashboard/summary` | Fetches aggregate totals, alert rates, and recent alerts. |
| **POST** | `/api/v1/prediction/predict` | Scores a manual transaction payload. |
| **POST** | `/api/v1/explainability/explain` | Computes local SHAP attributions and returns waterfall PNG paths. |
| **GET** | `/api/v1/graph/neighbors/{type}/{id}` | Queries immediate connections in the graph database. |
| **GET** | `/api/v1/graph/stats` | Fetches global node distribution counts and density. |
| **POST** | `/api/v1/prediction/train` | Spawns background PyTorch retraining processes. |

### Sample Score Request cURL
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

## 🎨 Visual Design System (Obsidian Slate Theme)

The UI overrides Streamlit default styles utilizing [styles.css](file:///d:/fraud_detection/visualization/styles.css) and configuration files:
* **Background**: Obsidian (`#060913`)
* **Containers & Cards**: Slate (`#0F172A`)
* **Primary Accents**: Neon Cyan (`#00E5FF`) and Emerald Green (`#005F43` / `#10B981`)
* **Status Flags**: High Risk Crimson (`#FF1744`) | Warning Amber (`#F59E11`) | Clear Mint (`#34D399`)
* **Fonts**: `Outfit` (for display headings/numeric values) and `Inter` (for text inputs and tables).

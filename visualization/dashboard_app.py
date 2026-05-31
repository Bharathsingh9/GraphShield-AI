import streamlit as st
import requests
from visualization.utils import apply_lbg_theme, render_section_header

# Global config
st.set_page_config(
    page_title="GraphShield AI - Fraud Ops Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_lbg_theme()

st.markdown('# <span class="gradient-text">🛡️ GraphShield AI</span>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.25rem; font-weight: 500; color: #94A3B8; margin-top: -0.5rem; margin-bottom: 2rem;">Enterprise Fraud Operations Center</div>', unsafe_allow_html=True)

# Create layout columns
col_main, col_status = st.columns([7, 3])

with col_main:
    with st.container(border=True):
        render_section_header("Welcome to GraphShield AI", "Unified GraphSAGE-based financial crime detection", "📖")
        st.markdown("""
        GraphShield AI represents the next generation of transaction monitoring, moving beyond legacy rules-based engines.
        By modeling bank accounts, customers, devices, and merchants as a single heterogeneous relational graph, 
        our GraphSAGE GNN model flags structural fraud rings, mule layering networks, and credentials sharing anomalies in real-time.
        
        ### 🧭 Navigating the Workspaces
        Please use the sidebar menu to access your designated system workspace:
        
        1. **Executive Dashboard**: High-level operational summaries, trends, and risk distribution charts.
        2. **Fraud Alerts**: Security Operations Center (SOC) style alert triage queue with assignment controls.
        3. **Investigation Center**: Single-pane-of-glass workspace compile-tracking transaction details, SHAP waterfalls, and narratives.
        4. **Network Explorer**: Interactive relationship mapper tracing shortest paths between entity accounts.
        5. **Explainable AI**: Local/global SHAP attributions explaining model prediction drivers.
        6. **Model Performance**: ROC curves, confusion matrices, and metrics mathematical verification checks.
        7. **Transaction Simulator**: Real-time payload injector sandbox displaying scores and raw JSON logs instantly.
        8. **System Administration**: PyTorch optimizer triggers and drag-and-drop CSV validation checks.
        """)

with col_status:
    with st.container(border=True):
        render_section_header("Services Status", "Real-time API connection check", "🔌")
        try:
            res = requests.get("http://127.0.0.1:8000/", timeout=2.5)
            if res.status_code == 200:
                st.success("🟢 **FASTAPI BACKEND**: Connected & Healthy")
            else:
                st.warning(f"🟡 **FASTAPI BACKEND**: Status {res.status_code}")
        except Exception:
            st.error("🔴 **FASTAPI BACKEND**: Offline / Disconnected\n\nEnsure the FastAPI Uvicorn service is active on port 8000.")

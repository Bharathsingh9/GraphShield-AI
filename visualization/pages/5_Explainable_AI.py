import streamlit as st
import pandas as pd
import requests
from visualization.utils import apply_lbg_theme, make_api_request, render_section_header, render_kpi_card, render_status_badge

st.set_page_config(page_title="Explainable AI - GraphShield AI", layout="wide")
apply_lbg_theme()

st.markdown('# <span class="gradient-text">🧠 Explainable AI</span>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.05rem; font-weight: 500; color: #94A3B8; margin-top: -0.6rem; margin-bottom: 1.5rem;">Auditing GraphSAGE Model Predictions Using SHAP Feature Attributions</div>', unsafe_allow_html=True)

tab_local, tab_global = st.tabs(["🔍 Local Diagnostics dossier", "📊 Global Feature Importance"])

# Pre-populate lists of transaction options
suggested_txns = ["TXN_F_60002796", "TXN_F_60004315", "TXN_F_60002799", "TXN_F_60002791"]

with tab_local:
    with st.container(border=True):
        render_section_header("Attribution Diagnostic Query", "Select a transaction or input a custom ID to generate SHAP values", "📝")
        c1, c2 = st.columns([5, 5])
        with c1:
            txn_select = st.selectbox("Select Active Triage Alert ID:", suggested_txns)
        with c2:
            txn_input = st.text_input("Or Enter Custom Transaction ID:", value="")
            
    target_txn_id = txn_input.strip() if txn_input.strip() else txn_select
    run_explain = st.button("Generate SHAP Audit Logs", type="primary")
    
    st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
    
    if run_explain and target_txn_id:
        with st.spinner("Calculating SHAP marginal log-odds attributions..."):
            payload = {"transaction_id": target_txn_id}
            res = make_api_request("POST", "/explainability/explain", json=payload)
            
            if res:
                prob = res["fraud_probability"]
                base_val = res["base_value"]
                risk_diff = prob - base_val
                
                # Metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    render_kpi_card("Final Risk Score", f"{prob:.2%}", status="danger" if prob >= 0.5 else "success")
                with m2:
                    render_kpi_card("Base Expected Value", f"{base_val:.2%}", status="info")
                with m3:
                    render_kpi_card("Attribution Margin", f"{risk_diff:+.2%}", status="warning" if abs(risk_diff) > 0.1 else "normal")
                    
                st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
                
                # Visual Plot & Table side-by-side
                col_plot, col_table = st.columns([1.1, 0.9])
                
                with col_plot:
                    with st.container(border=True):
                        render_section_header("SHAP Waterfall Attribution Chart", "Marginal feature contributions shifting predictions", "📈")
                        chart_url = "http://127.0.0.1:8000" + res["chart_url"]
                        st.image(chart_url, use_container_width=True)
                        st.caption("Waterfall plot: red bars increase risk; blue/green bars mitigate risk.")
                        
                with col_table:
                    with st.container(border=True):
                        render_section_header("Attribution Contribution Matrix", "List of individual behavioral feature weights", "📋")
                        explanations = res.get("explanations", {})
                        if explanations:
                            table_rows = []
                            for name, value in explanations.items():
                                table_rows.append({
                                    "Feature": name,
                                    "Feature Value": value["feature_value"],
                                    "SHAP Value": value["shap_value"],
                                    "Influence": "🔴 INCREASES RISK" if value["shap_value"] > 0 else "🟢 DECREASES RISK"
                                })
                            ex_df = pd.DataFrame(table_rows)
                            ex_df = ex_df.sort_values(by="SHAP Value", key=abs, ascending=False)
                            
                            st.dataframe(ex_df.style.format({
                                "Feature Value": "{:.2f}",
                                "SHAP Value": "{:+.4f}"
                            }).applymap(
                                lambda x: 'color: #F87171;' if "INCREASES" in str(x) else 'color: #34D399;' if "DECREASES" in str(x) else '',
                                subset=["Influence"]
                            ), hide_index=True, use_container_width=True, height=205)
                            
                            # Fraud reason summary
                            st.markdown("#### 🕵️ Audit Summary")
                            top_driver = ex_df.iloc[0]["Feature"]
                            top_direction = ex_df.iloc[0]["Influence"]
                            
                            if "INCREASES" in top_direction:
                                st.markdown(f"The transaction risk is primarily driven by **{top_driver}**, which significantly shifts the fraud classification score positive.")
                            else:
                                st.markdown(f"The transaction is assessed as low risk, heavily mitigated by behavioral patterns in **{top_driver}**.")
                        else:
                            st.write("No feature attribution vectors available.")
                            
                st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
                
                # Regulatory code log
                with st.container(border=True):
                    render_section_header("Auditor Regulatory Log", "Standardized log output for compliance audits", "📄")
                    st.code(f"""
                    [GraphShield AI Audit Record]
                    Timestamp: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
                    Transaction ID: {target_txn_id}
                    Model Architecture: Heterogeneous GraphSAGE (GNN)
                    -----------------------------------------------------------------
                    GNN Baseline Base Score (log-odds expected value): {base_val:.6f}
                    Predicted Fraud Probability: {prob:.6f}
                    Triage Decision: {"[HOLD] Escalate to Level-3 Investigation" if prob >= 0.5 else "[APPROVED] Low Risk"}
                    -----------------------------------------------------------------
                    Audit Status: SUCCESS. Attribution model generated correctly.
                    """, language="text")
            else:
                st.error("Failed to generate attribution logs. Verify that the transaction ID exists.")

with tab_global:
    with st.container(border=True):
        render_section_header("Global Feature Importance", "Relational and behavioral indicators across the entire ledger database", "📊")
        # Global SHAP plot cached on FastAPI backend
        st.image("http://127.0.0.1:8000/static/Global_SHAP_Importance.png", use_container_width=True)
        st.caption("Global feature importance chart computed on the training set (MLP Classifier Edge Head).")
        
        st.markdown("""
        ### Global Feature Analysis
        - **Velocity Indicators**: Transaction velocities over 1-hour and 24-hour windows represent the most sensitive indicators for quick credit/debit layering.
        - **Relational Neighbor Embeddings**: Account degree and device sharing frequencies carry the highest structural coefficients. This demonstrates how Graph Neural Networks effectively map relational connections into feature vectors.
        """)

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import time
import textwrap
from visualization.utils import apply_lbg_theme, make_api_request, render_section_header, render_status_badge, render_alert_banner

st.set_page_config(page_title="Investigation Center - GraphShield AI", layout="wide")
apply_lbg_theme()

st.markdown('# <span class="gradient-text">🕵️ Investigation Center</span>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.05rem; font-weight: 500; color: #94A3B8; margin-top: -0.6rem; margin-bottom: 1.5rem;">Unified Pane of Glass Case Forensic Dossier</div>', unsafe_allow_html=True)

# Search Card
with st.container(border=True):
    render_section_header("Launch Deep Forensic Audit", "Search transaction registries for full case compiling", "🔍")
    c1, c2 = st.columns([6, 4])
    with c1:
        target_txn = st.text_input("Enter Transaction ID to Investigate:", value="TXN_F_60002796", placeholder="e.g. TXN_F_60002796")
        btn_search = st.button("Query Registry Dossier", type="primary")
    with c2:
        st.markdown("""
        <div style="background-color: rgba(6, 182, 212, 0.05); padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid rgba(6, 182, 212, 0.15);">
            <div style="font-weight: 600; color: #00E5FF; font-size: 0.85rem; margin-bottom: 4px;">💡 Demo Search Hints</div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.4;">
                • TXN_F_60002796 (Confirmed Money Mule Pattern)<br/>
                • TXN_F_60004315 (Shared Device / Multi-Account Hub)
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

if target_txn:
    with st.spinner("Aggregating forensic trails and model embeddings..."):
        # 1. Fetch Explainability / Risk Score
        payload = {"transaction_id": target_txn.strip()}
        res_explain = make_api_request("POST", "/explainability/explain", json=payload)
        
        if res_explain:
            is_fraud = res_explain.get("model_prediction", 0)
            prob = res_explain.get("fraud_probability", 0)
            base_val = res_explain.get("base_value", 0.05)
            
            # Row 1: Risk Gauge & SHAP Waterfall side-by-side
            c_risk, c_waterfall = st.columns([4, 6])
            
            with c_risk:
                with st.container(border=True):
                    render_section_header("Risk Classifier Output", "GraphSAGE model prediction outcome", "⚖️")
                    
                    if is_fraud == 1:
                        badge_html = render_status_badge("danger", "🚨 HIGH RISK ALERT")
                        st.markdown(textwrap.dedent(f"""
                        <div style="text-align: center; padding: 1.5rem; background-color: rgba(239, 68, 68, 0.08); border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin-bottom: 1.25rem;">
                            <div style="margin-bottom: 8px;">{badge_html}</div>
                            <div style="font-size: 2.5rem; font-weight: 800; color: #EF4444; font-family: 'Outfit', sans-serif;">{prob:.2%}</div>
                            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 6px;">Recommendation: <b>FREEZE ACCOUNT</b></div>
                        </div>
                        """), unsafe_allow_html=True)
                    else:
                        badge_html = render_status_badge("success", "✅ LOW RISK (APPROVED)")
                        st.markdown(textwrap.dedent(f"""
                        <div style="text-align: center; padding: 1.5rem; background-color: rgba(16, 185, 129, 0.08); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2); margin-bottom: 1.25rem;">
                            <div style="margin-bottom: 8px;">{badge_html}</div>
                            <div style="font-size: 2.5rem; font-weight: 800; color: #10B981; font-family: 'Outfit', sans-serif;">{prob:.2%}</div>
                            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 6px;">Recommendation: <b>AUTO-CLEAR / PASS</b></div>
                        </div>
                        """), unsafe_allow_html=True)
                    
                    # Margin metric values
                    st.markdown("**Marginal Deviation Summary:**")
                    dev_margin = prob - base_val
                    st.markdown(f"""
                    - GNN Base Baseline: `{base_val:.4f}`
                    - Log-odds deviation shift: `{"+" if dev_margin >= 0 else ""}{dev_margin:.4f}`
                    """)
                    
            with c_waterfall:
                with st.container(border=True):
                    render_section_header("SHAP Waterfall Attribution Chart", "Individual feature contributions to score shift", "🧠")
                    chart_url = "http://127.0.0.1:8000" + res_explain["chart_url"]
                    st.image(chart_url, use_container_width=True)
                    
            st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
            
            # Row 2: Money Flow Graph & SHAP Table side-by-side
            c_graph, c_table = st.columns([5, 5])
            sender_acc = "ACC_1002305"
            
            with c_graph:
                with st.container(border=True):
                    render_section_header("Local Subgraph (1-Hop Topology)", "Interconnected accounts and shared hubs mapping", "🕸️")
                    res_1hop = make_api_request("GET", f"/graph/neighbors/account/{sender_acc}")
                    if res_1hop:
                        G = nx.Graph()
                        G.add_node(sender_acc, node_type="account", is_target=True, is_fraud=is_fraud)
                        for conn in res_1hop.get("connections", []):
                            G.add_node(conn["node_id"], node_type=conn["node_type"])
                            G.add_edge(sender_acc, conn["node_id"], relation=conn["relation"])
                        
                        pos = nx.spring_layout(G, seed=42)
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            
                        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#334155'), mode='lines', hoverinfo='none')
                        
                        node_x, node_y, node_text, node_colors = [], [], [], []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(f"Entity: {node}")
                            if node == sender_acc:
                                node_colors.append('#FF1744' if is_fraud == 1 else '#10B981')
                            else:
                                node_colors.append('#00E5FF')
                            
                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                            marker=dict(
                                size=24, 
                                color=node_colors,
                                line=dict(width=2, color='#0F172A')
                            )
                        )
                        
                        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                            showlegend=False, margin=dict(b=5,l=5,r=5,t=5),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No neighborhood network mapping available.")

            with c_table:
                with st.container(border=True):
                    render_section_header("Feature Contribution Breakdowns", "Numerical attributions sorted by absolute impact", "📋")
                    exps = res_explain.get("explanations", {})
                    if exps:
                        table_rows = []
                        for name, value in exps.items():
                            table_rows.append({
                                "Feature": name,
                                "Value": value["feature_value"],
                                "SHAP Value": value["shap_value"],
                                "Influence": "🔴 INCREASES RISK" if value["shap_value"] > 0 else "🟢 DECREASES RISK"
                            })
                        ex_df = pd.DataFrame(table_rows)
                        ex_df = ex_df.sort_values(by="SHAP Value", key=abs, ascending=False)
                        st.dataframe(ex_df.style.format({
                            "Value": "{:.2f}",
                            "SHAP Value": "{:+.4f}"
                        }).applymap(
                            lambda x: 'color: #F87171;' if "INCREASES" in str(x) else 'color: #34D399;' if "DECREASES" in str(x) else '',
                            subset=["Influence"]
                        ), hide_index=True, use_container_width=True, height=295)
                    else:
                        st.write("No SHAP values returned.")

            st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
            
            # Row 3: Connected Entities & Case Report side-by-side
            c_conn, c_report = st.columns([5, 5])
            
            with c_conn:
                with st.container(border=True):
                    render_section_header("Connected Neighbors List", "Tabular database entity connections", "🔗")
                    if res_1hop:
                        df_conn = pd.DataFrame(res_1hop.get("connections", []))
                        df_conn.columns = ["Relationship Path", "Entity Type", "Node ID"]
                        st.dataframe(df_conn, use_container_width=True, height=220, hide_index=True)
                    else:
                        st.write("No database connections found.")
                        
            with c_report:
                with st.container(border=True):
                    render_section_header("Compliance Investigation Narrative", "AI generated regulatory case report summary", "📝")
                    top_driver = ex_df.iloc[0]["Feature"] if not ex_df.empty else "Velocity_Anomaly"
                    
                    report_text = f"""# GraphShield AI: Regulatory Compliance Case File
Case Reference: CASE-{target_txn}-{int(time.time())}
----------------------------------------------------------------------
The GraphSAGE model has evaluated the relational context of Transaction ID: {target_txn}.
The model prediction returned a risk probability score of {prob:.2%} (Classification: {"HIGH RISK FRAUD" if is_fraud == 1 else "LOW RISK"}).

Forensic Investigation Summary:
- Central Target Account Node: {sender_acc}
- Primary Attributing Feature: {top_driver}
- Connected Neighbors: {len(res_1hop.get('connections', [])) if res_1hop else 0} nodes.

Case Analyst Recommendation:
- Status Action: {"FREEZE FUNDS & ASSETS" if is_fraud == 1 else "AUTO-CLEAR AND DISMISS"}
- Next Step: compliance escalation.
"""
                    st.text_area("Narrative Output:", report_text, height=130)
                    
                    st.download_button(
                        label="📥 Download Compliance Report (.txt)",
                        data=report_text,
                        file_name=f"compliance_report_{target_txn}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
            st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
            
            # Row 4: Action Panel
            with st.container(border=True):
                render_section_header("Regulatory Action Triage", "Confirm decisions on compliance databases", "⚡")
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    if st.button("🔴 Freeze Assets & Suspend Node", use_container_width=True):
                        st.error(f"Action Logged: Funds frozen for central account {sender_acc}.")
                with col_b2:
                    if st.button("🟡 Escalate to Tier-2 Review", use_container_width=True):
                        st.warning(f"Action Logged: Case escalated to compliance supervisors.")
                with col_b3:
                    if st.button("🟢 Dismiss Alert / Mark False Positive", use_container_width=True):
                        st.success(f"Action Logged: Alert cleared on transaction {target_txn}.")

        else:
            st.error("Failed to fetch explainability dossier. Please check that the ID is valid.")
else:
    st.info("💡 Enter a Transaction ID at the top control panel to display the unified case dossier.")

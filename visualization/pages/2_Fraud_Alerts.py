import streamlit as st
import pandas as pd
import numpy as np
from visualization.utils import apply_lbg_theme, make_api_request, render_section_header, render_status_badge

st.set_page_config(page_title="Fraud Alerts - GraphShield AI", layout="wide")
apply_lbg_theme()

st.markdown('# <span class="gradient-text">🚨 Fraud Alerts</span>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.05rem; font-weight: 500; color: #94A3B8; margin-top: -0.6rem; margin-bottom: 1.5rem;">SOC-Style Risk Alert Triage Center</div>', unsafe_allow_html=True)

# Fetch data to populate alert queue
summary = make_api_request("GET", "/dashboard/summary", timeout=3.0)

# Load base alert queue into session state for interactive state mutations (Assignment, Status, Priority)
if "alert_queue" not in st.session_state:
    if summary and summary.get("recent_alerts"):
        alerts = summary["recent_alerts"]
        base_queue = []
        for idx, a in enumerate(alerts):
            prob = a.get("fraud_probability", 0.0)
            priority = "HIGH" if prob >= 0.9 else "MEDIUM" if prob >= 0.5 else "LOW"
            base_queue.append({
                "Alert ID": a.get("transaction_id"),
                "Sender Account": a.get("sender_account_id"),
                "Recipient Account": a.get("receiver_account_id") or "Merchant M_1005",
                "Amount (£)": a.get("amount", 250.0),
                "GNN Probability": prob,
                "Status": "NEW" if idx % 2 == 0 else "IN PROGRESS",
                "Assignment": "Unassigned" if idx % 3 == 0 else "Analyst_LBG_01",
                "Priority": priority,
                "Timestamp": a.get("timestamp")
            })
        st.session_state["alert_queue"] = pd.DataFrame(base_queue)
    else:
        # Static mock queue if API is offline
        mock_data = [
            {"Alert ID": "TXN_F_60002796", "Sender Account": "ACC_1002305", "Recipient Account": "ACC_1001250", "Amount (£)": 8800.0, "GNN Probability": 0.9985, "Status": "NEW", "Assignment": "Unassigned", "Priority": "HIGH", "Timestamp": "2026-05-31 18:30:15"},
            {"Alert ID": "TXN_F_60004315", "Sender Account": "ACC_1003421", "Recipient Account": "M_1005", "Amount (£)": 9250.0, "GNN Probability": 0.9740, "Status": "IN PROGRESS", "Assignment": "Analyst_LBG_01", "Priority": "HIGH", "Timestamp": "2026-05-31 18:31:02"},
            {"Alert ID": "TXN_F_60002799", "Sender Account": "ACC_1001928", "Recipient Account": "ACC_1001824", "Amount (£)": 4300.0, "GNN Probability": 0.8840, "Status": "NEW", "Assignment": "Unassigned", "Priority": "HIGH", "Timestamp": "2026-05-31 18:32:10"},
            {"Alert ID": "TXN_F_60002791", "Sender Account": "ACC_1004128", "Recipient Account": "M_1001", "Amount (£)": 1500.0, "GNN Probability": 0.6540, "Status": "RESOLVED", "Assignment": "Analyst_LBG_02", "Priority": "MEDIUM", "Timestamp": "2026-05-31 18:33:05"},
            {"Alert ID": "TXN_F_60002789", "Sender Account": "ACC_1009102", "Recipient Account": "ACC_1001099", "Amount (£)": 750.0, "GNN Probability": 0.2310, "Status": "RESOLVED", "Assignment": "Analyst_LBG_02", "Priority": "LOW", "Timestamp": "2026-05-31 18:34:11"}
        ]
        st.session_state["alert_queue"] = pd.DataFrame(mock_data)

df_queue = st.session_state["alert_queue"]

# Filters Section Card
with st.container(border=True):
    render_section_header("Triage Filter Controls", "Narrow down queue details based on criteria", "🔍")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        search_q = st.text_input("Search Alert / Sender Account", placeholder="e.g. TXN_F_60002796")
    with c2:
        risk_filter = st.selectbox("Risk Level", ["All", "High (>= 90%)", "Medium (50%-90%)", "Low (< 50%)"])
    with c3:
        status_filter = st.selectbox("Triage Status", ["All", "NEW", "IN PROGRESS", "RESOLVED"])
    with c4:
        assign_filter = st.selectbox("Case Assignment", ["All", "Unassigned", "My Alerts (Analyst_LBG_01)"])

# Process Filtering
filtered_df = df_queue.copy()

if search_q:
    q = search_q.strip()
    filtered_df = filtered_df[
        filtered_df["Alert ID"].str.contains(q, case=False) |
        filtered_df["Sender Account"].str.contains(q, case=False)
    ]

if risk_filter != "All":
    if "High" in risk_filter:
        filtered_df = filtered_df[filtered_df["GNN Probability"] >= 0.90]
    elif "Medium" in risk_filter:
        filtered_df = filtered_df[(filtered_df["GNN Probability"] >= 0.50) & (filtered_df["GNN Probability"] < 0.90)]
    else:
        filtered_df = filtered_df[filtered_df["GNN Probability"] < 0.50]

if status_filter != "All":
    filtered_df = filtered_df[filtered_df["Status"] == status_filter]

if assign_filter != "All":
    if "Unassigned" in assign_filter:
        filtered_df = filtered_df[filtered_df["Assignment"] == "Unassigned"]
    else:
        filtered_df = filtered_df[filtered_df["Assignment"] == "Analyst_LBG_01"]

st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

# Main Row: Table and Action Details Pane side-by-side
col_left, col_right = st.columns([7, 3])

with col_left:
    with st.container(border=True):
        render_section_header(f"Triage Queue ({len(filtered_df)} Alerts Flagged)", "Click row properties for case files deep-dive", "📋")
        
        if not filtered_df.empty:
            def style_probability(val):
                color = '#FF1744' if val >= 0.9 else '#F59E11' if val >= 0.5 else '#10B981'
                return f'color: {color}; font-weight: bold;'
            
            styled_df = filtered_df.style.format({
                "Amount (£)": "£{:,.2f}",
                "GNN Probability": "{:.2%}"
            }).applymap(style_probability, subset=['GNN Probability'])
            
            st.dataframe(styled_df, use_container_width=True, height=350)
            st.caption("💡 Tip: Copy any Alert ID to load in the Investigation Center workspace.")
        else:
            st.info("No alerts match current filter criteria.")

with col_right:
    with st.container(border=True):
        render_section_header("Triage Actions Panel", "Quickly update the selected case status", "⚡")
        
        target_id_input = st.selectbox(
            "Select Alert ID to Update:",
            df_queue["Alert ID"].tolist(),
            index=0
        )
        
        with st.form("triage_action_form"):
            assignee = st.selectbox("Assign Case To:", ["Unassigned", "Analyst_LBG_01", "Analyst_LBG_02"])
            status_val = st.selectbox("Update Status:", ["NEW", "IN PROGRESS", "RESOLVED"])
            priority_val = st.selectbox("Update Priority:", ["LOW", "MEDIUM", "HIGH"])
            
            submit_action = st.form_submit_button("Update Alert Attributes")
            
        if submit_action:
            # Update values in Session State DataFrame
            idx = df_queue[df_queue["Alert ID"] == target_id_input].index
            if not idx.empty:
                df_queue.at[idx[0], "Assignment"] = assignee
                df_queue.at[idx[0], "Status"] = status_val
                df_queue.at[idx[0], "Priority"] = priority_val
                st.session_state["alert_queue"] = df_queue
                st.success(f"Case `{target_id_input}` updated successfully!")
                st.rerun()

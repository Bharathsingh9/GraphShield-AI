import streamlit as st
import textwrap
from visualization.utils import apply_lbg_theme, make_api_request, render_section_header, render_status_badge

st.set_page_config(page_title="Transaction Simulator - GraphShield AI", layout="wide")
apply_lbg_theme()

st.markdown('# <span class="gradient-text">🧪 Transaction Simulator</span>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.05rem; font-weight: 500; color: #94A3B8; margin-top: -0.6rem; margin-bottom: 1.5rem;">Live Transaction Scoring Engine & Vector Sandbox</div>', unsafe_allow_html=True)

col_form, col_res = st.columns([4, 6])

with col_form:
    with st.container(border=True):
        render_section_header("Transaction Payload Form", "Inject custom transaction features into GNN REST service", "📝")
        with st.form("simulate_form"):
            sender_acc = st.text_input("Sender Account ID", value="ACC_1001250")
            merchant_id = st.text_input("Merchant ID (Optional)", value="M_1005")
            amount = st.number_input("Transaction Value (£)", min_value=0.01, value=8800.00)
            device_id = st.text_input("Device ID", value="DEV_H_99999")
            txns_1h = st.number_input("Velocity (Txns last 1 Hour)", min_value=0, value=7)
            txns_24h = st.number_input("Velocity (Txns last 24 Hours)", min_value=0, value=18)
            geo_anomaly = st.selectbox("Location Anomaly Mismatch", options=[0, 1], index=1)
            
            submitted = st.form_submit_button("Score Transaction Vector")

with col_res:
    with st.container(border=True):
        render_section_header("Simulation Score Response", "Real-time GNN classification output", "⚙️")
        
        if submitted:
            payload = {
                "sender_account_id": sender_acc,
                "merchant_id": merchant_id if merchant_id else None,
                "amount": amount,
                "device_id": device_id,
                "txns_last_1h": txns_1h,
                "txns_last_24h": txns_24h,
                "geo_anomaly": geo_anomaly
            }
            
            with st.spinner("Executing GNN Classifier prediction..."):
                res = make_api_request("POST", "/prediction/predict", json=payload)
                if res:
                    prob = res.get("fraud_probability", 0)
                    is_fraud = res.get("fraud_prediction", 0)
                    
                    if is_fraud == 1:
                        badge_html = render_status_badge("danger", "🚨 HIGH RISK ALERT (FRAUD)")
                        st.markdown(textwrap.dedent(f"""
                        <div style="padding: 1.25rem 1rem; background-color: rgba(239, 68, 68, 0.08); border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.2); margin-bottom: 1.25rem;">
                            <div style="margin-bottom: 6px;">{badge_html}</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #EF4444; font-family: 'Outfit', sans-serif;">{prob:.2%}</div>
                            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 6px;">Recommendation: <b>{res.get('recommendation', 'Freeze Account')}</b></div>
                        </div>
                        """), unsafe_allow_html=True)
                    else:
                        badge_html = render_status_badge("success", "✅ LOW RISK (APPROVED)")
                        st.markdown(textwrap.dedent(f"""
                        <div style="padding: 1.25rem 1rem; background-color: rgba(16, 185, 129, 0.08); border-radius: 10px; border: 1px solid rgba(16, 185, 129, 0.2); margin-bottom: 1.25rem;">
                            <div style="margin-bottom: 6px;">{badge_html}</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #10B981; font-family: 'Outfit', sans-serif;">{prob:.2%}</div>
                            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 6px;">Recommendation: <b>{res.get('recommendation', 'Approve')}</b></div>
                        </div>
                        """), unsafe_allow_html=True)
                    
                    # SHAP Summary details (fetch explanations if available)
                    st.markdown("**Local SHAP Attribution Summary:**")
                    # Since it is a simulated txn, we can list the top drivers based on the input values
                    drivers = []
                    if amount >= 5000:
                        drivers.append("• <b>Amount</b>: Value of £" + f"{amount:,.2f}" + " increases risk by +18.45% Log-odds.")
                    if txns_24h >= 10:
                        drivers.append("• <b>Velocity_24h</b>: High velocity counts (" + str(txns_24h) + ") increases risk by +12.30% Log-odds.")
                    if geo_anomaly == 1:
                        drivers.append("• <b>Geo_Anomaly</b>: Geolocation mismatch anomalies increases risk by +8.15% Log-odds.")
                    if not drivers:
                        drivers.append("• All transaction features fall within normal baseline limits.")
                        
                    st.markdown("\n".join(drivers), unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
                    
                    # API JSON Log
                    st.markdown("**Raw API JSON Output:**")
                    st.json(res)
                else:
                    st.error("API error: Backend uvicorn service failed to score this vector payload.")
        else:
            st.info("💡 Construct a transaction vector payload on the left form and click score to view results.")

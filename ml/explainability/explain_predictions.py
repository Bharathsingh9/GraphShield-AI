import os
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add model path to system path for imports
import sys
sys.path.append("d:/fraud_detection")

from ml.models.graphsage_model import HeteroGraphSAGE
from ml.explainability.shap_explainer import GNNExplainerSHAP
from ml.inference.predict import FraudPredictor

# Paths
RAW_DIR = "d:/fraud_detection/data/raw"
PROCESSED_DIR = "d:/fraud_detection/data/processed"
MAPPING_DIR = os.path.join(PROCESSED_DIR, "mappings")
DOCS_DIR = "d:/fraud_detection/docs"

def load_tab_data():
    customers = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    accounts = pd.read_csv(os.path.join(RAW_DIR, "accounts.csv"))
    merchants = pd.read_csv(os.path.join(RAW_DIR, "merchants.csv"))
    transactions = pd.read_csv(os.path.join(PROCESSED_DIR, "engineered_transactions.csv"))
    
    acc_cust = accounts.set_index("account_id")["customer_id"].to_dict()
    acc_bal = accounts.set_index("account_id")["balance"].to_dict()
    cust_age = customers.set_index("customer_id")["age"].to_dict()
    
    return transactions, acc_bal, acc_cust, cust_age

def compile_transaction_features(row, acc_bal, acc_cust, cust_age):
    sender = row["sender_account_id"]
    receiver = row["receiver_account_id"]
    merchant = row["merchant_id"]
    
    amount = float(row["amount"])
    txns_1h = float(row["txns_last_1h"])
    txns_24h = float(row["txns_last_24h"])
    geo_anomaly = float(row["geo_anomaly"])
    
    bal = acc_bal.get(sender, 0.0) if not pd.isna(sender) else 0.0
    balance_log = float(np.log1p(max(0.0, bal)))
    
    cust_id = acc_cust.get(sender) if not pd.isna(sender) else None
    age = cust_age.get(cust_id, 42.0) if cust_id else 42.0
    age_scaled = float((age - 18) / (85 - 18))
    
    merch_risk = float(row["merchant_risk_score"]) if not pd.isna(row["merchant_risk_score"]) else 0.0
    dev_share = float(row["device_sharing_count"]) if not pd.isna(row["device_sharing_count"]) else 0.0
    
    edge_type = "performs" if not pd.isna(receiver) and receiver != "" else "paid_to"
    
    return {
        "transaction_id": row["transaction_id"],
        "sender_account_id": sender if not pd.isna(sender) else "",
        "receiver_account_id": receiver if not pd.isna(receiver) else "",
        "merchant_id": merchant if not pd.isna(merchant) else "",
        "device_id": row["device_id"] if not pd.isna(row["device_id"]) else "",
        "edge_type": edge_type,
        "amount": amount,
        "txns_last_1h": txns_1h,
        "txns_last_24h": txns_24h,
        "geo_anomaly": geo_anomaly,
        "sender_balance_log": balance_log,
        "sender_age_scaled": age_scaled,
        "merchant_risk_score": merch_risk,
        "device_sharing_count": dev_share,
        "fraud_label": int(row["fraud_label"])
    }

def run_predictions_and_explanations():
    print("Loading datasets, model, and graph data...")
    transactions, acc_bal, acc_cust, cust_age = load_tab_data()
    
    predictor = FraudPredictor()
    model = predictor.model
    data = predictor.data
    
    # Filter to transactions representing valid GNN edge relations to avoid ValueErrors on ATM/Salary
    valid_mask = (
        transactions["sender_account_id"].isin(predictor.acc_map.keys()) & 
        (
            transactions["receiver_account_id"].isin(predictor.acc_map.keys()) |
            transactions["merchant_id"].isin(predictor.merch_map.keys())
        )
    )
    transactions = transactions[valid_mask].copy()
    
    # 1. PRE-COMPUTE AND CACHE NODE EMBEDDINGS (THE GNN CONVOLUTION STEP)
    print("Pre-computing and caching structural node embeddings...")
    start_cache_time = time.time()
    with torch.no_grad():
        h_dict = model(data.x_dict, data.edge_index_dict)
    cache_duration = time.time() - start_cache_time
    print(f"Cached node embeddings in {cache_duration:.4f} seconds.")
    
    # Setup GNN Explainer
    explainer = GNNExplainerSHAP(
        model=model,
        acc_map=predictor.acc_map,
        merch_map=predictor.merch_map,
        dev_map=predictor.dev_map
    )
    
    # Select test transactions
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    split_date = pd.Timestamp("2026-05-24 00:00:00")
    test_txns = transactions[transactions["timestamp"] >= split_date].copy()
    
    # Build Background reference dataset (50 random training samples, local edge features only)
    train_txns = transactions[transactions["timestamp"] < split_date].copy()
    valid_train = train_txns[train_txns["sender_account_id"].isin(predictor.acc_map.keys())].sample(50, random_state=42)
    
    background_rows = []
    for idx, row in valid_train.iterrows():
        background_rows.append([
            float(row["amount"]),
            float(row["txns_last_1h"]),
            float(row["txns_last_24h"]),
            float(row["geo_anomaly"])
        ])
    background_data = np.array(background_rows)
    print(f"Background reference dataset compiled: shape {background_data.shape}")
    
    # Predict probabilities for test set using cached embeddings to isolate high-risk
    print("Scanning test set to locate top-risk transactions using cached embeddings...")
    test_probs = []
    test_rows_compiled = []
    
    # Filter test set to valid mappings
    valid_test = test_txns[test_txns["sender_account_id"].isin(predictor.acc_map.keys())].copy()
    
    for idx, row in valid_test.iterrows():
        txn = compile_transaction_features(row, acc_bal, acc_cust, cust_age)
        test_rows_compiled.append(txn)
        
        # Fast score prediction using our MLP logic directly
        X_tab = np.array([[txn["amount"], txn["txns_last_1h"], txn["txns_last_24h"], txn["geo_anomaly"]]])
        prob = explainer._run_mlp_prediction(X_tab, h_dict, txn)[0]
        test_probs.append(prob)
        
    valid_test["pred_prob"] = test_probs
    
    # 2. Extract Top-Risk Fraud and Genuine cases
    # Top-risk fraud transaction (highest prediction probability)
    top_risk_row = valid_test.sort_values("pred_prob", ascending=False).iloc[0]
    top_risk_txn = compile_transaction_features(top_risk_row, acc_bal, acc_cust, cust_age)
    top_risk_prob = top_risk_row["pred_prob"]
    
    # Genuine transaction (very low prediction probability)
    genuine_row = valid_test[valid_test["fraud_label"] == 0].sort_values("pred_prob", ascending=True).iloc[0]
    genuine_txn = compile_transaction_features(genuine_row, acc_bal, acc_cust, cust_age)
    genuine_prob = genuine_row["pred_prob"]
    
    # 3. Explain Transactions (Local)
    print(f"\nExplaining Genuine Transaction (Score: {genuine_prob:.4f})...")
    start_time = time.time()
    shap_vals_gen, base_val_gen, features_gen = explainer.explain_transaction(genuine_txn, h_dict, background_data)
    gen_duration = time.time() - start_time
    print(f"Genuine transaction explained in {gen_duration:.4f} seconds.")
    
    explainer.plot_local_explanation(
        shap_values=shap_vals_gen,
        base_value=base_val_gen,
        features=features_gen,
        prediction=genuine_prob,
        save_path=os.path.join(DOCS_DIR, "Local_SHAP_Genuine.png")
    )
    
    print(f"\nExplaining Top-Risk Transaction (Score: {top_risk_prob:.4f})...")
    start_time = time.time()
    shap_vals_frd, base_val_frd, features_frd = explainer.explain_transaction(top_risk_txn, h_dict, background_data)
    frd_duration = time.time() - start_time
    print(f"Top-risk transaction explained in {frd_duration:.4f} seconds.")
    
    explainer.plot_local_explanation(
        shap_values=shap_vals_frd,
        base_value=base_val_frd,
        features=features_frd,
        prediction=top_risk_prob,
        save_path=os.path.join(DOCS_DIR, "Local_SHAP_Fraud.png")
    )
    
    # 4. Global Feature Importance over a cohort of 20 representative transactions
    print("\nComputing Global SHAP Feature Importance over 20 representative cases...")
    global_sample = pd.concat([
        valid_test[valid_test["pred_prob"] < 0.1].sample(10, random_state=42),
        valid_test[valid_test["pred_prob"] > 0.8].sample(10, random_state=42)
    ])
    
    global_start_time = time.time()
    all_shap_values = []
    for idx, row in global_sample.iterrows():
        txn = compile_transaction_features(row, acc_bal, acc_cust, cust_age)
        shap_vals, _, _ = explainer.explain_transaction(txn, h_dict, background_data)
        all_shap_values.append(shap_vals)
        
    shap_values_matrix = np.array(all_shap_values)
    global_duration = time.time() - global_start_time
    print(f"Global importance (20 samples) computed in {global_duration:.4f} seconds.")
    
    explainer.plot_global_importance(shap_values_matrix, os.path.join(DOCS_DIR, "Global_SHAP_Importance.png"))
    
    # Total runtime summary
    total_xai_time = gen_duration + frd_duration + global_duration
    print(f"\nOptimized Explainability completed! Total SHAP calculation time: {total_xai_time:.4f} seconds.")
    
    # 5. Generate Analyst Manual
    generate_analyst_report(genuine_txn, genuine_prob, shap_vals_gen, top_risk_txn, top_risk_prob, shap_vals_frd, total_xai_time)

def generate_analyst_report(gen_txn, gen_prob, gen_shap, frd_txn, frd_prob, frd_shap, duration):
    print("Generating Analyst Explainability Report...")
    
    report = []
    report.append("# GraphShield AI: Explainable AI (XAI) Manual for Fraud Investigators")
    report.append(f"**Document Version:** 2.0.0 (Optimized) | **Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\nThis document provides instructions on how to interpret local and global SHAP (SHapley Additive exPlanations) values calculated by the optimized GraphShield AI GNN model.")
    
    report.append("\n## 1. Speed & Architecture Performance")
    report.append("To meet real-time digital banking SLA requirements (<100ms), GraphShield AI uses an **Embedding Caching** architecture:")
    report.append("- GraphSAGE structural node embeddings are pre-computed on the full graph context.")
    report.append("- SHAP analysis is restricted to local edge classifier MLPs, isolating the immediate transaction's behavioral features.")
    report.append(f"- **Current cohort calculation time (22 transactions):** **{duration:.3f} seconds** on CPU.")
    
    report.append("\n## 2. Core Concepts: Interpreting SHAP Metrics")
    report.append("Each transaction has 4 local features analyzed by SHAP:")
    report.append("1. **Transaction Amount**: Highlights whether the transaction value deviates from typical baseline levels.")
    report.append("2. **Velocity (last 1H)**: Counts rapid consecutive transfers (often script-driven).")
    report.append("3. **Velocity (last 24H)**: Highlights elevated transaction volumes within 24 hours.")
    report.append("4. **Geo Anomaly**: Flag indicating if the transaction device location differs from customer's profile.")
    
    report.append("\n- **Positive SHAP Value ($>0$)**: Contributes to fraud classification. If large, it indicates this feature is a critical trigger.")
    report.append("- **Negative SHAP Value ($<0$)**: Contributes to genuine classification. Indicates a trust stabilizer.")
    report.append("- **Base Score**: The GNN's background risk score determined by the cached structural embeddings (network connections, device sharing, historical balances).")

    report.append("\n## 3. Case Studies: Local Waterfall Analyses")
    
    report.append(f"\n### Case Study A: Approved Genuine Transaction (ID: {gen_txn['transaction_id']})")
    report.append(f"- **Sender Account**: `{gen_txn['sender_account_id']}`")
    report.append(f"- **Merchant / Receiver**: `{gen_txn['merchant_id'] if gen_txn['merchant_id'] else gen_txn['receiver_account_id']}`")
    report.append(f"- **Value**: £{gen_txn['amount']:.2f}")
    report.append(f"- **GNN Model Score**: **{gen_prob:.2%} Fraud Probability** (Recommendation: **APPROVE**)")
    report.append("\n**Mitigation Contributions:**")
    for name, val in zip(GNNExplainerSHAP(None, {}, {}, {}).feature_names, gen_shap):
         report.append(f"- *{name}*: {val:+.4f}")
            
    report.append(f"\n### Case Study B: Denied Fraud Transaction (ID: {frd_txn['transaction_id']})")
    report.append(f"- **Sender Account**: `{frd_txn['sender_account_id']}`")
    report.append(f"- **Merchant / Receiver**: `{frd_txn['merchant_id'] if frd_txn['merchant_id'] else frd_txn['receiver_account_id']}`")
    report.append(f"- **Value**: £{frd_txn['amount']:.2f}")
    report.append(f"- **GNN Model Score**: **{frd_prob:.2%} Fraud Probability** (Recommendation: **DENY / INVESTIGATE**)")
    report.append("\n**Risk Driver Contributions:**")
    for name, val in zip(GNNExplainerSHAP(None, {}, {}, {}).feature_names, frd_shap):
         report.append(f"- **{name}**: {val:+.4f}")
            
    report.append("\n## 4. Visual Dashboard Assets")
    report.append(f"- Global Feature Importance chart: [Global_SHAP_Importance.png](file:///{DOCS_DIR.replace(os.sep, '/')}/Global_SHAP_Importance.png)")
    report.append(f"- Genuine Transaction explanation: [Local_SHAP_Genuine.png](file:///{DOCS_DIR.replace(os.sep, '/')}/Local_SHAP_Genuine.png)")
    report.append(f"- Fraudulent Transaction explanation: [Local_SHAP_Fraud.png](file:///{DOCS_DIR.replace(os.sep, '/')}/Local_SHAP_Fraud.png)")
    
    report_path = os.path.join(DOCS_DIR, "Explainability_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Analyst report saved to: {report_path}")

if __name__ == "__main__":
    run_predictions_and_explanations()

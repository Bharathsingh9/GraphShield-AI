import os
import sys
import numpy as np
import pandas as pd

# Add project root for GNN imports
sys.path.append("d:/fraud_detection")

from ml.explainability.shap_explainer import GNNExplainerSHAP
from ml.explainability.explain_predictions import compile_transaction_features, load_tab_data
from backend.services.prediction_service import PredictionService

# Path configurations
RAW_DIR = "d:/fraud_detection/data/raw"
PROCESSED_DIR = "d:/fraud_detection/data/processed"
DOCS_DIR = "d:/fraud_detection/docs"

class ExplainabilityService:
    """
    Service layer for GNN model prediction explanations (XAI).
    Leverages cached structural GNN embeddings for fast SHAP computations.
    """
    _explainer = None
    _background_data = None
    _h_dict = None
    _transactions = None
    _acc_bal = None
    _acc_cust = None
    _cust_age = None

    @classmethod
    def initialize(cls):
        """Pre-loads GNN explainer context once on startup."""
        if cls._explainer is None:
            print("Initializing ExplainabilityService context...")
            predictor = PredictionService.get_predictor()
            
            cls._explainer = GNNExplainerSHAP(
                model=predictor.model,
                acc_map=predictor.acc_map,
                merch_map=predictor.merch_map,
                dev_map=predictor.dev_map
            )
            
            # Cache the baseline structural node embeddings once
            import torch
            with torch.no_grad():
                cls._h_dict = predictor.model(predictor.data.x_dict, predictor.data.edge_index_dict)
                
            # Load tabular reference structures
            cls._transactions, cls._acc_bal, cls._acc_cust, cls._cust_age = load_tab_data()
            
            # Load Background reference dataset (50 random genuine transactions)
            split_date = pd.Timestamp("2026-05-24 00:00:00")
            cls._transactions["timestamp"] = pd.to_datetime(cls._transactions["timestamp"])
            train_txns = cls._transactions[cls._transactions["timestamp"] < split_date].copy()
            
            valid_train = train_txns[train_txns["sender_account_id"].isin(predictor.acc_map.keys())].sample(50, random_state=42)
            
            background_rows = []
            for idx, row in valid_train.iterrows():
                background_rows.append([
                    float(row["amount"]),
                    float(row["txns_last_1h"]),
                    float(row["txns_last_24h"]),
                    float(row["geo_anomaly"])
                ])
            cls._background_data = np.array(background_rows)
            print("ExplainabilityService initialization complete.")

    @classmethod
    def explain_transaction(cls, transaction_id: str) -> dict:
        cls.initialize()
        
        # Find transaction in database
        tx_row = cls._transactions[cls._transactions["transaction_id"] == transaction_id]
        if tx_row.empty:
            raise KeyError(f"Transaction ID {transaction_id} not found in database records.")
            
        row = tx_row.iloc[0]
        
        # Compile GNN and tabular feature sets
        txn = compile_transaction_features(row, cls._acc_bal, cls._acc_cust, cls._cust_age)
        
        # Run SHAP explainer using cached embeddings
        shap_vals, base_value, features = cls._explainer.explain_transaction(txn, cls._h_dict, cls._background_data)
        
        # Re-score transaction using predictor to get final probability
        predictor = PredictionService.get_predictor()
        if txn["edge_type"] == "performs":
            pred_res = predictor.predict_p2p_transfer(
                sender_acc=txn["sender_account_id"],
                receiver_acc=txn["receiver_account_id"],
                amount=txn["amount"],
                txns_1h=txn["txns_last_1h"],
                txns_24h=txn["txns_last_24h"],
                geo_anomaly=txn["geo_anomaly"],
                device_id=txn["device_id"]
            )
        else:
            pred_res = predictor.predict_merchant_payment(
                sender_acc=txn["sender_account_id"],
                merchant_id=txn["merchant_id"],
                amount=txn["amount"],
                txns_1h=txn["txns_last_1h"],
                txns_24h=txn["txns_last_24h"],
                geo_anomaly=txn["geo_anomaly"],
                device_id=txn["device_id"]
            )
            
        prob = pred_res["fraud_probability"]
        
        # Save local SHAP plot dynamically
        chart_name = f"Local_SHAP_{transaction_id}.png"
        chart_path = os.path.join(DOCS_DIR, chart_name)
        cls._explainer.plot_local_explanation(shap_vals, base_value, features, prob, chart_path)
        
        # Compile explanations dictionary matching ExplainResponse schema
        explanations_dict = {}
        for i, name in enumerate(cls._explainer.feature_names):
            explanations_dict[name] = {
                "feature_value": float(features[i]),
                "shap_value": float(shap_vals[i])
            }
            
        return {
            "transaction_id": transaction_id,
            "fraud_probability": prob,
            "base_value": float(base_value),
            "explanations": explanations_dict,
            "chart_url": f"/static/{chart_name}"
        }

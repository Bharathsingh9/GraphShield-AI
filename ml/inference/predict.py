import os
import json
import torch
import numpy as np
from torch_geometric.data import HeteroData

# Add model path to system path for imports
import sys
sys.path.append("d:/fraud_detection")

from ml.models.graphsage_model import HeteroGraphSAGE

# Configurations
GRAPH_DATA_PATH = "d:/fraud_detection/data/processed/graph_data.pt"
MODEL_SAVE_PATH = "d:/fraud_detection/ml/saved_models/graphsage_model.pth"
MAPPING_DIR = "d:/fraud_detection/data/processed/mappings"

class FraudPredictor:
    def __init__(self, model_path=MODEL_SAVE_PATH, graph_path=GRAPH_DATA_PATH, mapping_dir=MAPPING_DIR):
        print("Initializing FraudPredictor...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load node-to-index mappings
        self.cust_map = self._load_mapping(os.path.join(mapping_dir, "customer_mapping.json"))
        self.acc_map = self._load_mapping(os.path.join(mapping_dir, "account_mapping.json"))
        self.merch_map = self._load_mapping(os.path.join(mapping_dir, "merchant_mapping.json"))
        self.dev_map = self._load_mapping(os.path.join(mapping_dir, "device_mapping.json"))
        
        # Load graph
        self.data = torch.load(graph_path, weights_only=False)
        
        # Load checkpoint and model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        hidden_channels = checkpoint.get("hidden_channels", 64)
        edge_dim = checkpoint.get("edge_dim", 4)
        
        self.model = HeteroGraphSAGE(hidden_channels=hidden_channels, edge_dim=edge_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        print("FraudPredictor successfully loaded and ready for inference.")

    def _load_mapping(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mapping file not found at: {filepath}")
        with open(filepath, "r") as f:
            return json.load(f)

    def predict_p2p_transfer(self, sender_acc, receiver_acc, amount, txns_1h, txns_24h, geo_anomaly, device_id=None):
        """
        Runs GNN inference to evaluate the fraud risk of a peer-to-peer money transfer.
        """
        # Validate sender and receiver accounts in mapping
        if sender_acc not in self.acc_map:
            return {"status": "ERROR", "message": f"Sender account {sender_acc} not found in database mappings."}
        if receiver_acc not in self.acc_map:
            return {"status": "ERROR", "message": f"Receiver account {receiver_acc} not found in database mappings."}
            
        sender_idx = self.acc_map[sender_acc]
        receiver_idx = self.acc_map[receiver_acc]
        
        # Build transaction edge feature vector
        amount_log = np.log1p(amount)
        edge_attr = torch.tensor([[amount_log, float(txns_1h), float(txns_24h), float(geo_anomaly)]], dtype=torch.float, device=self.device)
        
        # Candidate edge connection
        candidate_edge = torch.tensor([[sender_idx], [receiver_idx]], dtype=torch.long, device=self.device)
        
        # If device is provided, dynamically update the USES edge index for GNN context
        x_dict = {k: v.to(self.device) for k, v in self.data.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}
        
        if device_id and device_id in self.dev_map:
            dev_idx = self.dev_map[device_id]
            uses_edges = edge_index_dict[("account", "uses", "device")]
            new_uses_edge = torch.tensor([[sender_idx], [dev_idx]], dtype=torch.long, device=self.device)
            edge_index_dict[("account", "uses", "device")] = torch.cat([uses_edges, new_uses_edge], dim=-1)
            
        with torch.no_grad():
            # Get updated structural GNN representations
            h_dict = self.model(x_dict, edge_index_dict)
            
            # Predict probability of fraud
            logits = self.model.classify_performs(h_dict, candidate_edge, edge_attr)
            probability = torch.sigmoid(logits).item()
            
        is_fraud = probability >= 0.5
        
        return {
            "transaction_type": "P2P_TRANSFER",
            "sender_account": sender_acc,
            "receiver_account": receiver_acc,
            "amount": amount,
            "fraud_probability": round(probability, 4),
            "fraud_prediction": int(is_fraud),
            "recommendation": "DENY / INVESTIGATE" if is_fraud else "APPROVE"
        }

    def predict_merchant_payment(self, sender_acc, merchant_id, amount, txns_1h, txns_24h, geo_anomaly, device_id=None):
        """
        Runs GNN inference to evaluate the fraud risk of a merchant payment.
        """
        if sender_acc not in self.acc_map:
            return {"status": "ERROR", "message": f"Sender account {sender_acc} not found in database mappings."}
        if merchant_id not in self.merch_map:
            return {"status": "ERROR", "message": f"Merchant {merchant_id} not found in database mappings."}
            
        sender_idx = self.acc_map[sender_acc]
        merchant_idx = self.merch_map[merchant_id]
        
        amount_log = np.log1p(amount)
        edge_attr = torch.tensor([[amount_log, float(txns_1h), float(txns_24h), float(geo_anomaly)]], dtype=torch.float, device=self.device)
        
        candidate_edge = torch.tensor([[sender_idx], [merchant_idx]], dtype=torch.long, device=self.device)
        
        x_dict = {k: v.to(self.device) for k, v in self.data.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}
        
        if device_id and device_id in self.dev_map:
            dev_idx = self.dev_map[device_id]
            uses_edges = edge_index_dict[("account", "uses", "device")]
            new_uses_edge = torch.tensor([[sender_idx], [dev_idx]], dtype=torch.long, device=self.device)
            edge_index_dict[("account", "uses", "device")] = torch.cat([uses_edges, new_uses_edge], dim=-1)
            
        with torch.no_grad():
            h_dict = self.model(x_dict, edge_index_dict)
            logits = self.model.classify_paid_to(h_dict, candidate_edge, edge_attr)
            probability = torch.sigmoid(logits).item()
            
        is_fraud = probability >= 0.5
        
        return {
            "transaction_type": "MERCHANT_PAYMENT",
            "sender_account": sender_acc,
            "merchant_id": merchant_id,
            "amount": amount,
            "fraud_probability": round(probability, 4),
            "fraud_prediction": int(is_fraud),
            "recommendation": "DENY / INVESTIGATE" if is_fraud else "APPROVE"
        }

if __name__ == "__main__":
    # Self-test block to demonstrate inference function
    print("Running predictor self-test...")
    try:
        predictor = FraudPredictor()
        
        # Test 1: Mock P2P Transfer (Genuine-like profile)
        p2p_res = predictor.predict_p2p_transfer(
            sender_acc="ACC_1000001",
            receiver_acc="ACC_1000005",
            amount=150.0,
            txns_1h=0,
            txns_24h=1,
            geo_anomaly=0,
            device_id="DEV_10000"
        )
        print("\nTest P2P Transfer Result:")
        print(json.dumps(p2p_res, indent=4))
        
        # Test 2: Mock Merchant Payment (High-risk-like profile, e.g. Night hour, new hacker device, high velocity)
        merch_res = predictor.predict_merchant_payment(
            sender_acc="ACC_1001250",
            merchant_id="M_1005",
            amount=4500.0,
            txns_1h=5,
            txns_24h=12,
            geo_anomaly=1,
            device_id="DEV_11100"
        )
        print("\nTest Merchant Payment Result:")
        print(json.dumps(merch_res, indent=4))
        
    except Exception as e:
        print(f"Self-test failed due to: {str(e)}")

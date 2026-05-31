import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append("d:/fraud_detection")

from ml.models.graphsage_model import HeteroGraphSAGE
from ml.inference.predict import FraudPredictor

MODEL_PATH = "d:/fraud_detection/ml/saved_models/graphsage_model.pth"
GRAPH_PATH = "d:/fraud_detection/data/processed/graph_data.pt"

def run_validation_pipeline():
    print("======================================================================")
    print("           GRAPHSHEILD AI - MODEL PIPELINE VALIDATION SUITE            ")
    print("======================================================================")
    
    validation_status = "SUCCESS"
    errors = []
    
    # Requirement 1: Verify .pth file exists
    print("Check 1: Verifying model checkpoint file existence...")
    if os.path.exists(MODEL_PATH):
        print(f"PASS: Checkpoint file found at: {MODEL_PATH}")
    else:
        print(f"FAIL: Checkpoint file NOT found at: {MODEL_PATH}")
        validation_status = "FAILURE"
        errors.append("Model checkpoint file (.pth) does not exist.")
        return validation_status, errors
        
    # Requirement 2: Verify state_dict loads correctly
    print("\nCheck 2: Loading checkpoint parameters...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint, "Missing model_state_dict in checkpoint."
        assert "hidden_channels" in checkpoint, "Missing hidden_channels configuration."
        assert "edge_dim" in checkpoint, "Missing edge_dim configuration."
        print("PASS: Checkpoint loaded successfully with all required key fields.")
    except Exception as e:
        print(f"FAIL: Failed to load checkpoint. Error: {str(e)}")
        validation_status = "FAILURE"
        errors.append(f"Failed to load checkpoint file: {str(e)}")
        return validation_status, errors
        
    # Requirement 3: Verify architecture matches saved weights
    print("\nCheck 3: Instantiating architecture and verifying weights shape...")
    try:
        hidden_channels = checkpoint["hidden_channels"]
        edge_dim = checkpoint["edge_dim"]
        model = HeteroGraphSAGE(hidden_channels=hidden_channels, edge_dim=edge_dim)
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"PASS: Model instantiated matching weights parameters (hidden_channels={hidden_channels}, edge_dim={edge_dim}).")
        print("PASS: State dict keys successfully matched with model architecture parameters.")
    except Exception as e:
        print(f"FAIL: State dict load mismatch. Error: {str(e)}")
        validation_status = "FAILURE"
        errors.append(f"Model state dict keys or shape mismatch: {str(e)}")
        
    # Requirement 4 & 5: Verify device selection (CPU/GPU)
    print("\nCheck 4: Verifying device mapping selection...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   System Active Device: {device.type.upper()}")
        model.to(device)
        print(f"PASS: Successfully transferred model layers to device memory: {device}")
    except Exception as e:
        print(f"FAIL: Device transfer failed. Error: {str(e)}")
        validation_status = "FAILURE"
        errors.append(f"Device mapping allocation failed: {str(e)}")
        
    # Requirement 6: Verify inference works and probabilities are valid
    print("\nCheck 5: Performing inference self-tests and probability bounds checks...")
    try:
        predictor = FraudPredictor()
        
        # Test Genuine Case
        p2p_res = predictor.predict_p2p_transfer(
            sender_acc="ACC_1000001",
            receiver_acc="ACC_1000005",
            amount=120.0,
            txns_1h=0,
            txns_24h=1,
            geo_anomaly=0
        )
        
        # Test Fraud Case
        merch_res = predictor.predict_merchant_payment(
            sender_acc="ACC_1001250",
            merchant_id="M_1005",
            amount=4900.0,
            txns_1h=5,
            txns_24h=15,
            geo_anomaly=1
        )
        
        # Validate output structures
        assert p2p_res["transaction_type"] == "P2P_TRANSFER"
        assert merch_res["transaction_type"] == "MERCHANT_PAYMENT"
        
        # Probability bounds checks
        p2p_prob = p2p_res["fraud_probability"]
        merch_prob = merch_res["fraud_probability"]
        
        print(f"   P2P Genuine Fraud Probability: {p2p_prob:.4f}")
        print(f"   Merchant Fraud Probability: {merch_prob:.4f}")
        
        assert 0.0 <= p2p_prob <= 1.0, f"P2P Probability {p2p_prob} out of bounds [0.0, 1.0]"
        assert 0.0 <= merch_prob <= 1.0, f"Merchant Probability {merch_prob} out of bounds [0.0, 1.0]"
        
        print("PASS: Inference executed successfully.")
        print("PASS: Fraud probability values verified within [0.0, 1.0] mathematical bounds.")
        
    except Exception as e:
        print(f"FAIL: Inference validation failed. Error: {str(e)}")
        validation_status = "FAILURE"
        errors.append(f"Inference execution failed: {str(e)}")
        
    print("\n======================================================================")
    print(f"STATUS: Validation Pipeline completed with status: {validation_status}")
    if errors:
        print(f"   Errors found: {errors}")
    print("======================================================================")
    
    return validation_status, errors

if __name__ == "__main__":
    run_validation_pipeline()

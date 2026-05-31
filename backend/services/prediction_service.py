import os
import sys

# Ensure project root is in system path for GNN imports
sys.path.append("d:/fraud_detection")

from ml.inference.predict import FraudPredictor

class PredictionService:
    """
    Service layer for GNN model loading and live scoring.
    Implements a Singleton pattern to prevent reloading GNN weights on each API call.
    """
    _predictor = None

    @classmethod
    def get_predictor(cls) -> FraudPredictor:
        if cls._predictor is None:
            print("Singleton instance for FraudPredictor not found. Loading model weights into memory...")
            cls._predictor = FraudPredictor()
        return cls._predictor

    @classmethod
    def score_transaction(cls, payload: dict) -> dict:
        predictor = cls.get_predictor()
        
        sender = payload["sender_account_id"]
        receiver = payload.get("receiver_account_id")
        merchant = payload.get("merchant_id")
        amount = payload["amount"]
        device = payload.get("device_id")
        txns_1h = payload.get("txns_last_1h", 0)
        txns_24h = payload.get("txns_last_24h", 0)
        geo_anomaly = payload.get("geo_anomaly", 0)
        
        # Branch based on destination node type (P2P vs Merchant payment)
        if receiver and receiver != "":
            return predictor.predict_p2p_transfer(
                sender_acc=sender,
                receiver_acc=receiver,
                amount=amount,
                txns_1h=txns_1h,
                txns_24h=txns_24h,
                geo_anomaly=geo_anomaly,
                device_id=device
            )
        elif merchant and merchant != "":
            return predictor.predict_merchant_payment(
                sender_acc=sender,
                merchant_id=merchant,
                amount=amount,
                txns_1h=txns_1h,
                txns_24h=txns_24h,
                geo_anomaly=geo_anomaly,
                device_id=device
            )
        else:
            return {
                "status": "ERROR",
                "message": "Validation failed: Transaction must specify either a receiver_account_id or a merchant_id."
            }

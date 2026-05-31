from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class TransactionRequest(BaseModel):
    sender_account_id: str = Field(..., example="ACC_1001250")
    receiver_account_id: Optional[str] = Field(None, example="ACC_1000005")
    merchant_id: Optional[str] = Field(None, example="M_1005")
    amount: float = Field(..., gt=0.0, example=4500.00)
    device_id: Optional[str] = Field(None, example="DEV_11100")
    txns_last_1h: int = Field(0, ge=0, example=5)
    txns_last_24h: int = Field(0, ge=0, example=12)
    geo_anomaly: int = Field(0, ge=0, le=1, example=1)

class TransactionResponse(BaseModel):
    transaction_type: str
    sender_account: str
    receiver_account: Optional[str] = None
    merchant_id: Optional[str] = None
    amount: float
    fraud_probability: float
    fraud_prediction: int
    recommendation: str

class TrainResponse(BaseModel):
    status: str
    message: str
    job_id: str

class ExplainRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_F_60002183")

class FeatureContribution(BaseModel):
    feature_value: float
    shap_value: float

class ExplainResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    base_value: float
    explanations: Dict[str, FeatureContribution]
    chart_url: str

class GraphStatsResponse(BaseModel):
    node_counts: Dict[str, int]
    edge_counts: Dict[str, int]
    density: float

class AlertItem(BaseModel):
    transaction_id: str
    sender_account: str
    receiver_or_merchant: str
    amount: float
    fraud_probability: float
    timestamp: str

class DashboardSummaryResponse(BaseModel):
    total_transactions_scanned: int
    total_alerts_triggered: int
    alert_rate: float
    avg_risk_score: float
    recent_alerts: List[AlertItem]

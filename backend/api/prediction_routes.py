import time
import subprocess
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from backend.schemas.transaction_schema import TransactionRequest, TransactionResponse, TrainResponse
from backend.services.prediction_service import PredictionService

router = APIRouter(prefix="/prediction", tags=["prediction"])

def run_gnn_training():
    """Runs GNN training script as a background process."""
    print("Background task: Starting GNN model training optimization...")
    try:
        # Run GNN training script in the workspace root
        res = subprocess.run(["python", "ml/training/train_graphsage.py"], check=True, capture_output=True, text=True)
        print("Background task: GNN model training completed successfully.")
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Background task ERROR: GNN model training failed with exit code {e.returncode}")
        print(e.stderr)
    except Exception as e:
        print(f"Background task ERROR: {str(e)}")

@router.post("/predict", response_model=TransactionResponse)
def predict_fraud(request: TransactionRequest):
    """
    Evaluates risk and scores a transaction edge for fraud using GraphSAGE GNN.
    """
    try:
        payload = request.dict()
        result = PredictionService.score_transaction(payload)
        
        if result.get("status") == "ERROR":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model(background_tasks: BackgroundTasks):
    """
    Triggers asynchronous GNN training in the background. Returns a Job ID immediately.
    """
    job_id = f"job_gnn_{int(time.time())}"
    background_tasks.add_task(run_gnn_training)
    
    return {
        "status": "ACCEPTED",
        "message": "GNN optimization training job successfully spawned in background.",
        "job_id": job_id
    }

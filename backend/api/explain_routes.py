from fastapi import APIRouter, HTTPException, status
from backend.schemas.transaction_schema import ExplainRequest, ExplainResponse
from backend.services.explainability_service import ExplainabilityService

router = APIRouter(prefix="/explainability", tags=["explainability"])

@router.post("/explain", response_model=ExplainResponse)
def explain_prediction(request: ExplainRequest):
    """
    Computes local SHAP values and waterfall explanation metrics for a transaction.
    """
    try:
        result = ExplainabilityService.explain_transaction(request.transaction_id)
        return result
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

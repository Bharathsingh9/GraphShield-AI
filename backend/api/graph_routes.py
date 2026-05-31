from fastapi import APIRouter, HTTPException, status
from backend.schemas.transaction_schema import GraphStatsResponse
from backend.services.graph_service import GraphService

router = APIRouter(prefix="/graph", tags=["graph"])

@router.get("/stats", response_model=GraphStatsResponse)
def get_graph_statistics():
    """
    Returns global network stats including node counts, edge counts, and graph density.
    """
    try:
        return GraphService.get_graph_stats()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/neighbors/{node_type}/{node_id}")
def get_node_neighbors(node_type: str, node_id: str):
    """
    Retrieves local neighborhood details (degrees, relation names, neighboring IDs).
    """
    try:
        result = GraphService.get_node_neighbors(node_type, node_id)
        if "status" in result and result["status"] == "ERROR":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

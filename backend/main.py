import os
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add root directory to path for GNN imports
import sys
sys.path.append("d:/fraud_detection")

from backend.api import prediction_routes, explain_routes, graph_routes, dashboard_routes
from backend.services.prediction_service import PredictionService
from backend.services.explainability_service import ExplainabilityService

# Initialize Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("backend")

# Initialize FastAPI App
app = FastAPI(
    title="GraphShield AI - Banking Fraud Detection API",
    description="FastAPI Backend for real-time GNN-based transaction risk monitoring and SHAP explanations.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing) for Dashboard UI Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits access from all ports (useful for dev and dev servers)
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

# Serve generated explanation charts and reports statically
DOCS_DIR = "d:/fraud_detection/docs"
os.makedirs(DOCS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=DOCS_DIR), name="static")

# Register API Routers
app.include_router(prediction_routes.router, prefix="/api/v1")
app.include_router(explain_routes.router, prefix="/api/v1")
app.include_router(graph_routes.router, prefix="/api/v1")
app.include_router(dashboard_routes.router, prefix="/api/v1")

# Global Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the system administrator."}
    )

# Pre-load GNN Model and Explainability Cache on App Startup
@app.on_event("startup")
def startup_event():
    logger.info("Starting GraphShield AI Backend...")
    try:
        # Pre-load predictor weights
        PredictionService.get_predictor()
        # Pre-load explainer node embeddings cache
        ExplainabilityService.initialize()
        logger.info("GNN Model and structural embeddings cached successfully on startup.")
    except Exception as e:
        logger.error(f"FATAL: Model loading failed during startup: {str(e)}", exc_info=True)

@app.get("/")
def read_root():
    return {
        "status": "HEALTHY",
        "service": "GraphShield AI Backend",
        "documentation": "/docs"
    }

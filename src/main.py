import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from src.api.routes import router
from config.settings import settings

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_pipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")
    try:
        from src.api.routes import db_client, dense_encoder, sparse_encoder
        logger.info(f"Qdrant collection '{settings.COLLECTION_NAME}' is ready.")
        logger.info(f"Dense model '{settings.DENSE_MODEL_NAME}' initialized.")
        logger.info(f"Sparse model '{settings.SPARSE_MODEL_NAME}' initialized.")
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready Hybrid Search (Dense + Sparse + RRF) RAG Pipeline API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal Server Error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred."
        }
    )


# System Health / Status info endpoint
@app.get("/api/v1/status", tags=["System"])
def system_status():
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "api_prefix": "/api/v1"
    }


# Mount built frontend single-page application if dist directory exists
frontend_dist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist_path):
    logger.info(f"Serving built frontend from {frontend_dist_path}")
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
else:
    @app.get("/", tags=["System"])
    def root():
        return {
            "status": "online",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "documentation": "/docs",
            "api_prefix": "/api/v1"
        }
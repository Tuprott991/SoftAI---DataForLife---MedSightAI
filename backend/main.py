"""
MedSight AI Backend - Main Application
FastAPI application for MedSight AI medical imaging analysis platform
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.config.database import init_db
from app.api.v1 import api_router
from app.middleware import setup_cors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    Startup and shutdown logic
    """
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize database tables
    print("Initializing database...")
    try:
        init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
    
    # Note: Zilliz Cloud connection is managed per-request in zilliz_service
    # No persistent connection needed for serverless REST API
    print("✓ Zilliz Cloud configured (serverless)")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Medical imaging AI analysis platform with explainable AI and education features",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup CORS
setup_cors(app)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


# Root endpoint
@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "api": settings.API_V1_STR
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

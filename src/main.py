from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import traceback

from src.config.settings import settings
from src.routes import router as api_routes
from src.config.database import engine, Base

# Import all models to ensure they're registered with Base
from src.models.instagram_post_model import InstagramPost
from src.models.product_model import Product
from src.models.product_item_model import ProductItem

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Backend API for Vibe Search"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        async with engine.begin() as conn:
            # Create all tables defined in your models
            await conn.run_sync(Base.metadata.create_all)
            print("[Startup] Database tables created successfully")
            
            # Print created tables for verification
            table_names = list(Base.metadata.tables.keys())
            print(f"[Startup] Created tables: {table_names}")
            
    except Exception as e:
        print(f"[Startup] Error creating database tables: {e}")
        # Don't raise here to allow app to start even if DB fails
        # raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    await engine.dispose()
    print("[Shutdown] Database engine disposed")

# Root healthcheck endpoint
@app.get("/")
async def root():
    return {
        "status": "OK", 
        "test": "success",
        "message": f"{settings.APP_NAME} is running",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database_url": settings.DATABASE_URL,
        "tables": list(Base.metadata.tables.keys())
    }

# Routes setup
app.include_router(api_routes, prefix="/api/v1")
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import traceback

from src.config.settings import settings
from src.routes import router as api_routes

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Backend API for Vibe Search"
)

# CORS configuration to allow all traffic (similar to your Node.js setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Set to False when allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)


# Root healthcheck endpoint (equivalent to your "/" endpoint)
@app.get("/")
async def root():
    return {
        "status": "OK", 
        "test": "success",
        "message": f"{settings.APP_NAME} is running",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database url": settings.DATABASE_URL
    }


# Routes setup
app.include_router(api_routes, prefix="/api/v1")
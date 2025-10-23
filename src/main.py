from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from sqlalchemy import text
import traceback
import logging  # Add this import

from src.config.settings import settings
from src.routes import router as api_routes
from src.config.database import engine, Base

# Import all models to ensure they're registered with Base
from src.models.instagram_post_model import InstagramPost
from src.models.product_model import Product
from src.models.product_item_model import ProductItem
from src.models.store_item_model import StoreItem  # Add this import

# Configure logging - Add this section
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set SQLAlchemy engine logging to WARNING to reduce query logs
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Backend API for Vibe Search"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        async with engine.begin() as conn:
            # First, try to create the pgvector extension
            try:
                print("[Startup] Creating pgvector extension...")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                print("[Startup] ✓ pgvector extension installed/verified")
            except Exception as e:
                print(f"[Startup] ✗ Could not install pgvector extension: {e}")
                print("[Startup] The pgvector/pgvector Docker image should have this pre-installed")
                raise Exception("pgvector extension not available")
            
            # Create all tables defined in your models
            await conn.run_sync(Base.metadata.create_all)
            print("[Startup] Database tables created successfully")
            
            # Print created tables for verification
            table_names = list(Base.metadata.tables.keys())
            print(f"[Startup] Created tables: {table_names}")
            
            # Create vector indexes for store_items if table exists
            try:
                print("[Startup] Creating vector indexes for store_items...")
                
                # Create regular indexes one by one
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_store_items_sku_id ON store_items(sku_id);"))
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_store_items_category ON store_items(category);"))
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_store_items_brand ON store_items(brand_name);"))
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_store_items_product_type ON store_items(product_type);"))
                
                # Create vector indexes one by one
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_store_items_textual_embedding 
                        ON store_items USING hnsw (textual_embedding vector_cosine_ops);
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_store_items_visual_embedding 
                        ON store_items USING hnsw (visual_embedding vector_cosine_ops);
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_store_items_multimodal_embedding 
                        ON store_items USING hnsw (multimodal_embedding vector_cosine_ops);
                """))
                
                print("[Startup] ✓ Vector indexes created for store_items")
            except Exception as e:
                print(f"[Startup] Note: Vector indexes might already exist or table not created yet: {e}")
                print(traceback.format_exc())
            
    except Exception as e:
        print(f"[Startup] Error creating database tables: {e}")
        print(traceback.format_exc())
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
        "database_url": settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "***",
        "tables": list(Base.metadata.tables.keys())
    }

# Routes setup
app.include_router(api_routes, prefix="/api/v1")
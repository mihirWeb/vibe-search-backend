"""
Database migration script for Product and ProductItem tables.
This script creates the tables with pgvector support for embeddings.

Run this script after ensuring pgvector extension is installed in PostgreSQL:
    CREATE EXTENSION IF NOT EXISTS vector;
"""

import asyncio
from sqlalchemy import text
from src.config.database import engine, Base
from src.models.product_model import Product
from src.models.product_item_model import ProductItem

async def create_tables():
    """Create product and product_items tables"""
    async with engine.begin() as conn:
        print("[Migration] Creating pgvector extension...")
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            print("[Migration] pgvector extension created successfully")
        except Exception as e:
            print(f"[Migration] Warning: Could not create pgvector extension: {str(e)}")
            print("[Migration] Please ensure pgvector is installed in PostgreSQL")
        
        print("[Migration] Dropping existing tables if they exist...")
        # Drop tables in correct order (items first, then products)
        await conn.execute(text("DROP TABLE IF EXISTS product_items CASCADE;"))
        await conn.execute(text("DROP TABLE IF EXISTS products CASCADE;"))
        
        print("[Migration] Creating new tables...")
        await conn.run_sync(Base.metadata.create_all)
        print("[Migration] Tables created successfully")

async def verify_tables():
    """Verify that tables were created successfully"""
    async with engine.begin() as conn:
        print("[Migration] Verifying tables...")
        
        # Check products table
        result = await conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'products';"
        ))
        if result.fetchone():
            print("[Migration] ✓ products table exists")
        else:
            print("[Migration] ✗ products table not found")
        
        # Check product_items table
        result = await conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'product_items';"
        ))
        if result.fetchone():
            print("[Migration] ✓ product_items table exists")
        else:
            print("[Migration] ✗ product_items table not found")
        
        # Check vector extension
        result = await conn.execute(text(
            "SELECT * FROM pg_extension WHERE extname = 'vector';"
        ))
        if result.fetchone():
            print("[Migration] ✓ vector extension is installed")
        else:
            print("[Migration] ✗ vector extension not found")
            print("[Migration] Please install pgvector: CREATE EXTENSION vector;")

async def main():
    """Main migration function"""
    print("[Migration] Starting product tables migration...")
    print("[Migration] This will drop and recreate products and product_items tables")
    
    try:
        await create_tables()
        await verify_tables()
        print("[Migration] Migration completed successfully!")
        print("\n[Migration] Tables created:")
        print("  - products (with vector embeddings)")
        print("  - product_items (with vector embeddings)")
        print("\n[Migration] Next steps:")
        print("  1. Ensure pgvector extension is installed")
        print("  2. Start extracting products from Instagram posts")
        print("  3. Use the /products/extract endpoint to process posts")
    except Exception as e:
        print(f"[Migration] Error during migration: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())

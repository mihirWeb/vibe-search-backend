"""
Product repository for database operations on products and product items.
Handles all database interactions following the async pattern.
"""

from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError
from datetime import datetime

from src.models.product_model import Product
from src.models.product_item_model import ProductItem


class ProductRepository:
    """Repository for Product and ProductItem database operations"""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        print("[Product Repository] Initialized")
    
    async def create_product(self, product_data: Dict) -> Product:
        """Create a new product with its items in the database"""
        try:
            print(f"[Product Repository] Creating product: {product_data['name']}")
            
            # Create product record - note: using meta_info instead of metadata
            product = Product(
                name=product_data["name"],
                description=product_data["description"],
                image_url=product_data["image_url"],
                source_url=product_data["source_url"],
                brand=product_data["brand"],
                category=product_data["category"],
                style=product_data["style"],
                colors=product_data["colors"],
                caption=product_data["caption"],
                embedding=product_data["embedding"],
                text_embedding=product_data["text_embedding"],
                meta_info=product_data.get("metadata", {})  # Changed from metadata to meta_info
            )
            
            self.db_session.add(product)
            await self.db_session.flush()  # Flush to get the product ID
            
            print(f"[Product Repository] Product created with ID: {product.id}")
            
            # Save items to database
            for item_data in product_data["items"]:
                item = ProductItem(
                    product_id=product.id,
                    name=item_data["name"],
                    brand=item_data["brand"],
                    category=item_data["category"],
                    style=item_data["style"],
                    colors=item_data["colors"],
                    product_type=item_data["product_type"],
                    description=item_data["description"],
                    visual_features=item_data["visual_features"],
                    embedding=item_data["embedding"],
                    text_embedding=item_data["text_embedding"],
                    bounding_box=item_data["bounding_box"],
                    confidence_score=item_data["confidence_score"],
                    meta_info=item_data.get("metadata", {})  # Changed from metadata to meta_info
                )
                self.db_session.add(item)
            
            # Commit changes
            await self.db_session.commit()
            await self.db_session.refresh(product)
            
            print(f"[Product Repository] Product saved with {len(product_data['items'])} items")
            return product
            
        except IntegrityError as e:
            await self.db_session.rollback()
            print(f"[Product Repository] IntegrityError: {str(e)}")
            raise ValueError(f"Product creation failed due to integrity constraint: {str(e)}")
        except Exception as e:
            await self.db_session.rollback()
            print(f"[Product Repository] Error creating product: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise e
    
    # ...existing code...
    
    async def get_product_by_id(self, product_id: int) -> Optional[Product]:
        """Get a product by ID"""
        try:
            print(f"[Product Repository] Fetching product with ID: {product_id}")
            
            stmt = select(Product).where(Product.id == product_id)
            result = await self.db_session.execute(stmt)
            product = result.scalar_one_or_none()
            
            if product:
                print(f"[Product Repository] Product found: {product.name}")
            else:
                print(f"[Product Repository] Product not found with ID: {product_id}")
            
            return product
            
        except Exception as e:
            print(f"[Product Repository] Error fetching product: {str(e)}")
            raise e
    
    async def get_products_by_brand(self, brand: str, limit: int = 50) -> List[Product]:
        """Get products by brand"""
        try:
            print(f"[Product Repository] Fetching products for brand: {brand}")
            
            stmt = select(Product).where(Product.brand == brand).order_by(desc(Product.created_at)).limit(limit)
            result = await self.db_session.execute(stmt)
            products = result.scalars().all()
            
            print(f"[Product Repository] Found {len(products)} products for brand: {brand}")
            return list(products)
            
        except Exception as e:
            print(f"[Product Repository] Error fetching products by brand: {str(e)}")
            raise e
    
    async def get_recent_products(self, limit: int = 50) -> List[Product]:
        """Get recent products"""
        try:
            print(f"[Product Repository] Fetching {limit} recent products")
            
            stmt = select(Product).order_by(desc(Product.created_at)).limit(limit)
            result = await self.db_session.execute(stmt)
            products = result.scalars().all()
            
            print(f"[Product Repository] Found {len(products)} recent products")
            return list(products)
            
        except Exception as e:
            print(f"[Product Repository] Error fetching recent products: {str(e)}")
            raise e
    
    async def get_product_items_by_product_id(self, product_id: int) -> List[ProductItem]:
        """Get all items for a specific product"""
        try:
            print(f"[Product Repository] Fetching items for product ID: {product_id}")
            
            stmt = select(ProductItem).where(ProductItem.product_id == product_id)
            result = await self.db_session.execute(stmt)
            items = result.scalars().all()
            
            print(f"[Product Repository] Found {len(items)} items for product ID: {product_id}")
            return list(items)
            
        except Exception as e:
            print(f"[Product Repository] Error fetching product items: {str(e)}")
            raise e
    
    async def delete_product(self, product_id: int) -> bool:
        """Delete a product and its items"""
        try:
            print(f"[Product Repository] Deleting product with ID: {product_id}")
            
            product = await self.get_product_by_id(product_id)
            if not product:
                print(f"[Product Repository] Product not found with ID: {product_id}")
                return False
            
            await self.db_session.delete(product)
            await self.db_session.commit()
            
            print(f"[Product Repository] Product deleted: {product_id}")
            return True
            
        except Exception as e:
            await self.db_session.rollback()
            print(f"[Product Repository] Error deleting product: {str(e)}")
            raise e
    
    async def batch_create_products(self, products_data: List[Dict]) -> List[Product]:
        """Create multiple products in batch"""
        try:
            print(f"[Product Repository] Batch creating {len(products_data)} products")
            
            created_products = []
            
            for product_data in products_data:
                try:
                    product = await self.create_product(product_data)
                    created_products.append(product)
                except Exception as e:
                    print(f"[Product Repository] Error creating product in batch: {str(e)}")
                    continue
            
            print(f"[Product Repository] Batch created {len(created_products)} products")
            return created_products
            
        except Exception as e:
            print(f"[Product Repository] Error in batch create: {str(e)}")
            raise e
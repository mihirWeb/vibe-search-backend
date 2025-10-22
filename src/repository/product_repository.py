"""
Product repository for database operations on products and product items.
Handles all database interactions following the async pattern.
"""

from typing import List, Optional, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, asc, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
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

            # Create product record
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
                meta_info=product_data.get("metadata", {})
            )

            self.db_session.add(product)
            await self.db_session.flush()

            print(f"[Product Repository] Product created with ID: {product.id}")

            # Create items with new fields
            items = []
            for item_data in product_data["items"]:
                item = ProductItem(
                    product_id=product.id,
                    name=item_data["name"],
                    brand=item_data.get("brand"),
                    category=item_data["category"],
                    sub_category=item_data.get("sub_category"),
                    product_type=item_data["product_type"],
                    gender=item_data.get("gender"),
                    style=item_data["style"],
                    colors=item_data["colors"],
                    description=item_data["description"],
                    visual_features=item_data["visual_features"],
                    embedding=item_data["embedding"],
                    text_embedding=item_data["text_embedding"],
                    bounding_box=item_data["bounding_box"],
                    confidence_score=item_data["confidence_score"],
                    meta_info=item_data.get("metadata", {})
                )
                items.append(item)
                self.db_session.add(item)

            await self.db_session.commit()
            await self.db_session.refresh(product)

            print(f"[Product Repository] Product saved with {len(items)} items")
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

    async def get_product_by_id(self, product_id: int) -> Optional[Product]:
        """Get a product by ID with items eagerly loaded"""
        try:
            print(f"[Product Repository] Fetching product with ID: {product_id}")

            stmt = (
                select(Product)
                .options(selectinload(Product.items))
                .where(Product.id == product_id)
            )
            result = await self.db_session.execute(stmt)
            product = result.scalar_one_or_none()

            if product:
                print(f"[Product Repository] Product found: {product.name} with {len(product.items)} items")
            else:
                print(f"[Product Repository] Product not found with ID: {product_id}")

            return product

        except Exception as e:
            print(f"[Product Repository] Error fetching product: {str(e)}")
            raise e

    async def get_products_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        brand: Optional[str] = None,
        category: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[Product], int]:
        """
        Get paginated products with optional filters.
        Returns tuple of (products, total_count)
        """
        try:
            print(f"[Product Repository] Fetching paginated products - Page: {page}, Size: {page_size}")

            # Base query with eager loading of items
            query = select(Product).options(selectinload(Product.items))

            # Apply filters
            if brand:
                query = query.where(Product.brand == brand)
                print(f"[Product Repository] Filtering by brand: {brand}")

            if category:
                query = query.where(Product.category == category)
                print(f"[Product Repository] Filtering by category: {category}")

            # Get total count
            count_query = select(func.count()).select_from(Product)
            if brand:
                count_query = count_query.where(Product.brand == brand)
            if category:
                count_query = count_query.where(Product.category == category)
            
            count_result = await self.db_session.execute(count_query)
            total_count = count_result.scalar()

            # Apply sorting
            sort_column = getattr(Product, sort_by, Product.created_at)
            if sort_order.lower() == "asc":
                query = query.order_by(asc(sort_column))
            else:
                query = query.order_by(desc(sort_column))

            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)

            # Execute query
            result = await self.db_session.execute(query)
            products = result.scalars().all()

            print(f"[Product Repository] Found {len(products)} products (Total: {total_count})")
            return list(products), total_count

        except Exception as e:
            print(f"[Product Repository] Error fetching paginated products: {str(e)}")
            raise e

    async def get_products_by_brand(self, brand: str, limit: int = 50) -> List[Product]:
        """Get products by brand with items eagerly loaded"""
        try:
            print(f"[Product Repository] Fetching products for brand: {brand}")

            stmt = (
                select(Product)
                .options(selectinload(Product.items))
                .where(Product.brand == brand)
                .order_by(desc(Product.created_at))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            products = result.scalars().all()

            print(f"[Product Repository] Found {len(products)} products for brand: {brand}")
            return list(products)

        except Exception as e:
            print(f"[Product Repository] Error fetching products by brand: {str(e)}")
            raise e

    async def get_recent_products(self, limit: int = 50) -> List[Product]:
        """Get recent products with items eagerly loaded"""
        try:
            print(f"[Product Repository] Fetching {limit} recent products")

            stmt = (
                select(Product)
                .options(selectinload(Product.items))
                .order_by(desc(Product.created_at))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            products = result.scalars().all()

            print(f"[Product Repository] Found {len(products)} recent products")
            return list(products)

        except Exception as e:
            print(f"[Product Repository] Error fetching recent products: {str(e)}")
            raise e

    async def get_product_items_by_product_id(self, product_id: int) -> List[ProductItem]:
        """Get all items for a specific product in db"""
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
        

    async def check_post_extracted(self, instagram_post_id: str) -> bool:
        """Check if an Instagram post has already been extracted"""
        try:
            print(f"[Product Repository] Checking if post {instagram_post_id} is extracted")
            
            stmt = select(Product).where(
                Product.meta_info['post_id'].astext == instagram_post_id
            ).limit(1)
            
            result = await self.db_session.execute(stmt)
            product = result.scalar_one_or_none()
            
            is_extracted = product is not None
            print(f"[Product Repository] Post {instagram_post_id} extracted: {is_extracted}")
            
            return is_extracted

        except Exception as e:
            print(f"[Product Repository] Error checking post extraction: {str(e)}")
            return False

    async def get_product_by_post_id(self, instagram_post_id: str) -> Optional[Product]:
        """Get product by Instagram post ID"""
        try:
            print(f"[Product Repository] Fetching product for post ID: {instagram_post_id}")
            
            stmt = select(Product).where(
                Product.meta_info['post_id'].astext == instagram_post_id
            ).options(selectinload(Product.items))
            
            result = await self.db_session.execute(stmt)
            product = result.scalar_one_or_none()
            
            if product:
                print(f"[Product Repository] Found product {product.id} for post {instagram_post_id}")
            else:
                print(f"[Product Repository] No product found for post {instagram_post_id}")
            
            return product

        except Exception as e:
            print(f"[Product Repository] Error fetching product by post ID: {str(e)}")
            raise e
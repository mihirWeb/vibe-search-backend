"""
Store item repository for database operations.
Handles all database interactions for store items.
"""

from typing import List, Optional, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from sqlalchemy.exc import IntegrityError
import math

from src.models.store_item_model import StoreItem


class StoreItemRepository:
    """Repository for StoreItem database operations"""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        print("[Store Item Repository] Initialized")

    async def create_store_item(self, item_data: dict) -> Optional[StoreItem]:
        """Create a new store item"""
        try:
            store_item = StoreItem(**item_data)
            self.db_session.add(store_item)
            await self.db_session.commit()
            await self.db_session.refresh(store_item)
            
            print(f"[Store Item Repository] Created store item: {store_item.sku_id}")
            return store_item

        except IntegrityError as e:
            await self.db_session.rollback()
            print(f"[Store Item Repository] Duplicate SKU: {item_data.get('sku_id')}")
            return None
        except Exception as e:
            await self.db_session.rollback()
            print(f"[Store Item Repository] Error creating store item: {str(e)}")
            raise e

    async def get_by_sku_id(self, sku_id: str) -> Optional[StoreItem]:
        """Get store item by SKU ID"""
        try:
            stmt = select(StoreItem).where(StoreItem.sku_id == sku_id)
            result = await self.db_session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            print(f"[Store Item Repository] Error fetching item by SKU: {str(e)}")
            raise e

    async def get_all_sku_ids(self) -> List[str]:
        """Get all existing SKU IDs"""
        try:
            stmt = select(StoreItem.sku_id)
            result = await self.db_session.execute(stmt)
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            print(f"[Store Item Repository] Error fetching SKU IDs: {str(e)}")
            raise e

    async def get_recent_items(self, limit: int = 50) -> List[StoreItem]:
        """Get recent store items"""
        try:
            stmt = (
                select(StoreItem)
                .order_by(StoreItem.created_at.desc())
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            print(f"[Store Item Repository] Error fetching recent items: {str(e)}")
            raise e

    async def count_items(self) -> int:
        """Count total store items"""
        try:
            stmt = select(func.count(StoreItem.id))
            result = await self.db_session.execute(stmt)
            return result.scalar_one()
        except Exception as e:
            print(f"[Store Item Repository] Error counting items: {str(e)}")
            raise e
        
    async def get_paginated_items(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[StoreItem], int]:
        """
        Get paginated store items with filters.
        Returns (items, total_count)
        """
        try:
            # Build base query
            query = select(StoreItem)
            count_query = select(func.count(StoreItem.id))
            
            # Apply filters
            filter_conditions = []
            
            if filters:
                # Category filter
                if filters.get("category"):
                    filter_conditions.append(
                        StoreItem.category.in_(filters["category"])
                    )
                
                # Brand name filter
                if filters.get("brand_name"):
                    filter_conditions.append(
                        StoreItem.brand_name.in_(filters["brand_name"])
                    )
                
                # Product type filter
                if filters.get("product_type"):
                    filter_conditions.append(
                        StoreItem.product_type.in_(filters["product_type"])
                    )
                
                # Gender filter
                if filters.get("gender"):
                    filter_conditions.append(
                        StoreItem.gender.in_(filters["gender"])
                    )
                
                # Price range filter
                if filters.get("min_price") is not None:
                    filter_conditions.append(
                        StoreItem.lowest_price >= filters["min_price"]
                    )
                
                if filters.get("max_price") is not None:
                    filter_conditions.append(
                        StoreItem.lowest_price <= filters["max_price"]
                    )
                
                # Search query (title, description, tags)
                if filters.get("search_query"):
                    search_term = f"%{filters['search_query']}%"
                    filter_conditions.append(
                        or_(
                            StoreItem.title.ilike(search_term),
                            StoreItem.description.ilike(search_term),
                            StoreItem.tags.ilike(search_term)
                        )
                    )
            
            # Apply all filters
            if filter_conditions:
                query = query.where(and_(*filter_conditions))
                count_query = count_query.where(and_(*filter_conditions))
            
            # Get total count
            total_result = await self.db_session.execute(count_query)
            total_count = total_result.scalar() or 0
            
            # Apply sorting
            sort_column = getattr(StoreItem, sort_by, StoreItem.created_at)
            if sort_order.lower() == "asc":
                query = query.order_by(sort_column.asc())
            else:
                query = query.order_by(sort_column.desc())
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = await self.db_session.execute(query)
            items = result.scalars().all()
            
            return list(items), total_count
            
        except Exception as e:
            print(f"[Store Item Repository] Error in get_paginated_items: {str(e)}")
            raise
    
    async def get_unique_brands(self) -> List[str]:
        """Get all unique brand names"""
        try:
            query = select(StoreItem.brand_name).distinct().where(
                StoreItem.brand_name.isnot(None)
            ).order_by(StoreItem.brand_name)
            
            result = await self.db_session.execute(query)
            brands = [row[0] for row in result.fetchall() if row[0]]
            
            return brands
            
        except Exception as e:
            print(f"[Store Item Repository] Error getting unique brands: {str(e)}")
            return []
    
    async def get_price_range(self) -> Dict[str, Optional[float]]:
        """Get min and max price in the database"""
        try:
            query = select(
                func.min(StoreItem.lowest_price),
                func.max(StoreItem.lowest_price)
            ).where(StoreItem.lowest_price.isnot(None))
            
            result = await self.db_session.execute(query)
            min_price, max_price = result.one()
            
            return {
                "min_price": float(min_price) if min_price else None,
                "max_price": float(max_price) if max_price else None
            }
            
        except Exception as e:
            print(f"[Store Item Repository] Error getting price range: {str(e)}")
            return {"min_price": None, "max_price": None}
"""
Store item repository for database operations.
Handles all database interactions for store items.
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError

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
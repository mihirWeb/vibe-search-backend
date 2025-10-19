"""
Store item controller for handling business logic.
Orchestrates between CSV service and repository.
"""

from typing import List
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import os

from src.services.store_item_service import StoreItemCSVService
from src.repository.store_item_repository import StoreItemRepository
from src.schemas.store_item_schema import (
    ImportStoreItemsRequest,
    ImportStoreItemsResponse,
    StoreItemListResponse,
    StoreItemMinimalSchema
)


class StoreItemController:
    """Controller for store item operations"""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.repository = StoreItemRepository(db_session)
        
        # Path to CSV file
        self.csv_path = os.path.join(os.getcwd(), "products_export_20250916_071707.csv")
        
        print("[Store Item Controller] Initialized")
    
    async def import_items_from_csv(
        self, 
        request: ImportStoreItemsRequest
    ) -> ImportStoreItemsResponse:
        """Import store items from CSV with embeddings"""
        try:
            print(f"[Store Item Controller] Importing {request.num_items} items from CSV")
            
            # Check if CSV exists
            if not os.path.exists(self.csv_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"CSV file not found: {self.csv_path}"
                )
            
            # Initialize CSV service
            csv_service = StoreItemCSVService(self.csv_path)
            
            # Parse CSV
            df = csv_service.parse_csv()
            
            # Get existing SKU IDs if skip_existing is True
            existing_skus = set()
            if request.skip_existing:
                existing_skus = set(await self.repository.get_all_sku_ids())
                print(f"[Store Item Controller] Found {len(existing_skus)} existing SKUs")
            
            # Process items
            items_processed = 0
            items_created = 0
            items_skipped = 0
            items_failed = 0
            created_items = []
            errors = []
            
            for index, row in df.iterrows():
                if items_processed >= request.num_items:
                    break
                
                try:
                    # Convert row to dict
                    item_dict = csv_service.convert_row_to_dict(row)
                    sku_id = item_dict.get('sku_id')
                    
                    # Skip if exists
                    if request.skip_existing and sku_id in existing_skus:
                        items_skipped += 1
                        print(f"[Store Item Controller] Skipping existing SKU: {sku_id}")
                        continue
                    
                    # Process with embeddings
                    item_with_embeddings = await csv_service.process_item_with_embeddings(item_dict)
                    
                    # Create in database
                    created_item = await self.repository.create_store_item(item_with_embeddings)
                    
                    if created_item:
                        items_created += 1
                        created_items.append(self._to_minimal_schema(created_item))
                        print(f"[Store Item Controller] Created item: {sku_id}")
                    else:
                        items_skipped += 1
                        print(f"[Store Item Controller] Failed to create (duplicate?): {sku_id}")
                    
                    items_processed += 1
                    
                except Exception as e:
                    items_failed += 1
                    error_msg = f"Failed to process row {index}: {str(e)}"
                    errors.append({
                        'row_index': index,
                        'sku_id': item_dict.get('sku_id', 'unknown'),
                        'error': str(e)
                    })
                    print(f"[Store Item Controller] {error_msg}")
                    continue
            
            response = ImportStoreItemsResponse(
                success=items_created > 0,
                message=f"Import completed: {items_created} created, {items_skipped} skipped, {items_failed} failed",
                total_items_requested=request.num_items,
                items_processed=items_processed,
                items_created=items_created,
                items_skipped=items_skipped,
                items_failed=items_failed,
                items=created_items[:20],  # Return first 20 for preview
                errors=errors,
                imported_at=datetime.now()
            )
            
            print(f"[Store Item Controller] Import completed")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[Store Item Controller] Error importing items: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error importing items: {str(e)}"
            )
    
    async def get_recent_items(self, limit: int = 50) -> StoreItemListResponse:
        """Get recent store items"""
        try:
            print(f"[Store Item Controller] Fetching {limit} recent items")
            
            items = await self.repository.get_recent_items(limit)
            
            item_schemas = [self._to_minimal_schema(item) for item in items]
            
            response = StoreItemListResponse(
                success=True,
                message=f"Retrieved {len(items)} items",
                total_items=len(items),
                items=item_schemas
            )
            
            return response
            
        except Exception as e:
            print(f"[Store Item Controller] Error fetching recent items: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching recent items: {str(e)}"
            )
    
    def _to_minimal_schema(self, store_item) -> StoreItemMinimalSchema:
        """Convert StoreItem model to minimal schema"""
        return StoreItemMinimalSchema(
            id=store_item.id,
            sku_id=store_item.sku_id,
            title=store_item.title,
            brand_name=store_item.brand_name,
            category=store_item.category,
            featured_image=store_item.featured_image,
            lowest_price=store_item.lowest_price,
            pdp_url=store_item.pdp_url
        )
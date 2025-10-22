"""
Store item controller for handling business logic.
Orchestrates between CSV service and repository.
"""

from typing import List
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import os
import math

from src.services.store_item_service import StoreItemCSVService
from src.repository.store_item_repository import StoreItemRepository
from src.schemas.store_item_schema import (
    ImportStoreItemsRequest,
    ImportStoreItemsResponse,
    StoreItemListResponse,
    StoreItemMinimalSchema,
    StoreItemPaginationRequest,
    StoreItemPaginatedResponse,
    StoreItemDetailSchema,
    PaginationMeta,
    FindSimilarItemsRequest,
    FindSimilarItemsResponse,
    SimilarStoreItemSchema
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
            
    async def get_paginated_items(
        self, 
        request: StoreItemPaginationRequest
    ) -> StoreItemPaginatedResponse:
        """Get paginated store items with filters"""
        try:
            print(f"[Store Item Controller] Fetching page {request.page} with size {request.page_size}")
            
            # Convert filters to dict
            filters = None
            if request.filters:
                filters = {
                    k: v for k, v in request.filters.dict().items() 
                    if v is not None and v != [] and v != ""
                }
            
            # Get items and total count
            items, total_count = await self.repository.get_paginated_items(
                page=request.page,
                page_size=request.page_size,
                filters=filters,
                sort_by=request.sort_by,
                sort_order=request.sort_order
            )
            
            # Convert to schemas
            item_schemas = [self._to_detail_schema(item) for item in items]
            
            # Calculate pagination metadata
            total_pages = math.ceil(total_count / request.page_size) if total_count > 0 else 0
            
            pagination_meta = PaginationMeta(
                current_page=request.page,
                page_size=request.page_size,
                total_items=total_count,
                total_pages=total_pages,
                has_next=request.page < total_pages,
                has_previous=request.page > 1
            )
            
            response = StoreItemPaginatedResponse(
                success=True,
                message=f"Retrieved {len(items)} items (page {request.page} of {total_pages})",
                items=item_schemas,
                pagination=pagination_meta,
                filters_applied=filters
            )
            
            return response
            
        except Exception as e:
            print(f"[Store Item Controller] Error fetching paginated items: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching paginated items: {str(e)}"
            )
    
    async def get_filter_options(self) -> dict:
        """Get available filter options"""
        try:
            from src.constants.store_item_enums import (
                get_all_categories,
                get_all_product_types,
                get_all_genders
            )
            
            # Get unique brands from database
            brands = await self.repository.get_unique_brands()
            
            # Get price range
            price_range = await self.repository.get_price_range()
            
            return {
                "categories": get_all_categories(),
                "product_types": get_all_product_types(),
                "genders": get_all_genders(),
                "brands": brands,
                "price_range": price_range
            }
            
        except Exception as e:
            print(f"[Store Item Controller] Error fetching filter options: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching filter options: {str(e)}"
            )
    
    def _to_detail_schema(self, store_item) -> StoreItemDetailSchema:
        """Convert StoreItem model to detail schema"""
        return StoreItemDetailSchema(
            id=store_item.id,
            sku_id=store_item.sku_id,
            title=store_item.title,
            slug=store_item.slug,
            category=store_item.category,
            sub_category=store_item.sub_category,
            brand_name=store_item.brand_name,
            product_type=store_item.product_type,
            gender=store_item.gender,
            colorways=store_item.colorways,
            brand_sku=store_item.brand_sku,
            model=store_item.model,
            lowest_price=store_item.lowest_price,
            description=store_item.description,
            is_d2c=store_item.is_d2c,
            is_active=store_item.is_active,
            is_certificate_required=store_item.is_certificate_required,
            featured_image=store_item.featured_image,
            pdp_url=store_item.pdp_url,
            quantity_left=store_item.quantity_left,
            wishlist_num=store_item.wishlist_num,
            stock_claimed_percent=store_item.stock_claimed_percent,
            discount_percentage=store_item.discount_percentage,
            note=store_item.note,
            tags=store_item.tags,
            release_date=store_item.release_date,
            created_at=store_item.created_at,
            updated_at=store_item.updated_at
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
        

    async def find_similar_items_by_product_item(
        self,
        request: FindSimilarItemsRequest
    ) -> FindSimilarItemsResponse:
        """Find similar store items based on product item's visual embedding"""
        try:
            print(f"[Store Item Controller] Finding similar items for product item {request.product_item_id}")
            
            # Get product item from product repository
            from src.repository.product_repository import ProductRepository
            product_repo = ProductRepository(self.db_session)
            
            # Get product item by ID
            from sqlalchemy import select
            from src.models.product_item_model import ProductItem
            
            stmt = select(ProductItem).where(ProductItem.id == request.product_item_id)
            result = await self.db_session.execute(stmt)
            product_item = result.scalar_one_or_none()
            
            if not product_item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Product item not found with ID: {request.product_item_id}"
                )
            
            # Check if product item has visual embedding
            # Fix: Check for None first, then check length
            if product_item.embedding is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Product item {request.product_item_id} does not have a visual embedding"
                )
            
            # Convert to list if it's a numpy array or similar
            embedding_list = list(product_item.embedding)
            
            if len(embedding_list) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Product item {request.product_item_id} has an empty visual embedding"
                )
            
            print(f"[Store Item Controller] Product item embedding dimension: {len(embedding_list)}")
            
            # Find similar store items
            similar_items_with_scores = await self.repository.find_similar_items_by_visual_embedding(
                visual_embedding=embedding_list,
                limit=request.limit,
                similarity_threshold=request.similarity_threshold
            )
            
            # Convert to response schemas
            similar_item_schemas = [
                SimilarStoreItemSchema(
                    item=self._to_detail_schema(item),
                    similarity_score=round(score, 4)
                )
                for item, score in similar_items_with_scores
            ]
            
            response = FindSimilarItemsResponse(
                success=True,
                message=f"Found {len(similar_item_schemas)} similar items",
                product_item_id=request.product_item_id,
                total_similar_items=len(similar_item_schemas),
                similar_items=similar_item_schemas
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[Store Item Controller] Error finding similar items: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error finding similar items: {str(e)}"
            )
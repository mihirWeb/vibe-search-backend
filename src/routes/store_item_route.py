"""
Store item routes for FastAPI endpoints.
Handles HTTP requests for importing and managing store items from CSV.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.store_item_schema import (
    ImportStoreItemsRequest,
    ImportStoreItemsResponse,
    StoreItemListResponse
)
from src.controller.store_item_controller import StoreItemController
from src.config.database import get_db

router = APIRouter()


def get_store_item_controller(db: AsyncSession = Depends(get_db)) -> StoreItemController:
    """Dependency injection for StoreItemController"""
    return StoreItemController(db)


@router.post("/import", response_model=ImportStoreItemsResponse)
async def import_store_items_from_csv(
    request: ImportStoreItemsRequest,
    controller: StoreItemController = Depends(get_store_item_controller)
) -> ImportStoreItemsResponse:
    """
    Import store items from CSV file with embeddings.
    
    This endpoint:
    1. Reads the specified number of items from the CSV file
    2. Generates textual embeddings (768-dim) using Sentence Transformers
    3. Generates visual embeddings (512-dim) using Open-CLIP
    4. Generates multimodal embeddings (1280-dim) by fusing text and visual
    5. Detects objects using YOLO and generates object-level embeddings
    6. Saves all data including embeddings to the database
    7. Tracks which items have been imported to avoid duplicates
    
    **Embedding Generation:**
    - **Textual**: Combined text from title, description, category, brand, tags
    - **Visual**: Extracted from product featured image using CLIP
    - **Multimodal**: Weighted fusion of text (60%) and visual (40%) embeddings
    - **Objects**: YOLO detection + CLIP embeddings for detected objects
    
    **Parameters:**
    - `num_items`: Number of items to import (1-1000)
    - `skip_existing`: Skip items that already exist (based on SKU ID)
    
    **Example Request:**
    ```json
    {
        "num_items": 50,
        "skip_existing": true
    }
    ```
    
    **Returns:**
    - Statistics: processed, created, skipped, failed counts
    - Sample of created items (first 20)
    - Error details for failed imports
    """
    try:
        return await controller.import_items_from_csv(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/", response_model=StoreItemListResponse)
async def get_recent_store_items(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of items to return"),
    controller: StoreItemController = Depends(get_store_item_controller)
) -> StoreItemListResponse:
    """
    Get recent store items.
    
    Retrieves the most recently imported store items, ordered by creation date.
    Embeddings are excluded from the response for performance.
    
    **Parameters:**
    - `limit`: Maximum number of items to return (default: 50, max: 100)
    
    **Returns:**
    - List of store items with basic information (no embeddings)
    - Total count of items returned
    """
    try:
        return await controller.get_recent_items(limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
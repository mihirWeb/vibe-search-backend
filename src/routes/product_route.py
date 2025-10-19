"""
Product routes for FastAPI endpoints.
Handles HTTP requests for product extraction from Instagram posts.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.product_schema import (
    ExtractProductRequest,
    ExtractProductResponse,
    BatchExtractProductRequest,
    BatchExtractProductResponse,
    ProductListResponse,
    ProductSchemaMinimal
)
from src.controller.product_controller import ProductController
from src.config.database import get_db

router = APIRouter()


def get_product_controller(db: AsyncSession = Depends(get_db)) -> ProductController:
    """Dependency injection for ProductController"""
    return ProductController(db)


@router.post("/extract", response_model=ExtractProductResponse)
async def extract_product_from_instagram_post(
    request: ExtractProductRequest,
    controller: ProductController = Depends(get_product_controller)
) -> ExtractProductResponse:
    """
    Extract products and items from an Instagram post.
    
    This endpoint:
    1. Fetches the Instagram post from the database using the provided post ID
    2. Downloads the post image and processes it using ML models
    3. Extracts products and individual items (clothing, accessories, etc.)
    4. Generates visual and text embeddings for products and items
    5. Saves the extracted products to the database
    
    **Process Flow:**
    - Downloads image from Instagram post
    - Extracts metadata from caption (brands, items, styles, colors)
    - Detects items using YOLO object detection
    - Classifies items using Open-CLIP zero-shot classification
    - Generates Open-CLIP visual embeddings (512-dim)
    - Generates Sentence-Transformers text embeddings (384-dim)
    - Extracts dominant colors using K-means clustering
    
    **Requirements:**
    - Instagram post must already exist in the database
    - Post must have a valid display URL for image download
    
    **Example Request:**
    ```json
    {
        "instagram_post_id": "3745039532065324748"
    }
    ```
    
    **Returns:**
    - Extracted product with metadata and items
    - Visual and text embeddings (excluded from response but stored in DB)
    - Bounding boxes for detected items
    - Confidence scores for detections
    """
    try:
        return await controller.extract_product_from_instagram_post(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.post("/extract/batch", response_model=BatchExtractProductResponse)
async def batch_extract_products_from_instagram_posts(
    request: BatchExtractProductRequest,
    controller: ProductController = Depends(get_product_controller)
) -> BatchExtractProductResponse:
    """
    Extract products and items from multiple Instagram posts in batch.
    
    This endpoint processes multiple Instagram posts and extracts products from each.
    Failed extractions are reported in the errors array without failing the entire batch.
    
    **Process:**
    - Iterates through all provided Instagram post IDs
    - Extracts products using the same pipeline as single extraction
    - Continues processing even if individual posts fail
    - Returns summary with successful and failed extractions
    
    **Example Request:**
    ```json
    {
        "instagram_post_ids": [
            "3745039532065324748",
            "3745039532065324749",
            "3745039532065324750"
        ]
    }
    ```
    
    **Returns:**
    - List of successfully extracted products
    - Count of successful and failed extractions
    - Detailed error messages for failed extractions
    """
    try:
        return await controller.batch_extract_products_from_instagram_posts(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/{product_id}", response_model=ProductSchemaMinimal)
async def get_product_by_id(
    product_id: int,
    controller: ProductController = Depends(get_product_controller)
) -> ProductSchemaMinimal:
    """
    Get a product by ID.
    
    Retrieves a single product with its metadata.
    Embeddings are excluded from the response for performance.
    
    **Parameters:**
    - `product_id`: The unique identifier of the product
    
    **Returns:**
    - Product details including name, description, brand, category
    - Style attributes and dominant colors
    - Original Instagram caption
    - Metadata (source, post ID, hashtags, engagement metrics)
    - Items with minimal information (id, product_id, name, category, style, bounding_box, confidence_score)
    """
    try:
        return await controller.get_product_by_id(product_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/", response_model=ProductListResponse)
async def get_recent_products(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of products to return"),
    controller: ProductController = Depends(get_product_controller)
) -> ProductListResponse:
    """
    Get recent products.
    
    Retrieves the most recently extracted products, ordered by creation date.
    Embeddings are excluded from the response for performance.
    
    **Parameters:**
    - `limit`: Maximum number of products to return (default: 50, max: 100)
    
    **Returns:**
    - List of products with metadata
    - Total count of products returned
    - Items with minimal information (id, product_id, name, category, style, bounding_box, confidence_score)
    """
    try:
        return await controller.get_recent_products(limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/brand/{brand}", response_model=ProductListResponse)
async def get_products_by_brand(
    brand: str,
    limit: int = Query(50, ge=1, le=100, description="Maximum number of products to return"),
    controller: ProductController = Depends(get_product_controller)
) -> ProductListResponse:
    """
    Get products by brand.
    
    Retrieves all products for a specific brand, ordered by creation date.
    Embeddings are excluded from the response for performance.
    
    **Parameters:**
    - `brand`: The brand name to filter by
    - `limit`: Maximum number of products to return (default: 50, max: 100)
    
    **Example:**
    - `/brand/Fear of God`
    - `/brand/Nike`
    
    **Returns:**
    - List of products from the specified brand
    - Total count of products returned
    - Items with minimal information (id, product_id, name, category, style, bounding_box, confidence_score)
    """
    try:
        return await controller.get_products_by_brand(brand, limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
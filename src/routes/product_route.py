"""
Product routes for FastAPI endpoints.
Handles HTTP requests for product extraction and management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from urllib.parse import unquote
import httpx

from src.schemas.product_schema import (
    ExtractProductRequest,
    ExtractProductResponse,
    BatchExtractProductRequest,
    BatchExtractProductResponse,
    ProductListResponse,
    ProductSchemaMinimal,
    ProductPaginationRequest,
    ProductPaginatedResponse
)
from src.controller.product_controller import ProductController
from src.config.database import get_db

router = APIRouter()


@router.get("/image-proxy")
async def proxy_image(request: Request, url: str = Query(...)):
    """Proxy external images to bypass CORS and hotlink protection"""
    try:
        # Decode the URL
        decoded_url = unquote(url)
        
        # Validate that it's an Instagram URL (security measure)
        if not decoded_url.startswith(('https://scontent-', 'https://instagram.', 'https://www.instagram.')):
            raise HTTPException(status_code=400, detail="Invalid URL domain")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                decoded_url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Referer': 'https://www.instagram.com/',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                },
                follow_redirects=True
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch image: {response.status_code}")
            
            # Get the content type from the response
            content_type = response.headers.get('content-type', 'image/jpeg')
            
            return Response(
                content=response.content,
                media_type=content_type,
                headers={
                    'Cache-Control': 'public, max-age=31536000, immutable',
                    'Access-Control-Allow-Origin': '*',  # Allow any origin to access the image
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                }
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Request timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/extract", response_model=ExtractProductResponse, status_code=status.HTTP_200_OK)
async def extract_product_from_instagram_post(
    request: ExtractProductRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract product and items from an Instagram post.
    
    - **instagram_post_id**: ID of the Instagram post to extract product from
    """
    controller = ProductController(db)
    return await controller.extract_product_from_instagram_post(request)


@router.post("/extract/batch", response_model=BatchExtractProductResponse, status_code=status.HTTP_200_OK)
async def batch_extract_products_from_instagram_posts(
    request: BatchExtractProductRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract products from multiple Instagram posts in batch.
    
    - **instagram_post_ids**: List of Instagram post IDs to extract products from
    """
    controller = ProductController(db)
    return await controller.batch_extract_products_from_instagram_posts(request)


@router.post("/list", response_model=ProductPaginatedResponse, status_code=status.HTTP_200_OK)
async def get_paginated_products(
    request: ProductPaginationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated list of all products with their items.
    
    Uses minimal schemas (excludes embeddings) for better performance.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **brand**: Optional brand filter
    - **category**: Optional category filter
    - **sort_by**: Sort field (default: created_at)
    - **sort_order**: Sort order - 'asc' or 'desc' (default: desc)
    """
    controller = ProductController(db)
    return await controller.get_paginated_products(request)


@router.get("/{product_id}", response_model=ProductSchemaMinimal, status_code=status.HTTP_200_OK)
async def get_product_by_id(
    product_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific product by ID with its items.
    
    Uses minimal schema (excludes embeddings).
    
    - **product_id**: Product ID
    """
    controller = ProductController(db)
    return await controller.get_product_by_id(product_id)

@router.delete("/{product_id}", status_code=status.HTTP_200_OK)
async def delete_product(
    product_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a product and its items.
    
    - **product_id**: Product ID to delete
    """
    controller = ProductController(db)
    return await controller.delete_product(product_id)


@router.get("/recent", response_model=ProductListResponse, status_code=status.HTTP_200_OK)
async def get_recent_products(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of products to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent products with their items.
    
    Uses minimal schemas (excludes embeddings).
    
    - **limit**: Maximum number of products (default: 50, max: 100)
    """
    controller = ProductController(db)
    return await controller.get_recent_products(limit)


@router.get("/brand/{brand}", response_model=ProductListResponse, status_code=status.HTTP_200_OK)
async def get_products_by_brand(
    brand: str,
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of products to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get products by brand with their items.
    
    Uses minimal schemas (excludes embeddings).
    
    - **brand**: Brand name
    - **limit**: Maximum number of products (default: 50, max: 100)
    """
    controller = ProductController(db)
    return await controller.get_products_by_brand(brand, limit)

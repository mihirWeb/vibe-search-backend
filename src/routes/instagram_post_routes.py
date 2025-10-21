"""
Instagram Post routes for FastAPI endpoints.
Handles HTTP requests for Instagram post management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.instagram_post_response_schema import (
    InstagramPostPaginationRequest,
    InstagramPostPaginatedResponse,
    InstagramPostDetailResponse
)
from src.controller.instagram_post_controller import InstagramPostController
from src.config.database import get_db

router = APIRouter()


@router.post("/list", response_model=InstagramPostPaginatedResponse, status_code=status.HTTP_200_OK)
async def get_paginated_instagram_posts(
    request: InstagramPostPaginationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated list of all Instagram posts.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **owner_username**: Optional username filter
    - **sort_by**: Sort field (default: scraped_date)
    - **sort_order**: Sort order - 'asc' or 'desc' (default: desc)
    """
    controller = InstagramPostController(db)
    return await controller.get_paginated_posts(request)


@router.get("/{post_id}", response_model=InstagramPostDetailResponse, status_code=status.HTTP_200_OK)
async def get_instagram_post_by_id(
    post_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific Instagram post by ID with full details.
    
    - **post_id**: Instagram post ID
    """
    controller = InstagramPostController(db)
    return await controller.get_post_by_id(post_id)
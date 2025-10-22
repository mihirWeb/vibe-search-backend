from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class InstagramPostMinimalSchema(BaseModel):
    """Minimal schema for Instagram Post - used in list responses"""
    id: str
    type: str
    url: str
    display_url: str
    caption: Optional[str] = None
    likes_count: int
    comments_count: int
    timestamp: datetime
    owner_username: str
    owner_full_name: str
    scraped_date: datetime

    class Config:
        from_attributes = True


class PaginationMeta(BaseModel):
    """Pagination metadata"""
    current_page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool


class InstagramPostPaginationRequest(BaseModel):
    """Request schema for paginated Instagram post list"""
    page: int = Field(default=1, ge=1, description="Page number (starts from 1)")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of items per page")
    owner_username: Optional[str] = Field(default=None, description="Filter by owner username")
    sort_by: Optional[str] = Field(default="scraped_date", description="Sort field (scraped_date, timestamp, likes_count)")
    sort_order: Optional[str] = Field(default="desc", description="Sort order (asc or desc)")


class InstagramPostPaginatedResponse(BaseModel):
    """Response schema for paginated Instagram post list"""
    success: bool
    message: str
    posts: List[InstagramPostMinimalSchema] = []
    pagination: PaginationMeta


class InstagramPostDetailResponse(BaseModel):
    """Response schema for detailed Instagram post"""
    success: bool
    message: str
    post: Dict[str, Any]
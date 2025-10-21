from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from decimal import Decimal


class StoreItemSchema(BaseModel):
    """Full schema for StoreItem"""
    id: int
    sku_id: str
    title: str
    slug: str
    category: Optional[str] = None
    sub_category: Optional[str] = None
    brand_name: Optional[str] = None
    product_type: Optional[str] = None
    gender: Optional[str] = None
    colorways: Optional[str] = None
    brand_sku: Optional[str] = None
    model: Optional[str] = None
    lowest_price: Optional[Decimal] = None
    description: Optional[str] = None
    is_d2c: bool = False
    is_active: bool = True
    is_certificate_required: bool = False
    featured_image: Optional[str] = None
    pdp_url: Optional[str] = None
    quantity_left: int = 0
    wishlist_num: int = 0
    stock_claimed_percent: int = 0
    discount_percentage: int = 0
    note: Optional[str] = None
    tags: Optional[str] = None
    release_date: Optional[date] = None
    csv_created_at: Optional[datetime] = None
    csv_updated_at: Optional[datetime] = None
    textual_embedding: Optional[List[float]] = None
    visual_embedding: Optional[List[float]] = None
    multimodal_embedding: Optional[List[float]] = None
    object_embeddings: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StoreItemMinimalSchema(BaseModel):
    """Minimal schema for StoreItem - excludes embeddings"""
    id: int
    sku_id: str
    title: str
    brand_name: Optional[str] = None
    category: Optional[str] = None
    featured_image: Optional[str] = None
    lowest_price: Optional[Decimal] = None
    pdp_url: Optional[str] = None

    class Config:
        from_attributes = True


class ImportStoreItemsRequest(BaseModel):
    """Request schema for importing store items from CSV"""
    num_items: int = Field(..., ge=1, le=1000, description="Number of items to import from CSV")
    skip_existing: bool = Field(True, description="Skip items that already exist in database")


class ImportStoreItemsResponse(BaseModel):
    """Response schema for store item import"""
    success: bool
    message: str
    total_items_requested: int
    items_processed: int
    items_created: int
    items_skipped: int
    items_failed: int
    items: List[StoreItemMinimalSchema] = []
    errors: List[Dict[str, Any]] = []
    imported_at: datetime


class StoreItemListResponse(BaseModel):
    """Response schema for store item list"""
    success: bool
    message: str
    total_items: int
    items: List[StoreItemMinimalSchema] = []
    
class StoreItemFilterRequest(BaseModel):
    """Filter parameters for store items"""
    category: Optional[List[str]] = Field(None, description="Filter by categories")
    brand_name: Optional[List[str]] = Field(None, description="Filter by brand names")
    product_type: Optional[List[str]] = Field(None, description="Filter by product types")
    gender: Optional[List[str]] = Field(None, description="Filter by gender")
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price")
    search_query: Optional[str] = Field(None, description="Search in title, description, tags")


class StoreItemPaginationRequest(BaseModel):
    """Pagination and filter request for store items"""
    page: int = Field(1, ge=1, description="Page number (starts from 1)")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page")
    filters: Optional[StoreItemFilterRequest] = None
    sort_by: Optional[str] = Field("created_at", description="Field to sort by")
    sort_order: Optional[str] = Field("desc", description="Sort order: asc or desc")


class StoreItemDetailSchema(BaseModel):
    """Detailed store item schema"""
    id: int
    sku_id: str
    title: str
    slug: str
    category: Optional[str] = None
    sub_category: Optional[str] = None
    brand_name: Optional[str] = None
    product_type: Optional[str] = None
    gender: Optional[str] = None
    colorways: Optional[str] = None
    brand_sku: Optional[str] = None
    model: Optional[str] = None
    lowest_price: Optional[Decimal] = None
    description: Optional[str] = None
    is_d2c: bool = False
    is_active: bool = True
    is_certificate_required: bool = False
    featured_image: Optional[str] = None
    pdp_url: Optional[str] = None
    quantity_left: int = 0
    wishlist_num: int = 0
    stock_claimed_percent: int = 0
    discount_percentage: int = 0
    note: Optional[str] = None
    tags: Optional[str] = None
    release_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
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


class StoreItemPaginatedResponse(BaseModel):
    """Paginated response for store items"""
    success: bool = True
    message: str
    items: List[StoreItemDetailSchema]
    pagination: PaginationMeta
    filters_applied: Optional[dict] = None
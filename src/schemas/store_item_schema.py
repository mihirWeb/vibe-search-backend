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
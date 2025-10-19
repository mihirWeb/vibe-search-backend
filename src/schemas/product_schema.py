from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ProductItemMinimalSchema(BaseModel):
    """Minimal schema for ProductItem - used in product list responses"""
    id: int
    product_id: int
    name: str
    category: str
    style: Optional[List[str]] = None
    bounding_box: Optional[List[float]] = None
    confidence_score: Optional[float] = None

    class Config:
        from_attributes = True


class ProductItemSchema(BaseModel):
    """Full schema for ProductItem"""
    id: int
    product_id: int
    name: str
    brand: Optional[str] = None
    category: str
    style: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    product_type: str
    description: Optional[str] = None
    visual_features: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None
    bounding_box: Optional[List[float]] = None
    confidence_score: Optional[float] = None
    meta_info: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProductSchemaMinimal(BaseModel):
    """Minimal schema for Product - excludes embeddings and returns minimal item info"""
    id: int
    name: str
    description: Optional[str] = None
    image_url: str
    source_url: str
    brand: Optional[str] = None
    category: Optional[str] = None
    style: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    caption: Optional[str] = None
    meta_info: Optional[Dict[str, Any]] = None
    items: List[ProductItemMinimalSchema] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProductSchema(BaseModel):
    """Full schema for Product - includes all fields"""
    id: int
    name: str
    description: Optional[str] = None
    image_url: str
    source_url: str
    brand: Optional[str] = None
    category: Optional[str] = None
    style: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    caption: Optional[str] = None
    embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None
    meta_info: Optional[Dict[str, Any]] = None
    items: List[ProductItemSchema] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExtractProductRequest(BaseModel):
    """Request schema for extracting product from Instagram post"""
    instagram_post_id: str = Field(..., description="Instagram post ID")


class ExtractProductResponse(BaseModel):
    """Response schema for product extraction"""
    success: bool
    message: str
    product: Optional[ProductSchema] = None
    instagram_post_id: str
    extracted_at: datetime


class BatchExtractProductRequest(BaseModel):
    """Request schema for batch product extraction"""
    instagram_post_ids: List[str] = Field(..., description="List of Instagram post IDs")


class BatchExtractProductResponse(BaseModel):
    """Response schema for batch product extraction"""
    success: bool
    message: str
    total_posts: int
    successful_extractions: int
    failed_extractions: int
    products: List[ProductSchema] = []
    errors: List[Dict[str, Any]] = []
    extracted_at: datetime


class ProductListResponse(BaseModel):
    """Response schema for product list"""
    success: bool
    message: str
    total_products: int
    products: List[ProductSchemaMinimal] = []
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


class ProductItemSchema(BaseModel):
    """Schema for product item"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    product_id: Optional[int] = None
    name: str
    brand: Optional[str] = None
    category: str
    style: List[str] = []
    colors: List[str] = []
    product_type: str
    description: Optional[str] = None
    visual_features: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None
    bounding_box: List[float] = []
    confidence_score: float = 0.0
    meta_info: Optional[Dict[str, Any]] = None  # Changed from metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProductSchema(BaseModel):
    """Schema for product"""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    image_url: str
    source_url: str
    brand: Optional[str] = None
    category: str
    style: List[str] = []
    colors: List[str] = []
    caption: Optional[str] = None
    embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None
    meta_info: Optional[Dict[str, Any]] = None  # Changed from metadata
    items: List[ProductItemSchema] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ExtractProductRequest(BaseModel):
    """Request schema for extracting products from Instagram post"""
    instagram_post_id: str = Field(..., description="Instagram post ID to extract products from")


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
    products: List[ProductSchema] = []
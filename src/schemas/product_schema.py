"""
Pydantic schemas for product extraction API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ProductItemSchema(BaseModel):
    """Schema for a product item"""
    id: Optional[int] = Field(None, description="Item ID (set after creation)")
    product_id: Optional[int] = Field(None, description="Parent product ID")
    name: str = Field(..., description="Name of the item")
    brand: Optional[str] = Field(None, description="Brand of the item")
    category: str = Field(..., description="Category of the item")
    style: List[str] = Field(default_factory=list, description="Style attributes")
    colors: List[str] = Field(default_factory=list, description="Dominant colors (hex codes)")
    product_type: str = Field(..., description="Type of product")
    description: str = Field(..., description="Description of the item")
    visual_features: Dict[str, Any] = Field(default_factory=dict, description="Visual features")
    embedding: Optional[List[float]] = Field(None, description="Visual embedding vector")
    text_embedding: Optional[List[float]] = Field(None, description="Text embedding vector")
    bounding_box: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence_score: float = Field(..., description="Detection confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        from_attributes = True


class ProductSchema(BaseModel):
    """Schema for a product"""
    id: Optional[int] = Field(None, description="Product ID (set after creation)")
    name: str = Field(..., description="Name of the product")
    description: str = Field(..., description="Description of the product")
    image_url: str = Field(..., description="URL of the product image")
    source_url: Optional[str] = Field(None, description="Source URL (Instagram post)")
    brand: str = Field(..., description="Main brand")
    category: str = Field(..., description="Product category")
    style: List[str] = Field(default_factory=list, description="Style attributes")
    colors: List[str] = Field(default_factory=list, description="Dominant colors (hex codes)")
    caption: Optional[str] = Field(None, description="Original caption from Instagram")
    embedding: Optional[List[float]] = Field(None, description="Visual embedding vector")
    text_embedding: Optional[List[float]] = Field(None, description="Text embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    items: List[ProductItemSchema] = Field(default_factory=list, description="Product items")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        from_attributes = True


class ExtractProductRequest(BaseModel):
    """Request schema for extracting products from an Instagram post ID"""
    instagram_post_id: str = Field(..., description="Instagram post ID to extract products from")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instagram_post_id": "3745039532065324748"
            }
        }


class BatchExtractProductRequest(BaseModel):
    """Request schema for batch extracting products from multiple Instagram post IDs"""
    instagram_post_ids: List[str] = Field(..., description="List of Instagram post IDs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instagram_post_ids": ["3745039532065324748", "3745039532065324749"]
            }
        }


class ExtractProductResponse(BaseModel):
    """Response schema for product extraction"""
    success: bool = Field(..., description="Whether the extraction was successful")
    message: str = Field(..., description="Response message")
    product: Optional[ProductSchema] = Field(None, description="Extracted product")
    instagram_post_id: str = Field(..., description="Instagram post ID")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully extracted product from Instagram post",
                "product": {
                    "id": 1,
                    "name": "Fear of God Collection - Streetwear",
                    "description": "A streetwear collection featuring Tee, Shorts",
                    "brand": "Fear of God",
                    "category": "Fashion",
                    "items": []
                },
                "instagram_post_id": "3745039532065324748",
                "extracted_at": "2025-10-18T12:00:00"
            }
        }


class BatchExtractProductResponse(BaseModel):
    """Response schema for batch product extraction"""
    success: bool = Field(..., description="Whether the batch extraction was successful")
    message: str = Field(..., description="Response message")
    total_posts: int = Field(..., description="Total number of posts processed")
    successful_extractions: int = Field(..., description="Number of successful extractions")
    failed_extractions: int = Field(..., description="Number of failed extractions")
    products: List[ProductSchema] = Field(default_factory=list, description="Extracted products")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Errors encountered")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Batch extraction completed",
                "total_posts": 2,
                "successful_extractions": 2,
                "failed_extractions": 0,
                "products": [],
                "errors": [],
                "extracted_at": "2025-10-18T12:00:00"
            }
        }


class ProductListResponse(BaseModel):
    """Response schema for listing products"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    total_products: int = Field(..., description="Total number of products")
    products: List[ProductSchema] = Field(default_factory=list, description="List of products")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Products retrieved successfully",
                "total_products": 10,
                "products": []
            }
        }

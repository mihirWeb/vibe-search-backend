"""
Schemas for search requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from decimal import Decimal


class SearchFilters(BaseModel):
    """Filters for search queries"""
    category: Optional[List[str]] = Field(None, description="Filter by product categories")
    brands: Optional[List[str]] = Field(None, description="Filter by brand names")
    gender: Optional[str] = Field(None, description="Filter by gender")
    colors: Optional[List[str]] = Field(None, description="Filter by colors")
    price_range: Optional[List[float]] = Field(None, description="Price range [min, max]")


class TextSearchRequest(BaseModel):
    """Request schema for text-based search"""
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    filters: Optional[SearchFilters] = Field(None, description="Search filters")
    rerank: bool = Field(True, description="Whether to rerank results")


class ImageSearchRequest(BaseModel):
    """Request schema for image-based search"""
    image_url: str = Field(..., description="URL of the image to search")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    filters: Optional[SearchFilters] = Field(None, description="Search filters")
    rerank: bool = Field(True, description="Whether to rerank results")


class MatchResult(BaseModel):
    """Individual search result match"""
    product_id: str = Field(..., description="Product SKU ID")
    title: str = Field(..., description="Product title")
    brand: Optional[str] = Field(None, description="Brand name")
    category: Optional[str] = Field(None, description="Product category")
    sub_category: Optional[str] = Field(None, description="Product sub-category")
    price: Optional[Decimal] = Field(None, description="Product price")
    image_url: Optional[str] = Field(None, description="Product image URL")
    pdp_url: Optional[str] = Field(None, description="Product detail page URL")
    similarity_score: Optional[float] = Field(None, description="Similarity score")
    combined_score: Optional[float] = Field(None, description="Combined ranking score")
    match_reasons: List[str] = Field(default_factory=list, description="Reasons for match")
    colorways: Optional[str] = Field(None, description="Product colors")
    gender: Optional[str] = Field(None, description="Product gender")


class LookItem(BaseModel):
    """Item in a look/collection"""
    product_id: Optional[str] = Field(None, description="Product SKU ID")
    title: Optional[str] = Field(None, description="Product title")
    brand: Optional[str] = Field(None, description="Brand name")
    category: Optional[str] = Field(None, description="Product category")
    price: Optional[Decimal] = Field(None, description="Product price")
    image_url: Optional[str] = Field(None, description="Product image URL")
    pdp_url: Optional[str] = Field(None, description="Product detail page URL")
    type: str = Field(..., description="Item type (top, bottom, etc.)")
    is_existing: bool = Field(False, description="Whether user already has this item")
    brands: Optional[List[str]] = Field(None, description="Brands for existing items")


class Look(BaseModel):
    """A complete outfit/look"""
    look_id: int = Field(..., description="Look number")
    items: Dict[str, LookItem] = Field(..., description="Items in this look by type")


class CollectionSearchResponse(BaseModel):
    """Response for collection search queries"""
    success: bool = Field(True, description="Whether search was successful")
    query_understanding: Dict[str, Any] = Field(..., description="Query analysis")
    looks: List[Look] = Field(..., description="Generated looks/collections")
    search_strategy: str = Field(..., description="Search strategy used")
    total_looks: int = Field(..., description="Total number of looks")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    is_collection_query: bool = Field(True, description="Always true for collection queries")


class TextSearchResponse(BaseModel):
    """Response schema for text-based search"""
    success: bool = Field(True, description="Whether search was successful")
    query_understanding: Dict[str, Any] = Field(..., description="Query analysis details")
    matches: List[MatchResult] = Field(default_factory=list, description="Search results")
    search_strategy: str = Field(..., description="Search strategy used")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    is_collection_query: bool = Field(False, description="Whether this was a collection query")
    looks: Optional[List[Look]] = Field(None, description="Looks if collection query")
    total_looks: Optional[int] = Field(None, description="Total looks if collection query")


class ImageSearchResponse(BaseModel):
    """Response schema for image-based search"""
    success: bool = Field(True, description="Whether search was successful")
    query_analysis: Dict[str, Any] = Field(..., description="Image analysis details")
    matches: List[MatchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Literal, Union, Any
from datetime import datetime

class ScrapeRequest(BaseModel):
    url: HttpUrl = Field(..., description="Instagram or Pinterest URL to scrape")
    post_limit: int = Field(default=20, ge=1, le=100, description="Number of posts to return")
    use_api: bool = Field(default=True, description="Apify is always API-based (kept for compatibility)")

class ScrapedPost(BaseModel):
    # Handle both raw and structured data
    source: Literal["instagram", "pinterest"]
    raw_data: Optional[Dict[str, Any]] = None  # For Pinterest or fallback
    structured_data: Optional[Dict[str, Any]] = None  # For transformed Instagram data
    scraped_date: datetime
    extraction_method: str
    extraction_error: Optional[str] = None
    transformation_error: Optional[str] = None

class ScrapeResponse(BaseModel):
    success: bool
    message: str
    total_posts: int
    posts: List[ScrapedPost]
    url: str
    platform: str
    scraped_at: datetime
    estimated_cost: Optional[float] = None

# New batch scraping models
class BatchScrapeRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., description="List of Instagram or Pinterest URLs to scrape")
    post_limit: int = Field(default=20, ge=1, le=100, description="Number of posts to return per URL")

class BatchScrapeResponse(BaseModel):
    success: bool
    message: str
    total_posts: int
    posts: List[ScrapedPost]
    urls_processed: int
    urls_failed: int
    errors: List[Dict[str, str]]
    scraped_at: datetime
    total_cost: Optional[float] = None
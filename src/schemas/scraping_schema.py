from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Literal
from datetime import datetime

class ScrapeRequest(BaseModel):
    url: HttpUrl = Field(..., description="Instagram or Pinterest URL to scrape")
    post_limit: int = Field(default=20, ge=1, le=100, description="Number of posts to return")
    use_api: bool = Field(default=False, description="Whether to use paid API for Pinterest")

class OwnerInfo(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    full_name: Optional[str] = None
    profile_url: Optional[str] = None
    follower_count: Optional[int] = None

class BoardInfo(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    pin_count: Optional[int] = None

class CommentPreview(BaseModel):
    text: str
    owner_username: str
    likes_count: Optional[int] = None

class ScrapedPost(BaseModel):
    source: Literal["instagram", "pinterest"]
    post_url: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    description: Optional[str] = None
    hashtags: Optional[List[str]] = None
    mentions: Optional[List[str]] = None
    tagged_users: Optional[List[str]] = None
    likes_count: Optional[int] = None
    comments_count: Optional[int] = None
    comments_preview: Optional[List[CommentPreview]] = None
    video_view_count: Optional[int] = None
    posted_date: Optional[datetime] = None
    created_at: Optional[str] = None
    location: Optional[str] = None
    domain: Optional[str] = None
    link: Optional[str] = None
    owner: Optional[OwnerInfo] = None
    board: Optional[BoardInfo] = None
    is_video: Optional[bool] = None
    accessibility_caption: Optional[str] = None
    scraped_date: datetime
    extraction_method: str
    extraction_error: Optional[str] = None

class ScrapeResponse(BaseModel):
    success: bool
    message: str
    total_posts: int
    posts: List[ScrapedPost]
    url: str
    platform: str
    scraped_at: datetime
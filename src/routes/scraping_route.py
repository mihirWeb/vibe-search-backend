from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any

from src.schemas.scraping_schema import ScrapeRequest, ScrapeResponse
from src.controller.scraping_controller import ScrapingService

# Remove the prefix here since it will be handled by the main router in __init__.py
router = APIRouter()

def get_scraping_service() -> ScrapingService:
    return ScrapingService()

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_social_media_posts(
    request: ScrapeRequest,
    scraping_service: ScrapingService = Depends(get_scraping_service)
) -> ScrapeResponse:
    """
    Scrape posts from Instagram or Pinterest
    
    - **url**: Instagram or Pinterest URL to scrape
    - **post_limit**: Number of posts to return (1-100)
    - **use_api**: Whether to use paid API for Pinterest (requires API key)
    
    Supported URL formats:
    - Instagram Profile: https://instagram.com/username
    - Instagram Hashtag: https://instagram.com/explore/tags/hashtag
    - Pinterest User: https://pinterest.com/username
    - Pinterest Board: https://pinterest.com/username/boardname
    """
    return await scraping_service.scrape_social_media_posts(request)

@router.get("/health")
async def scraping_health_check() -> Dict[str, Any]:
    """Health check endpoint for scraping service"""
    return {
        "status": "healthy",
        "service": "scraping",
        "timestamp": "2025-10-18T00:00:00Z"
    }
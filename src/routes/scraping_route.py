from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.scraping_schema import ScrapeRequest, ScrapeResponse, BatchScrapeRequest, BatchScrapeResponse
from src.controller.scraping_controller import ScrapingService
from src.config.database import get_db

router = APIRouter()

def get_scraping_service(db: AsyncSession = Depends(get_db)) -> ScrapingService:
    return ScrapingService(db)

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_social_media_posts(
    request: ScrapeRequest,
    save_to_db: bool = True,
    scraping_service: ScrapingService = Depends(get_scraping_service)
) -> ScrapeResponse:
    """
    Scrape posts from Instagram or Pinterest using Apify and save to database
    
    - **url**: Instagram or Pinterest URL to scrape
    - **post_limit**: Number of posts to return (1-100)
    - **use_api**: Always True for Apify (kept for compatibility)
    - **save_to_db**: Whether to save scraped data to database (default: True)
    
    Supported URL formats:
    - Instagram Profile: https://instagram.com/username
    - Instagram Hashtag: https://instagram.com/explore/tags/hashtag
    - Pinterest User: https://pinterest.com/username
    - Pinterest Board: https://pinterest.com/username/boardname
    
    Requires:
    - APIFY_TOKEN in environment variables
    """
    try:
        return await scraping_service.scrape_social_media_posts(request, save_to_db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@router.post("/batch", response_model=BatchScrapeResponse)
async def scrape_batch(
    request: BatchScrapeRequest,
    save_to_db: bool = True,
    scraping_service: ScrapingService = Depends(get_scraping_service)
) -> BatchScrapeResponse:
    """
    Scrape multiple URLs in batch and save to database
    
    - **urls**: List of Instagram or Pinterest URLs to scrape
    - **post_limit**: Number of posts to return per URL (1-100)
    - **save_to_db**: Whether to save scraped data to database (default: True)
    
    This endpoint processes multiple URLs and returns combined results.
    """
    try:
        return await scraping_service.scrape_batch(request, save_to_db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@router.get("/health")
async def scraping_health_check() -> Dict[str, Any]:
    """Health check endpoint for scraping service"""
    try:
        from src.config.settings import settings
        apify_status = "configured" if settings.APIFY_TOKEN else "not_configured"
        
        return {
            "status": "healthy",
            "service": "scraping",
            "apify_status": apify_status,
            "instagram_actor": settings.APIFY_INSTAGRAM_ACTOR_ID,
            "pinterest_actor": settings.APIFY_PINTEREST_ACTOR_ID,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "scraping",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
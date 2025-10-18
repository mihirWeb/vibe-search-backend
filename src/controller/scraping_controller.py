from typing import List, Dict, Optional
import re
from fastapi import HTTPException, status
from datetime import datetime

from src.services.instagram_scraper_service import InstagramScraper
from src.services.pinterest_scraper_service import PinterestScraper
from src.schemas.scraping_schema import ScrapeRequest, ScrapeResponse, ScrapedPost
from src.config.settings import settings

class ScrapingService:
    def __init__(self):
        self.instagram_scraper = InstagramScraper()
        self.pinterest_scraper = PinterestScraper()
    
    async def scrape_social_media_posts(self, request: ScrapeRequest) -> ScrapeResponse:
        """Main method to scrape posts from Instagram or Pinterest"""
        try:
            # Validate post limit
            if request.post_limit > settings.max_posts_per_request:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Post limit cannot exceed {settings.max_posts_per_request}"
                )
            
            # Determine platform and scrape accordingly
            platform = self._detect_platform(str(request.url))
            
            if platform == "instagram":
                posts_data = await self._scrape_instagram(str(request.url), request.post_limit)
            elif platform == "pinterest":
                posts_data = await self._scrape_pinterest(str(request.url), request.post_limit, request.use_api)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported platform. Only Instagram and Pinterest URLs are supported."
                )
            
            # Convert to Pydantic models
            scraped_posts = [ScrapedPost(**post) for post in posts_data]
            
            return ScrapeResponse(
                success=True,
                message=f"Successfully scraped {len(scraped_posts)} posts from {platform}",
                total_posts=len(scraped_posts),
                posts=scraped_posts,
                url=str(request.url),
                platform=platform,
                scraped_at=datetime.now()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while scraping: {str(e)}"
            )
    
    async def _scrape_instagram(self, url: str, post_limit: int) -> List[Dict]:
        """Scrape Instagram posts"""
        # Determine if it's a profile or hashtag URL
        if "/explore/tags/" in url:
            return await self.instagram_scraper.scrape_hashtag_posts(url, post_limit)
        else:
            return await self.instagram_scraper.scrape_profile_posts(url, post_limit)
    
    async def _scrape_pinterest(self, url: str, post_limit: int, use_api: bool = False) -> List[Dict]:
        """Scrape Pinterest posts"""
        # Determine if it's a board or user URL
        if self._is_pinterest_board_url(url):
            return await self.pinterest_scraper.scrape_board_posts(url, post_limit, use_api)
        else:
            return await self.pinterest_scraper.scrape_user_posts(url, post_limit, use_api)
    
    def _detect_platform(self, url: str) -> str:
        """Detect which platform the URL belongs to"""
        if "instagram.com" in url.lower():
            return "instagram"
        elif "pinterest.com" in url.lower():
            return "pinterest"
        else:
            raise ValueError("Unsupported platform")
    
    def _is_pinterest_board_url(self, url: str) -> bool:
        """Check if Pinterest URL is a board URL"""
        # Board URLs typically have format: pinterest.com/username/boardname
        pattern = r'pinterest\.com/([^/?]+)/([^/?]+)'
        match = re.search(pattern, url)
        if match:
            path_parts = url.split('/')
            return len([part for part in path_parts if part]) >= 4
        return False
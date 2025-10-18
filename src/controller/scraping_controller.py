from typing import List, Dict, Optional
import re
from fastapi import HTTPException, status
from datetime import datetime

from src.services.instagram_scraper_service import InstagramScraper
from src.services.pinterest_scraper_service import PinterestScraper
from src.schemas.scraping_schema import ScrapeRequest, ScrapeResponse, ScrapedPost, BatchScrapeRequest, BatchScrapeResponse
from src.config.settings import settings

class ScrapingService:
    def __init__(self):
        print("[Scraping Service] Initializing Instagram and Pinterest scrapers")
        self.instagram_scraper = InstagramScraper()
        self.pinterest_scraper = PinterestScraper()
        print("[Scraping Service] Scrapers initialized successfully")
    
    async def scrape_social_media_posts(self, request: ScrapeRequest) -> ScrapeResponse:
        """Main method to scrape posts from Instagram or Pinterest"""
        try:
            print(f"[Scraping Service] Starting scraping request for URL: {request.url}")
            print(f"[Scraping Service] Post limit: {request.post_limit}")
            
            # Validate post limit
            if request.post_limit > settings.MAX_POSTS_PER_REQUEST:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Post limit cannot exceed {settings.MAX_POSTS_PER_REQUEST}"
                )
            
            # Determine platform and scrape accordingly
            platform = self._detect_platform(str(request.url))
            print(f"[Scraping Service] Detected platform: {platform}")
            
            if platform == "instagram":
                print("[Scraping Service] Calling Instagram scraper")
                posts_data = await self._scrape_instagram(str(request.url), request.post_limit)
                estimated_cost = len(posts_data) * 0.0015  # $1.50 per 1000 results
                print(f"[Scraping Service] Instagram scraping completed. Posts: {len(posts_data)}, Cost: ${estimated_cost:.4f}")
            elif platform == "pinterest":
                print("[Scraping Service] Calling Pinterest scraper")
                posts_data = await self._scrape_pinterest(str(request.url), request.post_limit, request.use_api)
                estimated_cost = len(posts_data) * 0.001  # Estimated cost
                print(f"[Scraping Service] Pinterest scraping completed. Posts: {len(posts_data)}, Cost: ${estimated_cost:.4f}")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported platform. Only Instagram and Pinterest URLs are supported."
                )
            
            # Convert to Pydantic models
            print("[Scraping Service] Converting to Pydantic models")
            scraped_posts = [ScrapedPost(**post) for post in posts_data]
            print(f"[Scraping Service] Successfully converted {len(scraped_posts)} posts")
            
            response = ScrapeResponse(
                success=True,
                message=f"Successfully scraped {len(scraped_posts)} posts from {platform}",
                total_posts=len(scraped_posts),
                posts=scraped_posts,
                url=str(request.url),
                platform=platform,
                scraped_at=datetime.now(),
                estimated_cost=estimated_cost
            )
            
            print("[Scraping Service] Response created successfully")
            return response
            
        except HTTPException:
            print("[Scraping Service] HTTPException occurred, re-raising")
            raise
        except Exception as e:
            print(f"[Scraping Service] Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while scraping: {str(e)}"
            )
    
    async def scrape_batch(self, request: BatchScrapeRequest) -> BatchScrapeResponse:
        """Scrape multiple URLs in batch"""
        try:
            print(f"[Scraping Service] Starting batch scraping for {len(request.urls)} URLs")
            
            all_posts = []
            total_cost = 0.0
            errors = []
            urls_processed = 0
            
            for i, url in enumerate(request.urls):
                try:
                    print(f"[Scraping Service] Processing URL {i+1}/{len(request.urls)}: {url}")
                    
                    # Create individual scrape request
                    individual_request = ScrapeRequest(url=url, post_limit=request.post_limit)
                    response = await self.scrape_social_media_posts(individual_request)
                    
                    # Add posts to the batch result
                    all_posts.extend(response.posts)
                    total_cost += response.estimated_cost or 0
                    urls_processed += 1
                    
                    print(f"[Scraping Service] URL {i+1} completed successfully. Posts: {len(response.posts)}")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"[Scraping Service] Error processing URL {i+1}: {error_msg}")
                    errors.append({
                        "url": str(url),
                        "error": error_msg
                    })
            
            print(f"[Scraping Service] Batch scraping completed. Total posts: {len(all_posts)}, Total cost: ${total_cost:.4f}")
            
            return BatchScrapeResponse(
                success=len(errors) < len(request.urls),
                message=f"Batch scraping completed. Processed {urls_processed}/{len(request.urls)} URLs",
                total_posts=len(all_posts),
                posts=all_posts,
                urls_processed=urls_processed,
                urls_failed=len(errors),
                errors=errors,
                scraped_at=datetime.now(),
                total_cost=total_cost
            )
            
        except Exception as e:
            print(f"[Scraping Service] Batch scraping failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch scraping failed: {str(e)}"
            )
    
    async def _scrape_instagram(self, url: str, post_limit: int) -> List[Dict]:
        """Scrape Instagram posts"""
        print(f"[Scraping Service] Determining Instagram scraping method for: {url}")
        
        # Determine if it's a profile or hashtag URL
        if "/explore/tags/" in url:
            print("[Scraping Service] Using hashtag scraping method")
            return await self.instagram_scraper.scrape_hashtag_posts(url, post_limit)
        else:
            print("[Scraping Service] Using profile scraping method")
            return await self.instagram_scraper.scrape_profile_posts(url, post_limit)
    
    async def _scrape_pinterest(self, url: str, post_limit: int, use_api: bool = True) -> List[Dict]:
        """Scrape Pinterest posts"""
        print(f"[Scraping Service] Determining Pinterest scraping method for: {url}")
        
        # Determine if it's a board or user URL
        if self._is_pinterest_board_url(url):
            print("[Scraping Service] Using board scraping method")
            return await self.pinterest_scraper.scrape_board_posts(url, post_limit, use_api)
        else:
            print("[Scraping Service] Using user scraping method")
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
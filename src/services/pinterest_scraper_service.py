from apify_client import ApifyClient
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from fastapi import HTTPException, status
import re
from src.config.settings import settings

class PinterestScraper:
    def __init__(self):
        if not settings.APIFY_TOKEN:
            raise ValueError("APIFY_TOKEN is required for Pinterest scraping")
        
        self.client = ApifyClient(settings.APIFY_TOKEN)
        self.actor_id = settings.APIFY_PINTEREST_ACTOR_ID
        self.min_delay = settings.MIN_DELAY_BETWEEN_REQUESTS
        self.last_request_time = None
        
    async def scrape_board_posts(
        self, 
        board_url: str, 
        post_limit: int = 50,
        use_api: bool = True  # Apify is always API-based
    ) -> List[Dict]:
        """Scrape Pinterest board posts using Apify"""
        try:
            print(f"[Pinterest Scraper] Starting board scraping for: {board_url}")
            print(f"[Pinterest Scraper] Post limit: {post_limit}")
            
            await self._enforce_rate_limit()
            
            # Validate URL
            if not board_url.startswith(("https://pinterest.com/", "https://www.pinterest.com/")):
                raise ValueError("Invalid Pinterest URL")
            
            # Prepare Actor input
            run_input = {
                "startUrls": [board_url],
                "maxItems": post_limit,
                "endPage": 1,
                "proxy": {"useApifyProxy": True}
            }
            
            print(f"[Pinterest Scraper] Actor input: {run_input}")
            
            # Run the Actor and wait for it to finish
            print(f"[Pinterest Scraper] Starting Apify actor: {self.actor_id}")
            run = self.client.actor(self.actor_id).call(run_input=run_input)
            print(f"[Pinterest Scraper] Actor run completed. Run ID: {run.get('id', 'unknown')}")
            
            # Fetch results from the run's dataset
            scraped_data = []
            print(f"[Pinterest Scraper] Fetching data from dataset: {run['defaultDatasetId']}")
            
            for i, item in enumerate(self.client.dataset(run["defaultDatasetId"]).iterate_items()):
                print(f"[Pinterest Scraper] Processing item {i+1}")
                
                # Return raw data with minimal processing
                raw_post = {
                    "source": "pinterest",
                    "raw_data": item,
                    "scraped_date": datetime.now(),
                    "extraction_method": "apify"
                }
                
                scraped_data.append(raw_post)
                
                # Limit results
                if len(scraped_data) >= post_limit:
                    print(f"[Pinterest Scraper] Reached post limit: {post_limit}")
                    break
            
            print(f"[Pinterest Scraper] Successfully scraped {len(scraped_data)} posts")
            return scraped_data
            
        except Exception as e:
            print(f"[Pinterest Scraper] Error occurred: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scraping Pinterest board with Apify: {str(e)}"
            )
    
    async def scrape_user_posts(
        self, 
        user_url: str, 
        post_limit: int = 50,
        use_api: bool = True  # Apify is always API-based
    ) -> List[Dict]:
        """Scrape Pinterest user posts using Apify"""
        try:
            print(f"[Pinterest Scraper] Starting user scraping for: {user_url}")
            print(f"[Pinterest Scraper] Post limit: {post_limit}")
            
            await self._enforce_rate_limit()
            
            # Validate URL
            if not user_url.startswith(("https://pinterest.com/", "https://www.pinterest.com/")):
                raise ValueError("Invalid Pinterest URL")
            
            # Prepare Actor input
            run_input = {
                "startUrls": [user_url],
                "maxItems": post_limit,
                "endPage": 1,
                "proxy": {"useApifyProxy": True}
            }
            
            print(f"[Pinterest Scraper] Actor input: {run_input}")
            
            # Run the Actor and wait for it to finish
            print(f"[Pinterest Scraper] Starting Apify actor: {self.actor_id}")
            run = self.client.actor(self.actor_id).call(run_input=run_input)
            print(f"[Pinterest Scraper] Actor run completed. Run ID: {run.get('id', 'unknown')}")
            
            # Fetch results from the run's dataset
            scraped_data = []
            print(f"[Pinterest Scraper] Fetching data from dataset: {run['defaultDatasetId']}")
            
            for i, item in enumerate(self.client.dataset(run["defaultDatasetId"]).iterate_items()):
                print(f"[Pinterest Scraper] Processing item {i+1}")
                
                # Return raw data with minimal processing
                raw_post = {
                    "source": "pinterest",
                    "raw_data": item,
                    "scraped_date": datetime.now(),
                    "extraction_method": "apify"
                }
                
                scraped_data.append(raw_post)
                
                # Limit results
                if len(scraped_data) >= post_limit:
                    print(f"[Pinterest Scraper] Reached post limit: {post_limit}")
                    break
            
            print(f"[Pinterest Scraper] Successfully scraped {len(scraped_data)} posts")
            return scraped_data
            
        except Exception as e:
            print(f"[Pinterest Scraper] Error occurred: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scraping Pinterest user with Apify: {str(e)}"
            )
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        if self.last_request_time:
            time_since_last = (datetime.now() - self.last_request_time).total_seconds()
            if time_since_last < self.min_delay:
                delay = self.min_delay - time_since_last
                print(f"[Pinterest Scraper] Rate limiting: waiting {delay:.2f} seconds")
                await asyncio.sleep(delay)
        
        self.last_request_time = datetime.now()
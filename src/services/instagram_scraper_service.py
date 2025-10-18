from apify_client import ApifyClient
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import re
from src.config.settings import settings
from src.services.instagram_data_transform_service import DataTransformerService
from src.services.instagram_post_service import InstagramPostService

class InstagramScraper:
    def __init__(self, db_session: Optional[AsyncSession] = None):
        if not settings.APIFY_TOKEN:
            raise ValueError("APIFY_TOKEN is required for Instagram scraping")
        
        self.client = ApifyClient(settings.APIFY_TOKEN)
        self.actor_id = settings.APIFY_INSTAGRAM_ACTOR_ID
        self.min_delay = settings.MIN_DELAY_BETWEEN_REQUESTS
        self.last_request_time = None
        self.transformer = DataTransformerService()
        self.db_session = db_session
        
    async def scrape_profile_posts(
        self, 
        profile_url: str, 
        post_limit: int = 50,
        save_to_db: bool = True
    ) -> List[Dict]:
        """Scrape Instagram profile posts using Apify"""
        try:
            print(f"[Instagram Scraper] Starting profile scraping for: {profile_url}")
            print(f"[Instagram Scraper] Post limit: {post_limit}, Save to DB: {save_to_db}")
            
            await self._enforce_rate_limit()
            
            # Validate URL
            if not profile_url.startswith(("https://instagram.com/", "https://www.instagram.com/")):
                raise ValueError("Invalid Instagram URL")
            
            search_type = self._determine_instagram_search_type(profile_url)
            print(f"[Instagram Scraper] Detected search type: {search_type}")
            
            # Prepare Actor input
            run_input = {
                "directUrls": [profile_url],
                "resultsType": "posts",
                "resultsLimit": post_limit,
                "searchType": search_type,
                "searchLimit": 1,
                "useProxy": True
            }
            
            print(f"[Instagram Scraper] Actor input: {run_input}")
            
            # Run the Actor and wait for it to finish
            print(f"[Instagram Scraper] Starting Apify actor: {self.actor_id}")
            run = self.client.actor(self.actor_id).call(run_input=run_input)
            print(f"[Instagram Scraper] Actor run completed. Run ID: {run.get('id', 'unknown')}")
            
            # Fetch results from the run's dataset
            scraped_data = []
            structured_posts = []
            print(f"[Instagram Scraper] Fetching data from dataset: {run['defaultDatasetId']}")
            
            for i, item in enumerate(self.client.dataset(run["defaultDatasetId"]).iterate_items()):
                print(f"[Instagram Scraper] Processing item {i+1}")
                
                try:
                    # Transform raw data to structured format using only existing data
                    print(f"[Instagram Scraper] Transforming raw data for item {i+1}")
                    structured_post = self.transformer.transform_instagram_post(item)
                    structured_posts.append(structured_post)
                    
                    # Convert to dictionary for response
                    transformed_post = {
                        "source": "instagram",
                        "structured_data": structured_post.dict(),
                        "scraped_date": datetime.now(),
                        "extraction_method": "apify_with_transformation"
                    }
                    
                    scraped_data.append(transformed_post)
                    print(f"[Instagram Scraper] Successfully transformed item {i+1}")
                    
                except Exception as e:
                    print(f"[Instagram Scraper] Error transforming item {i+1}: {e}")
                    # Fallback to raw data if transformation fails
                    raw_post = {
                        "source": "instagram",
                        "raw_data": item,
                        "scraped_date": datetime.now(),
                        "extraction_method": "apify",
                        "transformation_error": str(e)
                    }
                    scraped_data.append(raw_post)
                
                # Limit results
                if len(scraped_data) >= post_limit:
                    print(f"[Instagram Scraper] Reached post limit: {post_limit}")
                    break
            
            # Save to database if requested and we have structured data
            if save_to_db and structured_posts and self.db_session:
                try:
                    print(f"[Instagram Scraper] Saving {len(structured_posts)} posts to database")
                    post_service = InstagramPostService(self.db_session)
                    saved_posts = await post_service.save_scraped_posts(structured_posts)
                    print(f"[Instagram Scraper] Successfully saved {len(saved_posts)} posts to database")
                except Exception as e:
                    print(f"[Instagram Scraper] Error saving to database: {e}")
                    # Continue even if database save fails
            
            print(f"[Instagram Scraper] Successfully scraped {len(scraped_data)} posts")
            return scraped_data
            
        except Exception as e:
            print(f"[Instagram Scraper] Error occurred: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scraping Instagram profile with Apify: {str(e)}"
            )
    
    async def scrape_hashtag_posts(
        self, 
        hashtag_url: str, 
        post_limit: int = 50,
        save_to_db: bool = True
    ) -> List[Dict]:
        """Scrape Instagram hashtag posts using Apify"""
        try:
            print(f"[Instagram Scraper] Starting hashtag scraping for: {hashtag_url}")
            print(f"[Instagram Scraper] Post limit: {post_limit}, Save to DB: {save_to_db}")
            
            await self._enforce_rate_limit()
            
            # Validate URL
            if not hashtag_url.startswith(("https://instagram.com/", "https://www.instagram.com/")):
                raise ValueError("Invalid Instagram hashtag URL")
            
            search_type = self._determine_instagram_search_type(hashtag_url)
            print(f"[Instagram Scraper] Detected search type: {search_type}")
            
            # Prepare Actor input
            run_input = {
                "directUrls": [hashtag_url],
                "resultsType": "posts",
                "resultsLimit": post_limit,
                "searchType": search_type,
                "searchLimit": 1,
                "useProxy": True
            }
            
            print(f"[Instagram Scraper] Actor input: {run_input}")
            
            # Run the Actor and wait for it to finish
            print(f"[Instagram Scraper] Starting Apify actor: {self.actor_id}")
            run = self.client.actor(self.actor_id).call(run_input=run_input)
            print(f"[Instagram Scraper] Actor run completed. Run ID: {run.get('id', 'unknown')}")
            
            # Fetch results from the run's dataset
            scraped_data = []
            structured_posts = []
            print(f"[Instagram Scraper] Fetching data from dataset: {run['defaultDatasetId']}")
            
            for i, item in enumerate(self.client.dataset(run["defaultDatasetId"]).iterate_items()):
                print(f"[Instagram Scraper] Processing item {i+1}")
                
                try:
                    # Transform raw data to structured format using only existing data
                    print(f"[Instagram Scraper] Transforming raw data for item {i+1}")
                    structured_post = self.transformer.transform_instagram_post(item)
                    structured_posts.append(structured_post)
                    
                    # Convert to dictionary for response
                    transformed_post = {
                        "source": "instagram",
                        "structured_data": structured_post.dict(),
                        "scraped_date": datetime.now(),
                        "extraction_method": "apify_with_transformation"
                    }
                    
                    scraped_data.append(transformed_post)
                    print(f"[Instagram Scraper] Successfully transformed item {i+1}")
                    
                except Exception as e:
                    print(f"[Instagram Scraper] Error transforming item {i+1}: {e}")
                    # Fallback to raw data if transformation fails
                    raw_post = {
                        "source": "instagram",
                        "raw_data": item,
                        "scraped_date": datetime.now(),
                        "extraction_method": "apify",
                        "transformation_error": str(e)
                    }
                    scraped_data.append(raw_post)
                
                # Limit results
                if len(scraped_data) >= post_limit:
                    print(f"[Instagram Scraper] Reached post limit: {post_limit}")
                    break
            
            # Save to database if requested and we have structured data
            if save_to_db and structured_posts and self.db_session:
                try:
                    print(f"[Instagram Scraper] Saving {len(structured_posts)} posts to database")
                    post_service = InstagramPostService(self.db_session)
                    saved_posts = await post_service.save_scraped_posts(structured_posts)
                    print(f"[Instagram Scraper] Successfully saved {len(saved_posts)} posts to database")
                except Exception as e:
                    print(f"[Instagram Scraper] Error saving to database: {e}")
                    # Continue even if database save fails
            
            print(f"[Instagram Scraper] Successfully scraped {len(scraped_data)} posts")
            return scraped_data
            
        except Exception as e:
            print(f"[Instagram Scraper] Error occurred: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scraping Instagram hashtag with Apify: {str(e)}"
            )
    
    def _determine_instagram_search_type(self, url: str) -> str:
        """Determine Instagram search type based on URL"""
        if "/explore/tags/" in url:
            return "hashtag"
        elif "/p/" in url:
            return "post"
        else:
            return "user"
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        if self.last_request_time:
            time_since_last = (datetime.now() - self.last_request_time).total_seconds()
            if time_since_last < self.min_delay:
                delay = self.min_delay - time_since_last
                print(f"[Instagram Scraper] Rate limiting: waiting {delay:.2f} seconds")
                await asyncio.sleep(delay)
        
        self.last_request_time = datetime.now()
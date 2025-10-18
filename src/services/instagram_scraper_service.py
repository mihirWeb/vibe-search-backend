import instaloader
from datetime import datetime
import time
import random
from typing import List, Dict, Optional, Union
import asyncio
from fastapi import HTTPException, status
import re
from src.config.settings import settings

class InstagramScraper:
    def __init__(self):
        self.L = instaloader.Instaloader()
        self.min_delay = settings.MIN_DELAY_BETWEEN_REQUESTS
        self.last_request_time = None
        
    async def scrape_profile_posts(
        self, 
        profile_url: str, 
        post_limit: int = 50
    ) -> List[Dict]:
        """Scrape posts from an Instagram profile URL"""
        try:
            username = self._extract_username_from_url(profile_url)
            if not username:
                raise ValueError("Invalid Instagram profile URL")
            
            await self._enforce_rate_limit()
            
            profile = instaloader.Profile.from_username(self.L.context, username)
            posts = profile.get_posts()
            scraped_data = []
            
            for post in posts:
                if len(scraped_data) >= post_limit:
                    break
                
                post_data = await self._extract_post_data(post)
                scraped_data.append(post_data)
                
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            return scraped_data
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scraping Instagram profile: {str(e)}"
            )
    
    async def scrape_hashtag_posts(
        self, 
        hashtag_url: str, 
        post_limit: int = 50
    ) -> List[Dict]:
        """Scrape posts from an Instagram hashtag URL"""
        try:
            hashtag = self._extract_hashtag_from_url(hashtag_url)
            if not hashtag:
                raise ValueError("Invalid Instagram hashtag URL")
            
            await self._enforce_rate_limit()
            
            hashtag_obj = instaloader.Hashtag.from_name(self.L.context, hashtag)
            posts = hashtag_obj.get_posts()
            scraped_data = []
            
            for post in posts:
                if len(scraped_data) >= post_limit:
                    break
                
                post_data = await self._extract_post_data(post)
                scraped_data.append(post_data)
                
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            return scraped_data
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scraping Instagram hashtag: {str(e)}"
            )
    
    async def _extract_post_data(self, post) -> Dict:
        """Extract all required metadata from a post"""
        try:
            comments_count = post.comments
            likes_count = post.likes
            
            location_name = None
            if post.location:
                location_name = post.location.name
            
            comments_preview = []
            try:
                for comment in post.get_comments()[:3]:
                    comments_preview.append({
                        "text": comment.text,
                        "owner_username": comment.owner.username,
                        "likes_count": comment.likes_count
                    })
            except:
                pass
            
            post_data = {
                "source": "instagram",
                "post_url": f"https://instagram.com/p/{post.shortcode}/",
                "image_url": post.url,
                "caption": post.caption,
                "title": post.title,
                "accessibility_caption": post.accessibility_caption,
                "hashtags": post.caption_hashtags,
                "mentions": post.caption_mentions,
                "tagged_users": post.tagged_users,
                "likes_count": likes_count,
                "comments_count": comments_count,
                "comments_preview": comments_preview,
                "posted_date": post.date_utc,
                "is_video": post.is_video,
                "video_url": post.video_url if post.is_video else None,
                "video_view_count": post.video_view_count if post.is_video else None,
                "location": location_name,
                "owner": {
                    "username": post.owner_username,
                    "user_id": post.owner_id,
                    "profile_url": f"https://instagram.com/{post.owner_username}/"
                },
                "scraped_date": datetime.now(),
                "extraction_method": "instaloader"
            }
            
            return post_data
            
        except Exception as e:
            return {
                "source": "instagram",
                "post_url": f"https://instagram.com/p/{post.shortcode}/",
                "image_url": post.url,
                "caption": post.caption if hasattr(post, 'caption') else None,
                "posted_date": post.date_utc,
                "owner": {
                    "username": post.owner_username,
                    "user_id": post.owner_id
                },
                "scraped_date": datetime.now(),
                "extraction_method": "instaloader",
                "extraction_error": str(e)
            }
    
    def _extract_username_from_url(self, url: str) -> Optional[str]:
        """Extract username from Instagram profile URL"""
        patterns = [
            r'instagram\.com/([^/?]+)',
            r'instagram\.com/([^/?]+)/'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                username = match.group(1)
                return username.rstrip('/')
        return None
    
    def _extract_hashtag_from_url(self, url: str) -> Optional[str]:
        """Extract hashtag from Instagram hashtag URL"""
        patterns = [
            r'instagram\.com/explore/tags/([^/?]+)',
            r'instagram\.com/explore/tags/([^/?]+)/'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                hashtag = match.group(1)
                return hashtag.rstrip('/')
        return None
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        if self.last_request_time:
            time_since_last = (datetime.now() - self.last_request_time).total_seconds()
            if time_since_last < self.min_delay:
                delay = self.min_delay - time_since_last + random.uniform(0, 1)
                await asyncio.sleep(delay)
        
        self.last_request_time = datetime.now()
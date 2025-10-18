from typing import List, Dict, Optional
import asyncio
import re
import json
import requests
import random
from datetime import datetime
from fastapi import HTTPException, status
from playwright.async_api import async_playwright
from src.config.settings import settings

class PinterestScraper:
    def __init__(self, scrape_creators_api_key: Optional[str] = None):
        self.scrape_creators_api_key = settings.SCRAPE_CREATORS_API_KEY
        self.min_delay = settings.MIN_DELAY_BETWEEN_REQUESTS
        self.last_request_time = None
        
    async def scrape_board_posts(
        self, 
        board_url: str, 
        post_limit: int = 50,
        use_api: bool = False
    ) -> List[Dict]:
        """Scrape posts from a Pinterest board URL"""
        try:
            if use_api and self.scrape_creators_api_key:
                return await self._scrape_board_with_api(board_url, post_limit)
            else:
                return await self._scrape_board_with_playwright(board_url, post_limit)
                
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scraping Pinterest board: {str(e)}"
            )
    
    async def scrape_user_posts(
        self, 
        user_url: str, 
        post_limit: int = 50,
        use_api: bool = False
    ) -> List[Dict]:
        """Scrape posts from a Pinterest user URL"""
        try:
            if use_api and self.scrape_creators_api_key:
                return await self._scrape_user_with_api(user_url, post_limit)
            else:
                return await self._scrape_user_with_playwright(user_url, post_limit)
                
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scraping Pinterest user: {str(e)}"
            )
    
    async def _scrape_board_with_playwright(self, board_url: str, post_limit: int) -> List[Dict]:
        """Scrape Pinterest board using Playwright (free option)"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                await page.goto(board_url, wait_until='networkidle')
                await asyncio.sleep(3)
                
                # Scroll to load more pins
                scroll_count = min(post_limit // 10 + 2, 10)
                for _ in range(scroll_count):
                    await page.mouse.wheel(0, 1000)
                    await asyncio.sleep(2)
                
                # Extract pin data
                pins = await page.query_selector_all("div[data-test-id='pinWrapper']")
                scraped_data = []
                
                for pin in pins[:post_limit]:
                    try:
                        pin_data = await self._extract_pin_data_playwright(pin)
                        scraped_data.append(pin_data)
                    except Exception as e:
                        print(f"Error extracting pin data: {e}")
                
                return scraped_data
            finally:
                await browser.close()
    
    async def _scrape_user_with_playwright(self, user_url: str, post_limit: int) -> List[Dict]:
        """Scrape Pinterest user using Playwright (free option)"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                await page.goto(user_url, wait_until='networkidle')
                await asyncio.sleep(3)
                
                # Click on "Created" tab if available
                try:
                    created_tab = await page.query_selector("text=Created")
                    if created_tab:
                        await created_tab.click()
                        await asyncio.sleep(2)
                except:
                    pass
                
                # Scroll to load more pins
                scroll_count = min(post_limit // 10 + 2, 10)
                for _ in range(scroll_count):
                    await page.mouse.wheel(0, 1000)
                    await asyncio.sleep(2)
                
                # Extract pin data
                pins = await page.query_selector_all("div[data-test-id='pinWrapper']")
                scraped_data = []
                
                for pin in pins[:post_limit]:
                    try:
                        pin_data = await self._extract_pin_data_playwright(pin)
                        scraped_data.append(pin_data)
                    except Exception as e:
                        print(f"Error extracting pin data: {e}")
                
                return scraped_data
            finally:
                await browser.close()
    
    async def _extract_pin_data_playwright(self, pin) -> Dict:
        """Extract pin data using Playwright"""
        try:
            title_element = await pin.query_selector("a[aria-label]")
            img_element = await pin.query_selector("img")
            
            title = await title_element.get_attribute("aria-label") if title_element else None
            img_src = await img_element.get_attribute("src") if img_element else None
            pin_url = await title_element.get_attribute("href") if title_element else None
            
            # Get high-resolution image URL
            high_res_img = None
            if img_src:
                high_res_img = img_src.replace("/236x/", "/originals/")
            
            pin_data = {
                "source": "pinterest",
                "title": title,
                "image_url": high_res_img or img_src,
                "post_url": f"https://pinterest.com{pin_url}" if pin_url else None,
                "scraped_date": datetime.now(),
                "extraction_method": "playwright"
            }
            
            return pin_data
            
        except Exception as e:
            return {
                "source": "pinterest",
                "scraped_date": datetime.now(),
                "extraction_method": "playwright",
                "extraction_error": str(e)
            }
    
    async def _scrape_board_with_api(self, board_url: str, post_limit: int) -> List[Dict]:
        """Scrape Pinterest board using Scrape Creators API (paid option)"""
        board_id = self._extract_board_id_from_url(board_url)
        if not board_id:
            raise ValueError("Invalid Pinterest board URL")
        
        await self._enforce_rate_limit()
        
        api_url = f"https://api.scrapecreators.com/v1/pinterest/board/{board_id}"
        headers = {"x-api-key": self.scrape_creators_api_key}
        params = {"limit": post_limit}
        
        response = requests.get(api_url, headers=headers, params=params)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"API request failed: {response.text}"
            )
        
        data = response.json()
        
        if not data.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"API returned error: {data.get('error', 'Unknown error')}"
            )
        
        scraped_data = []
        for pin in data.get("pins", [])[:post_limit]:
            pin_data = self._transform_api_pin_data(pin)
            scraped_data.append(pin_data)
        
        return scraped_data
    
    async def _scrape_user_with_api(self, user_url: str, post_limit: int) -> List[Dict]:
        """Scrape Pinterest user using Scrape Creators API (paid option)"""
        username = self._extract_username_from_url(user_url)
        if not username:
            raise ValueError("Invalid Pinterest user URL")
        
        await self._enforce_rate_limit()
        
        api_url = f"https://api.scrapecreators.com/v1/pinterest/user/{username}"
        headers = {"x-api-key": self.scrape_creators_api_key}
        params = {"limit": post_limit}
        
        response = requests.get(api_url, headers=headers, params=params)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"API request failed: {response.text}"
            )
        
        data = response.json()
        
        if not data.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"API returned error: {data.get('error', 'Unknown error')}"
            )
        
        scraped_data = []
        for pin in data.get("pins", [])[:post_limit]:
            pin_data = self._transform_api_pin_data(pin)
            scraped_data.append(pin_data)
        
        return scraped_data
    
    def _transform_api_pin_data(self, pin: Dict) -> Dict:
        """Transform Scrape Creators API response to our format"""
        try:
            image_url = None
            if "images" in pin and "orig" in pin["images"]:
                image_url = pin["images"]["orig"].get("url")
            
            pinner = pin.get("pinner", {})
            owner = {
                "username": pinner.get("username"),
                "user_id": pinner.get("id"),
                "full_name": pinner.get("full_name"),
                "follower_count": pinner.get("follower_count")
            }
            
            board = pin.get("board", {})
            board_info = {
                "name": board.get("name"),
                "url": board.get("url"),
                "pin_count": board.get("pin_count")
            }
            
            pin_data = {
                "source": "pinterest",
                "post_url": pin.get("url"),
                "image_url": image_url,
                "title": pin.get("title"),
                "description": pin.get("description"),
                "created_at": pin.get("created_at"),
                "owner": owner,
                "board": board_info,
                "domain": pin.get("domain"),
                "link": pin.get("link"),
                "scraped_date": datetime.now(),
                "extraction_method": "scrape_creators_api"
            }
            
            return pin_data
            
        except Exception as e:
            return {
                "source": "pinterest",
                "scraped_date": datetime.now(),
                "extraction_method": "scrape_creators_api",
                "extraction_error": str(e),
                "raw_data": pin
            }
    
    def _extract_board_id_from_url(self, url: str) -> Optional[str]:
        """Extract board ID from Pinterest board URL"""
        patterns = [
            r'pinterest\.com/([^/?]+)/([^/?]+)',
            r'pinterest\.com/([^/?]+)/([^/?]+)/'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return f"{match.group(1)}/{match.group(2).rstrip('/')}"
        return None
    
    def _extract_username_from_url(self, url: str) -> Optional[str]:
        """Extract username from Pinterest user URL"""
        patterns = [
            r'pinterest\.com/([^/?]+)',
            r'pinterest\.com/([^/?]+)/'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                username = match.group(1)
                return username.rstrip('/')
        return None
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        if self.last_request_time:
            time_since_last = (datetime.now() - self.last_request_time).total_seconds()
            if time_since_last < self.min_delay:
                delay = self.min_delay - time_since_last + random.uniform(0, 1)
                await asyncio.sleep(delay)
        
        self.last_request_time = datetime.now()
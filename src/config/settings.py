from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # App Info
    APP_NAME: str = "Vibe Search Backend"
    VERSION: str = "1.0.0"
    
    # CORS  
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000"]
    
    # Environment
    ENVIRONMENT: str = "development"
    SECRET_KEY: str

    # Scraping Configuration
    APIFY_TOKEN: Optional[str] = None
    MIN_DELAY_BETWEEN_REQUESTS: int = 2
    MAX_POSTS_PER_REQUEST: int = 100
    
    # Apify Actor IDs
    APIFY_INSTAGRAM_ACTOR_ID: str = "apify/instagram-scraper"
    APIFY_PINTEREST_ACTOR_ID: str = "epctex/pinterest-scraper"
    
    class Config:
        env_file = ".env"  # This tells Pydantic to read from .env file


settings = Settings()
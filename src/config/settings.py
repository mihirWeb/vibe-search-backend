from pydantic_settings import BaseSettings
from typing import List


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
    SCRAPE_CREATORS_API_KEY: str
    MIN_DELAY_BETWEEN_REQUESTS: int = 2
    MAX_POSTS_PER_REQUEST: int = 100
    
    class Config:
        env_file = ".env"  # This tells Pydantic to read from .env file


settings = Settings()
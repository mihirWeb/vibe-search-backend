from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "vibe-search"
    DB_PORT: int = 5433
    
    # Backend Configuration
    BACKEND_PORT: int = 8000

    # App Info
    APP_NAME: str = "Vibe Search Backend"
    VERSION: str = "1.0.0"
    
    # CORS
    ALLOWED_HOSTS: str = '["*"]'
    
    @property
    def allowed_hosts_list(self) -> List[str]:
        """Parse ALLOWED_HOSTS from JSON string to list"""
        try:
            return json.loads(self.ALLOWED_HOSTS)
        except (json.JSONDecodeError, TypeError):
            return ["*"]
    
    # Environment
    ENVIRONMENT: str = "development"
    SECRET_KEY: str

    # Scraping Configuration
    APIFY_TOKEN: str
    MIN_DELAY_BETWEEN_REQUESTS: int = 2
    MAX_POSTS_PER_REQUEST: int = 100
    
    # Apify Actor IDs
    APIFY_INSTAGRAM_ACTOR_ID: str = "apify/instagram-scraper"
    APIFY_PINTEREST_ACTOR_ID: str = "epctex/pinterest-scraper"
    
    # PyTorch Configuration
    KMP_DUPLICATE_LIB_OK: str = "TRUE"
    OMP_NUM_THREADS: str = "1"
    TORCH_HOME: str = "./torch_cache"
    
    # Image Processing Configuration
    USE_MOCK_IMAGE_SERVICE: bool = False
    IMAGE_PROCESSING_TIMEOUT: int = 30
    MAX_IMAGE_SIZE: int = 2048
    
    # Hugging Face
    HUGGINGFACE_TOKEN: str
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
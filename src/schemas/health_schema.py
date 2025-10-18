from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str
    database: str
    app_name: str
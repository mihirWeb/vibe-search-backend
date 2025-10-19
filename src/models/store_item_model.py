from sqlalchemy import Column, Integer, String, Text, DECIMAL, Boolean, Date, TIMESTAMP, JSON
from pgvector.sqlalchemy import Vector
from datetime import datetime

from src.config.database import Base


class StoreItem(Base):
    """StoreItem model representing products imported from CSV with embeddings"""
    
    __tablename__ = "store_items"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # CSV fields
    sku_id = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    slug = Column(String(500), nullable=False)
    category = Column(String(100), index=True)
    sub_category = Column(String(100))
    brand_name = Column(String(100), index=True)
    product_type = Column(String(100))
    gender = Column(String(50))
    colorways = Column(Text)
    brand_sku = Column(String(100))
    model = Column(String(200))
    lowest_price = Column(DECIMAL(10, 2))
    description = Column(Text)
    
    # Boolean flags
    is_d2c = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_certificate_required = Column(Boolean, default=False)
    
    # URLs and metadata
    featured_image = Column(String(1000))
    pdp_url = Column(String(1000))
    
    # Inventory and engagement
    quantity_left = Column(Integer, default=0)
    wishlist_num = Column(Integer, default=0)
    stock_claimed_percent = Column(Integer, default=0)
    discount_percentage = Column(Integer, default=0)
    
    # Additional metadata
    note = Column(Text)
    tags = Column(Text)
    
    # Dates from CSV
    release_date = Column(Date, nullable=True)
    csv_created_at = Column(TIMESTAMP, nullable=True)
    csv_updated_at = Column(TIMESTAMP, nullable=True)
    
    # Vector embeddings
    textual_embedding = Column(Vector(768), nullable=True)  # all-MiniLM-L6-v2
    visual_embedding = Column(Vector(512), nullable=True)  # Open-CLIP
    multimodal_embedding = Column(Vector(1280), nullable=True)  # Combined
    
    # Object detection metadata
    object_embeddings = Column(JSON, nullable=True)  # Store YOLO object detections
    
    # System timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<StoreItem(id={self.id}, sku_id='{self.sku_id}', title='{self.title}')>"
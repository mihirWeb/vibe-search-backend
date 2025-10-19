from sqlalchemy import Column, Integer, String, Text, ARRAY, TIMESTAMP, JSON
from pgvector.sqlalchemy import Vector  # Import from pgvector instead
from sqlalchemy.orm import relationship
from datetime import datetime

from src.config.database import Base


class Product(Base):
    """Product model representing a fashion product extracted from Instagram posts"""
    
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    image_url = Column(Text, nullable=False)
    source_url = Column(Text, nullable=False)
    brand = Column(String(255), nullable=True, index=True)
    category = Column(String(100), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Style attributes
    style = Column(ARRAY(String), nullable=True)
    colors = Column(ARRAY(String), nullable=True)
    
    # Caption from Instagram post
    caption = Column(Text, nullable=True)
    
    # Vector embeddings - using pgvector's Vector type
    embedding = Column(Vector(512), nullable=True)  # Open-CLIP visual embedding
    text_embedding = Column(Vector(384), nullable=True)  # Sentence-Transformers text embedding
    
    # Metadata - RENAMED from 'metadata' to 'meta_info' to avoid conflict with SQLAlchemy
    meta_info = Column(JSON, nullable=True)
    
    # Relationships
    items = relationship("ProductItem", back_populates="product", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', brand='{self.brand}')>"
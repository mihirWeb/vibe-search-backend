from sqlalchemy import Column, Integer, String, Text, ARRAY, Float, ForeignKey, TIMESTAMP, JSON
from pgvector.sqlalchemy import Vector  # Import from pgvector instead
from sqlalchemy.orm import relationship
from datetime import datetime

from src.config.database import Base


class ProductItem(Base):
    """ProductItem model representing individual items detected in a product image"""
    
    __tablename__ = "product_items"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    
    name = Column(String(255), nullable=False)
    brand = Column(String(255), nullable=True)
    category = Column(String(100), nullable=False)
    style = Column(ARRAY(String), nullable=True)
    colors = Column(ARRAY(String), nullable=True)
    product_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Visual features
    visual_features = Column(JSON, nullable=True)
    
    # Vector embeddings - using pgvector's Vector type
    embedding = Column(Vector(512), nullable=True)  # Open-CLIP visual embedding
    text_embedding = Column(Vector(384), nullable=True)  # Sentence-Transformers text embedding
    
    # Detection information
    bounding_box = Column(ARRAY(Float), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Metadata - RENAMED from 'metadata' to 'meta_info'
    meta_info = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    product = relationship("Product", back_populates="items")
    
    def __repr__(self):
        return f"<ProductItem(id={self.id}, name='{self.name}', product_id={self.product_id})>"
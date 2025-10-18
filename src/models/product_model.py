from sqlalchemy import Column, Integer, String, Text, ARRAY, TIMESTAMP, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from pgvector.sqlalchemy import Vector
from src.config.database import Base

class Product(Base):
    """Product table with minimal information from Instagram posts"""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    image_url = Column(String, nullable=False)  # Only the display URL from Instagram
    source_url = Column(String)  # Original Instagram post URL
    brand = Column(String)
    category = Column(String)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata and embeddings
    style = Column(ARRAY(String))
    colors = Column(ARRAY(String))
    caption = Column(Text)  # Store the original Instagram caption
    embedding = Column(Vector(512))  # Open-CLIP visual embedding for the product image
    text_embedding = Column(Vector(384))  # Sentence-Transformers text embedding (384 for all-MiniLM-L6-v2)
    meta_info = Column(JSONB)  # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy reserved name)
    
    # Relationships
    items = relationship("ProductItem", back_populates="product", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Product(id={self.id}, name={self.name}, brand={self.brand})>"
    
    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'image_url': self.image_url,
            'source_url': self.source_url,
            'brand': self.brand,
            'category': self.category,
            'style': self.style,
            'colors': self.colors,
            'caption': self.caption,
            'embedding': self.embedding,
            'text_embedding': self.text_embedding,
            'meta_info': self.meta_info,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'items': [item.to_dict() for item in self.items] if self.items else []
        }
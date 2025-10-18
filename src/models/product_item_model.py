from sqlalchemy import Column, Integer, String, Text, ARRAY, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.config.database import Base

class ProductItem(Base):
    """Individual wearable items within a product"""
    __tablename__ = "product_items"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    brand = Column(String)
    category = Column(String)
    style = Column(ARRAY(String))
    colors = Column(ARRAY(String))
    product_type = Column(String)
    description = Column(Text)
    
    # Visual features
    visual_features = Column(JSONB)
    embedding = Column(Vector(512))  # Open-CLIP visual embedding for this specific item
    text_embedding = Column(Vector(384))  # Sentence-Transformers text embedding
    bounding_box = Column(JSONB)  # Location of item in the product image
    
    # Metadata
    confidence_score = Column(Float)
    meta_info = Column(JSONB)  # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy reserved name)
    
    # Relationships
    product = relationship("Product", back_populates="items")
    
    def __repr__(self):
        return f"<ProductItem(id={self.id}, name={self.name}, product_id={self.product_id})>"
    
    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            'id': self.id,
            'product_id': self.product_id,
            'name': self.name,
            'brand': self.brand,
            'category': self.category,
            'style': self.style,
            'colors': self.colors,
            'product_type': self.product_type,
            'description': self.description,
            'visual_features': self.visual_features,
            'embedding': self.embedding,
            'text_embedding': self.text_embedding,
            'bounding_box': self.bounding_box,
            'confidence_score': self.confidence_score,
            'meta_info': self.meta_info
        }
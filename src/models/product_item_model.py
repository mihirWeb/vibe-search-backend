from sqlalchemy import Column, Integer, String, Text, ARRAY, Float, ForeignKey, TIMESTAMP, JSON, Enum as SQLEnum
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

from src.config.database import Base


class Category(str, Enum):
    """Product category enum"""
    APPAREL = "Apparel"
    ACCESSORIES = "Accessories"
    HANDBAGS = "handbags"
    STREETWEAR = "streetwear"
    SNEAKERS = "sneakers"
    WATCHES = "watches"
    COLLECTIBLES = "collectibles"


class SubCategory(str, Enum):
    """Product sub-category enum"""
    T_SHIRT = "T-Shirt"
    TUMBLER = "Tumbler"
    SWEATSHIRT = "Sweatshirt"
    FIGURES = "Figures"
    FIGURINES = "Figurines"
    ACCESSORIES_HANDBAGS = "accessories handbags"
    APPAREL_STREETWEAR = "apparel streetwear"
    SHOES_SNEAKERS = "shoes sneakers"
    ACCESSORIES_WATCHES = "accessories watches"
    COLLECTIBLES_COLLECTIBLES = "collectibles"


class ProductType(str, Enum):
    """Product type enum"""
    CLOTHING = "clothing"
    ACCESSORIES = "accessories"
    HANDBAGS = "handbags"
    STREETWEAR = "streetwear"
    SNEAKERS = "sneakers"
    WATCHES = "watches"
    COLLECTIBLES = "collectibles"


class Gender(str, Enum):
    """Gender enum"""
    MALE = "male"
    UNISEX = "unisex"
    FEMALE = "female"


class ProductItem(Base):
    """ProductItem model representing individual items detected in a product image"""
    
    __tablename__ = "product_items"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    
    name = Column(String(255), nullable=False)
    brand = Column(String(255), nullable=True)
    
    # Updated category fields with enums
    category = Column(String(100), nullable=False)  # Will store Category enum value
    sub_category = Column(String(100), nullable=True)  # New field for SubCategory
    product_type = Column(String(100), nullable=False)  # Will store ProductType enum value
    gender = Column(String(20), nullable=True)  # New field for Gender
    
    style = Column(ARRAY(String), nullable=True)
    colors = Column(ARRAY(String), nullable=True)
    description = Column(Text, nullable=True)
    
    # Visual features
    visual_features = Column(JSON, nullable=True)
    
    # Vector embeddings
    embedding = Column(Vector(512), nullable=True)
    text_embedding = Column(Vector(384), nullable=True)
    
    # Detection information
    bounding_box = Column(ARRAY(Float), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Metadata
    meta_info = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    product = relationship("Product", back_populates="items")
    
    def __repr__(self):
        return f"<ProductItem(id={self.id}, name='{self.name}', product_id={self.product_id})>"
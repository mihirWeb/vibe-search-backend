from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from src.config.database import Base

class InstagramPost(Base):
    __tablename__ = "instagram_posts"
    
    # Core identifiers
    id = Column(String, primary_key=True, index=True)  # Instagram post ID
    type = Column(String, nullable=False)  # Image, Video, Sidecar
    short_code = Column(String, nullable=False, index=True)
    
    # URLs and media
    url = Column(String, nullable=False)
    display_url = Column(String, nullable=False)
    images = Column(JSON, default=list)  # Array of image URLs
    
    # Content and metadata
    caption = Column(Text, nullable=True)
    alt = Column(Text, nullable=True)
    
    # Engagement metrics
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    
    # Comments data
    first_comment = Column(Text, nullable=True)
    latest_comments = Column(JSON, default=list)  # Array of comment objects
    
    # Temporal information
    timestamp = Column(DateTime, nullable=False)
    
    # Dimensions
    dimensions_height = Column(Integer, default=0)
    dimensions_width = Column(Integer, default=0)
    
    # Owner information
    owner_full_name = Column(String, nullable=False)
    owner_username = Column(String, nullable=False, index=True)
    owner_id = Column(String, nullable=False, index=True)
    
    # Tags and mentions
    hashtags = Column(JSON, default=list)  # Array of hashtags
    mentions = Column(JSON, default=list)  # Array of mentions
    tagged_users = Column(JSON, default=list)  # Array of tagged user objects
    
    # Additional metadata
    is_comments_disabled = Column(Boolean, default=False)
    input_url = Column(String, nullable=False)
    is_sponsored = Column(Boolean, default=False)
    
    # Carousel specific
    child_posts = Column(JSON, default=list)  # Array of child post objects
    
    # Custom fields for search and categorization
    extracted_keywords = Column(JSON, default=list)
    detected_objects = Column(JSON, default=list)
    dominant_colors = Column(JSON, default=list)
    style_attributes = Column(JSON, default=list)
    product_type = Column(String, nullable=True)
    brand_name = Column(String, nullable=True)
    category = Column(String, nullable=True)
    
    # System metadata
    scraped_date = Column(DateTime, default=datetime.utcnow)
    primary_image_url = Column(String, nullable=False)
    
    # Additional tracking fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<InstagramPost(id={self.id}, owner_username={self.owner_username}, type={self.type})>"
    
    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            'id': self.id,
            'type': self.type,
            'short_code': self.short_code,
            'url': self.url,
            'display_url': self.display_url,
            'images': self.images,
            'caption': self.caption,
            'alt': self.alt,
            'likes_count': self.likes_count,
            'comments_count': self.comments_count,
            'first_comment': self.first_comment,
            'latest_comments': self.latest_comments,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'dimensions_height': self.dimensions_height,
            'dimensions_width': self.dimensions_width,
            'owner_full_name': self.owner_full_name,
            'owner_username': self.owner_username,
            'owner_id': self.owner_id,
            'hashtags': self.hashtags,
            'mentions': self.mentions,
            'tagged_users': self.tagged_users,
            'is_comments_disabled': self.is_comments_disabled,
            'input_url': self.input_url,
            'is_sponsored': self.is_sponsored,
            'child_posts': self.child_posts,
            'extracted_keywords': self.extracted_keywords,
            'detected_objects': self.detected_objects,
            'dominant_colors': self.dominant_colors,
            'style_attributes': self.style_attributes,
            'product_type': self.product_type,
            'brand_name': self.brand_name,
            'category': self.category,
            'scraped_date': self.scraped_date.isoformat() if self.scraped_date else None,
            'primary_image_url': self.primary_image_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
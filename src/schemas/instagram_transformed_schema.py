from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class PostType(str, Enum):
    IMAGE = "Image"
    VIDEO = "Video"
    SIDECAR = "Sidecar"

class CommentOwnerSchema(BaseModel):
    id: str = Field(..., description="ID of the comment owner")
    username: str = Field(..., description="Username of the comment owner")
    is_verified: bool = Field(False, description="Whether the comment owner is verified")
    profile_pic_url: Optional[str] = Field(None, description="URL to the comment owner's profile picture")

class CommentReplySchema(BaseModel):
    id: str = Field(..., description="ID of the reply")
    text: str = Field(..., description="Text content of the reply")
    timestamp: datetime = Field(..., description="When the reply was created")
    repliesCount: int = Field(0, description="Number of replies to the reply")
    likesCount: int = Field(0, description="Number of likes on the reply")
    owner: CommentOwnerSchema = Field(..., description="Owner of the reply")

class CommentSchema(BaseModel):
    id: str = Field(..., description="ID of the comment")
    text: str = Field(..., description="Text content of the comment")
    ownerUsername: str = Field(..., description="Username of the comment owner")
    ownerProfilePicUrl: Optional[str] = Field(None, description="URL to the comment owner's profile picture")
    timestamp: datetime = Field(..., description="When the comment was created")
    repliesCount: int = Field(0, description="Number of replies to the comment")
    replies: List[CommentReplySchema] = Field(default_factory=list, description="Replies to the comment")
    likesCount: int = Field(0, description="Number of likes on the comment")
    owner: CommentOwnerSchema = Field(..., description="Owner of the comment")

class TaggedUserSchema(BaseModel):
    full_name: str = Field(..., description="Full name of the tagged user")
    id: str = Field(..., description="ID of the tagged user")
    is_verified: bool = Field(False, description="Whether the tagged user is verified")
    profile_pic_url: Optional[str] = Field(None, description="URL to the tagged user's profile picture")
    username: str = Field(..., description="Username of the tagged user")

class ChildPostSchema(BaseModel):
    id: str = Field(..., description="ID of the child post")
    type: str = Field(..., description="Type of the child post")
    caption: Optional[str] = Field(None, description="Caption of the child post")
    hashtags: List[str] = Field(default_factory=list, description="Hashtags in the child post")
    mentions: List[str] = Field(default_factory=list, description="Mentions in the child post")
    url: str = Field(..., description="URL of the child post")
    commentsCount: int = Field(0, description="Number of comments on the child post")
    firstComment: Optional[str] = Field(None, description="First comment on the child post")
    latestComments: List[CommentSchema] = Field(default_factory=list, description="Latest comments on the child post")
    dimensionsHeight: int = Field(0, description="Height of the child post")
    dimensionsWidth: int = Field(0, description="Width of the child post")
    displayUrl: str = Field(..., description="Display URL of the child post")
    images: List[str] = Field(default_factory=list, description="Images in the child post")
    alt: Optional[str] = Field(None, description="Alt text of the child post")
    likesCount: Optional[int] = Field(None, description="Number of likes on the child post")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the child post")
    childPosts: List[Dict[str, Any]] = Field(default_factory=list, description="Child posts of the child post")
    taggedUsers: List[TaggedUserSchema] = Field(default_factory=list, description="Users tagged in the child post")

class InstagramPostSchema(BaseModel):
    # Core identifiers (exact field names from raw data)
    id: str = Field(..., description="Unique post identifier")
    type: PostType = Field(..., description="Type of post")
    shortCode: str = Field(..., description="Short code for the post")
    
    # URLs and media (exact field names)
    url: str = Field(..., description="URL to the post")
    displayUrl: str = Field(..., description="Display URL of the media")
    images: List[str] = Field(default_factory=list, description="Images in the post")
    
    # Content and metadata (exact field names)
    caption: Optional[str] = Field(None, description="Post caption/description")
    alt: Optional[str] = Field(None, description="Alt text of the image")
    
    # Engagement metrics (exact field names)
    likesCount: int = Field(0, description="Number of likes")
    commentsCount: int = Field(0, description="Number of comments")
    
    # Comments data (exact field names)
    firstComment: Optional[str] = Field(None, description="First comment on the post")
    latestComments: List[CommentSchema] = Field(default_factory=list, description="Latest comments on the post")
    
    # Temporal information (exact field names)
    timestamp: datetime = Field(..., description="When the post was created")
    
    # Dimensions (exact field names)
    dimensionsHeight: int = Field(0, description="Height of the image")
    dimensionsWidth: int = Field(0, description="Width of the image")
    
    # Owner information (exact field names)
    ownerFullName: str = Field(..., description="Full name of the post owner")
    ownerUsername: str = Field(..., description="Username of the post owner")
    ownerId: str = Field(..., description="ID of the post owner")
    
    # Tags and mentions (exact field names)
    hashtags: List[str] = Field(default_factory=list, description="Hashtags in the post")
    mentions: List[str] = Field(default_factory=list, description="Mentions in the post")
    taggedUsers: List[TaggedUserSchema] = Field(default_factory=list, description="Users tagged in the post")
    
    # Additional metadata (exact field names)
    isCommentsDisabled: bool = Field(False, description="Whether comments are disabled")
    inputUrl: str = Field(..., description="Input URL used for scraping")
    isSponsored: bool = Field(False, description="Whether the post is sponsored")
    
    # Carousel specific (exact field names)
    childPosts: List[ChildPostSchema] = Field(default_factory=list, description="Child posts for carousel")
    
    # For search and categorization (our custom fields)
    extracted_keywords: List[str] = Field(default_factory=list, description="Keywords extracted from caption")
    detected_objects: List[str] = Field(default_factory=list, description="Objects detected in image")
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant colors in image")
    style_attributes: List[str] = Field(default_factory=list, description="Style attributes (casual, formal, etc.)")
    product_type: Optional[str] = Field(None, description="Type of product (e.g., shoes, hoodie, etc.)")
    brand_name: Optional[str] = Field(None, description="Brand name extracted from content")
    category: Optional[str] = Field(None, description="Category of the content")
    
    # Metadata for our system
    scraped_date: datetime = Field(default_factory=datetime.now, description="When the post was scraped")
    primary_image_url: str = Field(..., description="Primary image URL (first image from images array)")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "3745039532065324748",
                "type": "Sidecar",
                "shortCode": "DP5DZzCjH7M",
                "url": "https://www.instagram.com/p/DP5DZzCjH7M/",
                "displayUrl": "https://scontent-atl3-3.cdninstagram.com/v/t51.2885-15/566479434_18531430417029827_8766797240640779948_n.jpg",
                "images": [
                    "https://scontent-atl3-3.cdninstagram.com/v/t51.2885-15/566479434_18531430417029827_8766797240640779948_n.jpg",
                    "https://scontent-atl3-3.cdninstagram.com/v/t51.2885-15/565179770_18531430411029827_7123646841153881080_n.jpg"
                ],
                "caption": "Today's top #outfitgrid is by @tmarkgotkickss. \n▫️ Fear of God x Barriers Worldwide Tee \n▫️ John Elliott Leather Shorts \n▫️ Infinite Archives x Jordan XVII",
                "alt": "Photo shared by OUTFITGRID™ on October 16, 2025 tagging @easyotabor, @jumpman23, @johnelliott_, @fearofgod, @tmarkgotkickss, @outfitgrid, @barriersworldwide, and @infinitearchives.",
                "likesCount": 1,
                "commentsCount": 10,
                "firstComment": "🔥🔥",
                "latestComments": [
                    {
                        "id": "18309965617247026",
                        "text": "🔥🔥",
                        "ownerUsername": "funerald_",
                        "timestamp": "2025-10-17T17:00:53.000Z",
                        "repliesCount": 0,
                        "replies": [],
                        "likesCount": 0,
                        "owner": {
                            "id": "23012605",
                            "is_verified": False,
                            "profile_pic_url": "https://scontent-atl3-3.cdninstagram.com/v/t51.2885-19/496780842_18500546188036606_7825485760592065559_n.jpg",
                            "username": "funerald_"
                        }
                    }
                ],
                "timestamp": "2025-10-17T01:12:03.000Z",
                "dimensionsHeight": 1080,
                "dimensionsWidth": 1080,
                "ownerFullName": "OUTFITGRID™",
                "ownerUsername": "outfitgrid",
                "ownerId": "251677826",
                "hashtags": ["outfitgrid"],
                "mentions": ["tmarkgotkickss."],
                "taggedUsers": [
                    {
                        "full_name": "Jordan",
                        "id": "5332352",
                        "is_verified": True,
                        "profile_pic_url": "https://scontent-atl3-2.cdninstagram.com/v/t51.2885-19/549283129_18536848891020353_515842368480894497_n.jpg",
                        "username": "jumpman23"
                    }
                ],
                "isCommentsDisabled": False,
                "inputUrl": "https://www.instagram.com/outfitgrid/",
                "isSponsored": False,
                "childPosts": [],
                "primary_image_url": "https://scontent-atl3-3.cdninstagram.com/v/t51.2885-15/566479434_18531430417029827_8766797240640779948_n.jpg",
                "scraped_date": "2025-01-20T10:30:00.000Z",
                "extracted_keywords": ["outfit", "streetwear", "fashion"],
                "dominant_colors": ["#000000", "#FFFFFF"],
                "style_attributes": ["casual", "streetwear"],
                "product_type": "outfit",
                "category": "Fashion"
            }
        }
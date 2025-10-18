from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, asc
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone
import json

from src.models.instagram_post_model import InstagramPost
from src.schemas.instagram_transformed_schema import InstagramPostSchema

class InstagramPostRepository:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    def _serialize_to_json(self, obj: Any) -> Any:
        """Convert Pydantic models to JSON-serializable format"""
        if hasattr(obj, 'dict'):
            # For Pydantic models, use exclude_none=True and handle datetime serialization
            return json.loads(json.dumps(obj.dict(), default=str))
        elif isinstance(obj, list):
            return [self._serialize_to_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._serialize_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Convert datetime to naive UTC datetime for PostgreSQL storage"""
        if dt is None:
            return None
        
        if dt.tzinfo is not None:
            # Convert timezone-aware datetime to UTC and make it naive
            utc_dt = dt.astimezone(timezone.utc)
            return utc_dt.replace(tzinfo=None)
        else:
            # Already naive, return as-is
            return dt
        
    async def create_post(self, post_data: InstagramPostSchema) -> InstagramPost:
        """Create a new Instagram post in the database"""
        try:
            print(f"[Instagram Repository] Creating post with ID: {post_data.id}")
            
            # Properly serialize nested objects
            latest_comments_serialized = self._serialize_to_json(post_data.latestComments)
            tagged_users_serialized = self._serialize_to_json(post_data.taggedUsers)
            child_posts_serialized = self._serialize_to_json(post_data.childPosts)
            
            # Normalize datetime objects to naive UTC
            normalized_timestamp = self._normalize_datetime(post_data.timestamp)
            normalized_scraped_date = self._normalize_datetime(post_data.scraped_date)
            current_time = datetime.utcnow()  # Always naive UTC
            
            print(f"[Instagram Repository] Original timestamp: {post_data.timestamp}")
            print(f"[Instagram Repository] Normalized timestamp: {normalized_timestamp}")
            print(f"[Instagram Repository] Scraped date: {normalized_scraped_date}")
            
            # Convert Pydantic model to SQLAlchemy model
            db_post = InstagramPost(
                id=post_data.id,
                type=post_data.type.value,
                short_code=post_data.shortCode,
                url=post_data.url,
                display_url=post_data.displayUrl,
                images=post_data.images,
                caption=post_data.caption,
                alt=post_data.alt,
                likes_count=post_data.likesCount,
                comments_count=post_data.commentsCount,
                first_comment=post_data.firstComment,
                latest_comments=latest_comments_serialized,
                timestamp=normalized_timestamp,  # ✅ Normalized datetime
                dimensions_height=post_data.dimensionsHeight,
                dimensions_width=post_data.dimensionsWidth,
                owner_full_name=post_data.ownerFullName,
                owner_username=post_data.ownerUsername,
                owner_id=post_data.ownerId,
                hashtags=post_data.hashtags,
                mentions=post_data.mentions,
                tagged_users=tagged_users_serialized,
                is_comments_disabled=post_data.isCommentsDisabled,
                input_url=post_data.inputUrl,
                is_sponsored=post_data.isSponsored,
                child_posts=child_posts_serialized,
                extracted_keywords=post_data.extracted_keywords,
                detected_objects=post_data.detected_objects,
                dominant_colors=post_data.dominant_colors,
                style_attributes=post_data.style_attributes,
                product_type=post_data.product_type,
                brand_name=post_data.brand_name,
                category=post_data.category,
                scraped_date=normalized_scraped_date,  # ✅ Normalized datetime
                primary_image_url=post_data.primary_image_url,
                created_at=current_time,  # ✅ Naive UTC datetime
                updated_at=current_time   # ✅ Naive UTC datetime
            )
            
            self.db_session.add(db_post)
            await self.db_session.commit()
            await self.db_session.refresh(db_post)
            
            print(f"[Instagram Repository] Successfully created post: {db_post.id}")
            return db_post
            
        except IntegrityError as e:
            await self.db_session.rollback()
            print(f"[Instagram Repository] Post already exists: {post_data.id}")
            # If post already exists, update it instead
            return await self.update_post_by_id(post_data.id, post_data)
        except Exception as e:
            await self.db_session.rollback()
            print(f"[Instagram Repository] Error creating post: {e}")
            raise e
    
    async def update_post_by_id(self, post_id: str, post_data: InstagramPostSchema) -> Optional[InstagramPost]:
        """Update an existing Instagram post"""
        try:
            print(f"[Instagram Repository] Updating post with ID: {post_id}")
            
            # Get existing post
            stmt = select(InstagramPost).where(InstagramPost.id == post_id)
            result = await self.db_session.execute(stmt)
            db_post = result.scalar_one_or_none()
            
            if not db_post:
                print(f"[Instagram Repository] Post not found for update: {post_id}")
                return None
            
            # Properly serialize nested objects
            latest_comments_serialized = self._serialize_to_json(post_data.latestComments)
            tagged_users_serialized = self._serialize_to_json(post_data.taggedUsers)
            child_posts_serialized = self._serialize_to_json(post_data.childPosts)
            
            # Normalize datetime objects
            normalized_timestamp = self._normalize_datetime(post_data.timestamp)
            normalized_scraped_date = self._normalize_datetime(post_data.scraped_date)
            current_time = datetime.utcnow()  # Always naive UTC
            
            # Update fields
            db_post.type = post_data.type.value
            db_post.short_code = post_data.shortCode
            db_post.url = post_data.url
            db_post.display_url = post_data.displayUrl
            db_post.images = post_data.images
            db_post.caption = post_data.caption
            db_post.alt = post_data.alt
            db_post.likes_count = post_data.likesCount
            db_post.comments_count = post_data.commentsCount
            db_post.first_comment = post_data.firstComment
            db_post.latest_comments = latest_comments_serialized
            db_post.timestamp = normalized_timestamp  # ✅ Normalized datetime
            db_post.dimensions_height = post_data.dimensionsHeight
            db_post.dimensions_width = post_data.dimensionsWidth
            db_post.owner_full_name = post_data.ownerFullName
            db_post.owner_username = post_data.ownerUsername
            db_post.owner_id = post_data.ownerId
            db_post.hashtags = post_data.hashtags
            db_post.mentions = post_data.mentions
            db_post.tagged_users = tagged_users_serialized
            db_post.is_comments_disabled = post_data.isCommentsDisabled
            db_post.input_url = post_data.inputUrl
            db_post.is_sponsored = post_data.isSponsored
            db_post.child_posts = child_posts_serialized
            db_post.extracted_keywords = post_data.extracted_keywords
            db_post.detected_objects = post_data.detected_objects
            db_post.dominant_colors = post_data.dominant_colors
            db_post.style_attributes = post_data.style_attributes
            db_post.product_type = post_data.product_type
            db_post.brand_name = post_data.brand_name
            db_post.category = post_data.category
            db_post.scraped_date = normalized_scraped_date  # ✅ Normalized datetime
            db_post.primary_image_url = post_data.primary_image_url
            db_post.updated_at = current_time  # ✅ Naive UTC datetime
            
            await self.db_session.commit()
            await self.db_session.refresh(db_post)
            
            print(f"[Instagram Repository] Successfully updated post: {db_post.id}")
            return db_post
            
        except Exception as e:
            await self.db_session.rollback()
            print(f"[Instagram Repository] Error updating post: {e}")
            raise e
    
    async def get_post_by_id(self, post_id: str) -> Optional[InstagramPost]:
        """Get Instagram post by ID"""
        try:
            stmt = select(InstagramPost).where(InstagramPost.id == post_id)
            result = await self.db_session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            print(f"[Instagram Repository] Error getting post by ID: {e}")
            raise e
    
    async def get_posts_by_owner_username(self, owner_username: str, limit: int = 50) -> List[InstagramPost]:
        """Get Instagram posts by owner username"""
        try:
            stmt = (
                select(InstagramPost)
                .where(InstagramPost.owner_username == owner_username)
                .order_by(desc(InstagramPost.timestamp))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            print(f"[Instagram Repository] Error getting posts by owner: {e}")
            raise e
    
    async def search_posts_by_caption(self, search_term: str, limit: int = 50) -> List[InstagramPost]:
        """Search Instagram posts by caption content"""
        try:
            stmt = (
                select(InstagramPost)
                .where(InstagramPost.caption.ilike(f"%{search_term}%"))
                .order_by(desc(InstagramPost.timestamp))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            print(f"[Instagram Repository] Error searching posts by caption: {e}")
            raise e
    
    async def get_recent_posts(self, limit: int = 50) -> List[InstagramPost]:
        """Get most recent Instagram posts"""
        try:
            stmt = (
                select(InstagramPost)
                .order_by(desc(InstagramPost.scraped_date))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            print(f"[Instagram Repository] Error getting recent posts: {e}")
            raise e
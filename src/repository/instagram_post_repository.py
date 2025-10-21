"""
Instagram Post repository for database operations.
Handles all database interactions for Instagram posts.
"""

from typing import List, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, asc, func, or_
from sqlalchemy.exc import IntegrityError
from datetime import datetime, date, time, timezone
import traceback
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

    async def get_post_by_id(self, post_id: str) -> Optional[InstagramPost]:
        """Get an Instagram post by ID"""
        try:
            print(f"[Instagram Post Repository] Fetching post with ID: {post_id}")
            
            stmt = select(InstagramPost).where(InstagramPost.id == post_id)
            result = await self.db_session.execute(stmt)
            post = result.scalar_one_or_none()
            
            if post:
                print(f"[Instagram Post Repository] Post found: {post_id}")
            else:
                print(f"[Instagram Post Repository] Post not found: {post_id}")
            
            return post

        except Exception as e:
            print(f"[Instagram Post Repository] Error fetching post: {str(e)}")
            raise e

    async def get_posts_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        owner_username: Optional[str] = None,
        sort_by: str = "scraped_date",
        sort_order: str = "desc"
    ) -> Tuple[List[InstagramPost], int]:
        """
        Get paginated Instagram posts with optional filters.
        Returns tuple of (posts, total_count)
        """
        try:
            print(f"[Instagram Post Repository] Getting paginated posts - Page: {page}, Size: {page_size}")
            
            # Base query
            query = select(InstagramPost)
            count_query = select(func.count(InstagramPost.id))
            
            # Apply filters
            if owner_username:
                query = query.where(InstagramPost.owner_username == owner_username)
                count_query = count_query.where(InstagramPost.owner_username == owner_username)
            
            # Get total count
            total_count_result = await self.db_session.execute(count_query)
            total_count = total_count_result.scalar()
            
            # Apply sorting
            sort_column = getattr(InstagramPost, sort_by, InstagramPost.scraped_date)
            if sort_order.lower() == "asc":
                query = query.order_by(asc(sort_column))
            else:
                query = query.order_by(desc(sort_column))
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = await self.db_session.execute(query)
            posts = result.scalars().all()
            
            print(f"[Instagram Post Repository] Found {len(posts)} posts (total: {total_count})")
            return list(posts), total_count

        except Exception as e:
            print(f"[Instagram Post Repository] Error getting paginated posts: {str(e)}")
            raise e

    async def get_posts_by_owner_username(self, owner_username: str, limit: int = 50) -> List[InstagramPost]:
        """Get posts by owner username"""
        try:
            print(f"[Instagram Post Repository] Fetching posts for owner: {owner_username}")
            
            stmt = (
                select(InstagramPost)
                .where(InstagramPost.owner_username == owner_username)
                .order_by(desc(InstagramPost.scraped_date))
                .limit(limit)
            )
            
            result = await self.db_session.execute(stmt)
            posts = result.scalars().all()
            
            print(f"[Instagram Post Repository] Found {len(posts)} posts for owner: {owner_username}")
            return list(posts)

        except Exception as e:
            print(f"[Instagram Post Repository] Error fetching posts by owner: {str(e)}")
            raise e

    async def search_posts_by_caption(self, search_term: str, limit: int = 50) -> List[InstagramPost]:
        """Search posts by caption"""
        try:
            print(f"[Instagram Post Repository] Searching posts with term: {search_term}")
            
            stmt = (
                select(InstagramPost)
                .where(InstagramPost.caption.ilike(f"%{search_term}%"))
                .order_by(desc(InstagramPost.scraped_date))
                .limit(limit)
            )
            
            result = await self.db_session.execute(stmt)
            posts = result.scalars().all()
            
            print(f"[Instagram Post Repository] Found {len(posts)} posts matching: {search_term}")
            return list(posts)

        except Exception as e:
            print(f"[Instagram Post Repository] Error searching posts: {str(e)}")
            raise e

    async def get_recent_posts(self, limit: int = 50) -> List[InstagramPost]:
        """Get recent Instagram posts"""
        try:
            print(f"[Instagram Post Repository] Fetching {limit} recent posts")
            
            stmt = (
                select(InstagramPost)
                .order_by(desc(InstagramPost.scraped_date))
                .limit(limit)
            )
            
            result = await self.db_session.execute(stmt)
            posts = result.scalars().all()
            
            print(f"[Instagram Post Repository] Found {len(posts)} recent posts")
            return list(posts)

        except Exception as e:
            print(f"[Instagram Post Repository] Error fetching recent posts: {str(e)}")
            raise e
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, asc
from sqlalchemy.exc import IntegrityError
from datetime import datetime
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
        
    async def create_post(self, post_data: InstagramPostSchema) -> InstagramPost:
        """Create a new Instagram post in the database"""
        try:
            print(f"[Instagram Repository] Creating post with ID: {post_data.id}")
            
            # Properly serialize nested objects
            latest_comments_serialized = self._serialize_to_json(post_data.latestComments)
            tagged_users_serialized = self._serialize_to_json(post_data.taggedUsers)
            child_posts_serialized = self._serialize_to_json(post_data.childPosts)
            
            print(f"[Instagram Repository] Serialized {len(latest_comments_serialized)} comments")
            print(f"[Instagram Repository] Serialized {len(tagged_users_serialized)} tagged users")
            print(f"[Instagram Repository] Serialized {len(child_posts_serialized)} child posts")
            
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
                latest_comments=latest_comments_serialized,  # ✅ Properly serialized
                timestamp=post_data.timestamp,
                dimensions_height=post_data.dimensionsHeight,
                dimensions_width=post_data.dimensionsWidth,
                owner_full_name=post_data.ownerFullName,
                owner_username=post_data.ownerUsername,
                owner_id=post_data.ownerId,
                hashtags=post_data.hashtags,
                mentions=post_data.mentions,
                tagged_users=tagged_users_serialized,  # ✅ Properly serialized
                is_comments_disabled=post_data.isCommentsDisabled,
                input_url=post_data.inputUrl,
                is_sponsored=post_data.isSponsored,
                child_posts=child_posts_serialized,  # ✅ Properly serialized
                extracted_keywords=post_data.extracted_keywords,
                detected_objects=post_data.detected_objects,
                dominant_colors=post_data.dominant_colors,
                style_attributes=post_data.style_attributes,
                product_type=post_data.product_type,
                brand_name=post_data.brand_name,
                category=post_data.category,
                scraped_date=post_data.scraped_date,
                primary_image_url=post_data.primary_image_url
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
            db_post.latest_comments = latest_comments_serialized  # ✅ Properly serialized
            db_post.timestamp = post_data.timestamp
            db_post.dimensions_height = post_data.dimensionsHeight
            db_post.dimensions_width = post_data.dimensionsWidth
            db_post.owner_full_name = post_data.ownerFullName
            db_post.owner_username = post_data.ownerUsername
            db_post.owner_id = post_data.ownerId
            db_post.hashtags = post_data.hashtags
            db_post.mentions = post_data.mentions
            db_post.tagged_users = tagged_users_serialized  # ✅ Properly serialized
            db_post.is_comments_disabled = post_data.isCommentsDisabled
            db_post.input_url = post_data.inputUrl
            db_post.is_sponsored = post_data.isSponsored
            db_post.child_posts = child_posts_serialized  # ✅ Properly serialized
            db_post.extracted_keywords = post_data.extracted_keywords
            db_post.detected_objects = post_data.detected_objects
            db_post.dominant_colors = post_data.dominant_colors
            db_post.style_attributes = post_data.style_attributes
            db_post.product_type = post_data.product_type
            db_post.brand_name = post_data.brand_name
            db_post.category = post_data.category
            db_post.primary_image_url = post_data.primary_image_url
            db_post.updated_at = datetime.utcnow()
            
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
            print(f"[Instagram Repository] Getting post by ID: {post_id}")
            
            stmt = select(InstagramPost).where(InstagramPost.id == post_id)
            result = await self.db_session.execute(stmt)
            db_post = result.scalar_one_or_none()
            
            if db_post:
                print(f"[Instagram Repository] Found post: {post_id}")
            else:
                print(f"[Instagram Repository] Post not found: {post_id}")
                
            return db_post
            
        except Exception as e:
            print(f"[Instagram Repository] Error getting post by ID: {e}")
            raise e
    
    async def get_posts_by_owner_username(self, owner_username: str, limit: int = 50) -> List[InstagramPost]:
        """Get Instagram posts by owner username"""
        try:
            print(f"[Instagram Repository] Getting posts by owner: {owner_username}, limit: {limit}")
            
            stmt = (
                select(InstagramPost)
                .where(InstagramPost.owner_username == owner_username)
                .order_by(desc(InstagramPost.timestamp))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            posts = result.scalars().all()
            
            print(f"[Instagram Repository] Found {len(posts)} posts for owner: {owner_username}")
            return list(posts)
            
        except Exception as e:
            print(f"[Instagram Repository] Error getting posts by owner: {e}")
            raise e
    
    async def get_posts_by_hashtags(self, hashtags: List[str], limit: int = 50) -> List[InstagramPost]:
        """Get Instagram posts by hashtags"""
        try:
            print(f"[Instagram Repository] Getting posts by hashtags: {hashtags}, limit: {limit}")
            
            # Create conditions for each hashtag
            hashtag_conditions = []
            for hashtag in hashtags:
                hashtag_conditions.append(InstagramPost.hashtags.contains([hashtag]))
            
            stmt = (
                select(InstagramPost)
                .where(or_(*hashtag_conditions))
                .order_by(desc(InstagramPost.timestamp))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            posts = result.scalars().all()
            
            print(f"[Instagram Repository] Found {len(posts)} posts for hashtags: {hashtags}")
            return list(posts)
            
        except Exception as e:
            print(f"[Instagram Repository] Error getting posts by hashtags: {e}")
            raise e
    
    async def search_posts_by_caption(self, search_term: str, limit: int = 50) -> List[InstagramPost]:
        """Search Instagram posts by caption content"""
        try:
            print(f"[Instagram Repository] Searching posts by caption: {search_term}, limit: {limit}")
            
            stmt = (
                select(InstagramPost)
                .where(InstagramPost.caption.ilike(f"%{search_term}%"))
                .order_by(desc(InstagramPost.timestamp))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            posts = result.scalars().all()
            
            print(f"[Instagram Repository] Found {len(posts)} posts for search term: {search_term}")
            return list(posts)
            
        except Exception as e:
            print(f"[Instagram Repository] Error searching posts by caption: {e}")
            raise e
    
    async def get_recent_posts(self, limit: int = 50) -> List[InstagramPost]:
        """Get most recent Instagram posts"""
        try:
            print(f"[Instagram Repository] Getting recent posts, limit: {limit}")
            
            stmt = (
                select(InstagramPost)
                .order_by(desc(InstagramPost.scraped_date))
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            posts = result.scalars().all()
            
            print(f"[Instagram Repository] Found {len(posts)} recent posts")
            return list(posts)
            
        except Exception as e:
            print(f"[Instagram Repository] Error getting recent posts: {e}")
            raise e
    
    async def delete_post_by_id(self, post_id: str) -> bool:
        """Delete Instagram post by ID"""
        try:
            print(f"[Instagram Repository] Deleting post by ID: {post_id}")
            
            stmt = select(InstagramPost).where(InstagramPost.id == post_id)
            result = await self.db_session.execute(stmt)
            db_post = result.scalar_one_or_none()
            
            if not db_post:
                print(f"[Instagram Repository] Post not found for deletion: {post_id}")
                return False
            
            await self.db_session.delete(db_post)
            await self.db_session.commit()
            
            print(f"[Instagram Repository] Successfully deleted post: {post_id}")
            return True
            
        except Exception as e:
            await self.db_session.rollback()
            print(f"[Instagram Repository] Error deleting post: {e}")
            raise e
    
    async def bulk_create_posts(self, posts_data: List[InstagramPostSchema]) -> List[InstagramPost]:
        """Create multiple Instagram posts in bulk"""
        try:
            print(f"[Instagram Repository] Bulk creating {len(posts_data)} posts")
            
            db_posts = []
            for post_data in posts_data:
                try:
                    # Check if post already exists
                    existing_post = await self.get_post_by_id(post_data.id)
                    if existing_post:
                        print(f"[Instagram Repository] Post already exists, skipping: {post_data.id}")
                        continue
                    
                    # Properly serialize nested objects
                    latest_comments_serialized = self._serialize_to_json(post_data.latestComments)
                    tagged_users_serialized = self._serialize_to_json(post_data.taggedUsers)
                    child_posts_serialized = self._serialize_to_json(post_data.childPosts)
                    
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
                        latest_comments=latest_comments_serialized,  # ✅ Properly serialized
                        timestamp=post_data.timestamp,
                        dimensions_height=post_data.dimensionsHeight,
                        dimensions_width=post_data.dimensionsWidth,
                        owner_full_name=post_data.ownerFullName,
                        owner_username=post_data.ownerUsername,
                        owner_id=post_data.ownerId,
                        hashtags=post_data.hashtags,
                        mentions=post_data.mentions,
                        tagged_users=tagged_users_serialized,  # ✅ Properly serialized
                        is_comments_disabled=post_data.isCommentsDisabled,
                        input_url=post_data.inputUrl,
                        is_sponsored=post_data.isSponsored,
                        child_posts=child_posts_serialized,  # ✅ Properly serialized
                        extracted_keywords=post_data.extracted_keywords,
                        detected_objects=post_data.detected_objects,
                        dominant_colors=post_data.dominant_colors,
                        style_attributes=post_data.style_attributes,
                        product_type=post_data.product_type,
                        brand_name=post_data.brand_name,
                        category=post_data.category,
                        scraped_date=post_data.scraped_date,
                        primary_image_url=post_data.primary_image_url
                    )
                    
                    self.db_session.add(db_post)
                    db_posts.append(db_post)
                    
                except Exception as e:
                    print(f"[Instagram Repository] Error preparing post {post_data.id}: {e}")
                    continue
            
            if db_posts:
                await self.db_session.commit()
                print(f"[Instagram Repository] Successfully created {len(db_posts)} posts")
            
            return db_posts
            
        except Exception as e:
            await self.db_session.rollback()
            print(f"[Instagram Repository] Error in bulk create: {e}")
            raise e
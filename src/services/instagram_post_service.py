from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from src.repository.instagram_post_repository import InstagramPostRepository  # Fixed path
from src.schemas.instagram_transformed_schema import InstagramPostSchema
from src.models.instagram_post_model import InstagramPost

class InstagramPostService:
    def __init__(self, db_session: AsyncSession):
        self.repository = InstagramPostRepository(db_session)
    
    async def save_scraped_posts(self, posts_data: List[InstagramPostSchema]) -> List[InstagramPost]:
        """Save scraped Instagram posts to database"""
        try:
            print(f"[Instagram Post Service] Saving {len(posts_data)} posts to database")
            
            saved_posts = []
            for post_data in posts_data:
                try:
                    # Try to create new post
                    saved_post = await self.repository.create_post(post_data)
                    saved_posts.append(saved_post)
                    print(f"[Instagram Post Service] Saved post: {post_data.id}")
                except Exception as e:
                    print(f"[Instagram Post Service] Error saving post {post_data.id}: {e}")
                    continue
            
            print(f"[Instagram Post Service] Successfully saved {len(saved_posts)} posts")
            return saved_posts
            
        except Exception as e:
            print(f"[Instagram Post Service] Error in save_scraped_posts: {e}")
            raise e
    
    async def get_post_by_id(self, post_id: str) -> Optional[InstagramPost]:
        """Get Instagram post by ID"""
        return await self.repository.get_post_by_id(post_id)
    
    async def get_posts_by_owner(self, owner_username: str, limit: int = 50) -> List[InstagramPost]:
        """Get posts by owner username"""
        return await self.repository.get_posts_by_owner_username(owner_username, limit)
    
    async def search_posts(self, search_term: str, limit: int = 50) -> List[InstagramPost]:
        """Search posts by caption"""
        return await self.repository.search_posts_by_caption(search_term, limit)
    
    async def get_recent_posts(self, limit: int = 50) -> List[InstagramPost]:
        """Get recent posts"""
        return await self.repository.get_recent_posts(limit)
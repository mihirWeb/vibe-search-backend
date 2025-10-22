"""
Instagram Post controller that handles business logic for Instagram posts.
Orchestrates between services and repositories.
"""

from typing import List, Optional, Dict
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.repository.product_repository import ProductRepository

import math

from src.repository.instagram_post_repository import InstagramPostRepository
from src.schemas.instagram_post_response_schema import (
    InstagramPostPaginationRequest,
    InstagramPostPaginatedResponse,
    InstagramPostDetailResponse,
    InstagramPostMinimalSchema,
    PaginationMeta
)


class InstagramPostController:
    """Controller for Instagram post operations"""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.instagram_repository = InstagramPostRepository(db_session)
        self.product_repository = ProductRepository(db_session)
        print("[Instagram Post Controller] Initialized")
    
    async def get_paginated_posts(
        self,
        request: InstagramPostPaginationRequest
    ) -> InstagramPostPaginatedResponse:
        """
        Get paginated list of Instagram posts.
        Supports filtering by owner username and sorting.
        """
        try:
            print(f"[Instagram Post Controller] Fetching paginated posts - Page: {request.page}, Size: {request.page_size}")
            
            # Fetch posts and total count from repository
            posts, total_count = await self.instagram_repository.get_posts_paginated(
                page=request.page,
                page_size=request.page_size,
                owner_username=request.owner_username,
                sort_by=request.sort_by,
                sort_order=request.sort_order
            )
            
            # Convert to minimal schemas
            post_schemas = [self._post_model_to_minimal_schema(p) for p in posts]
            
            # Calculate pagination metadata
            total_pages = math.ceil(total_count / request.page_size) if total_count > 0 else 0
            
            pagination_meta = PaginationMeta(
                current_page=request.page,
                page_size=request.page_size,
                total_items=total_count,
                total_pages=total_pages,
                has_next=request.page < total_pages,
                has_previous=request.page > 1
            )
            
            # Build response message
            filter_msg = f" for @{request.owner_username}" if request.owner_username else ""
            message = f"Retrieved {len(posts)} posts (page {request.page} of {total_pages}){filter_msg}"
            
            response = InstagramPostPaginatedResponse(
                success=True,
                message=message,
                posts=post_schemas,
                pagination=pagination_meta
            )
            
            print(f"[Instagram Post Controller] Successfully fetched {len(posts)} posts")
            return response
            
        except Exception as e:
            print(f"[Instagram Post Controller] Error fetching paginated posts: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching paginated posts: {str(e)}"
            )
    
    
    async def get_post_by_id(self, post_id: str) -> InstagramPostDetailResponse:
        """Get an Instagram post by ID with full details"""
        try:
            print(f"[Instagram Post Controller] Fetching post with ID: {post_id}")
            
            post = await self.instagram_repository.get_post_by_id(post_id)
            
            if not post:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Instagram post not found with ID: {post_id}"
                )
            
            # Check if post has been extracted
            is_extracted = await self.product_repository.check_post_extracted(post_id)
            
            # Convert to detailed schema
            post_detail = InstagramPostDetailResponse(
                success=True,
                message="Post retrieved successfully",
                post={
                    "id": post.id,
                    "type": post.type,
                    "short_code": post.short_code,
                    "url": post.url,
                    "display_url": post.display_url,
                    "images": post.images,
                    "caption": post.caption,
                    "alt": post.alt,
                    "likes_count": post.likes_count,
                    "comments_count": post.comments_count,
                    "first_comment": post.first_comment,
                    "latest_comments": post.latest_comments,
                    "timestamp": post.timestamp,
                    "dimensions_height": post.dimensions_height,
                    "dimensions_width": post.dimensions_width,
                    "owner_full_name": post.owner_full_name,
                    "owner_username": post.owner_username,
                    "owner_id": post.owner_id,
                    "hashtags": post.hashtags,
                    "mentions": post.mentions,
                    "tagged_users": post.tagged_users,
                    "is_comments_disabled": post.is_comments_disabled,
                    "is_sponsored": post.is_sponsored,
                    "child_posts": post.child_posts,
                    "scraped_date": post.scraped_date,
                    "created_at": post.created_at,
                    "updated_at": post.updated_at,
                    "is_extracted": is_extracted  # Add this flag
                }
            )
            
            return post_detail
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[Instagram Post Controller] Error fetching post: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching post: {str(e)}"
            )

    
    def _post_model_to_minimal_schema(self, post) -> InstagramPostMinimalSchema:
        """Convert Instagram post model to minimal schema"""
        return InstagramPostMinimalSchema(
            id=post.id,
            type=post.type,
            url=post.url,
            display_url=post.display_url,
            caption=post.caption[:200] if post.caption else None,  # Truncate caption
            likes_count=post.likes_count,
            comments_count=post.comments_count,
            timestamp=post.timestamp,
            owner_username=post.owner_username,
            owner_full_name=post.owner_full_name,
            scraped_date=post.scraped_date
        )
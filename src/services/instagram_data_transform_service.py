from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from src.schemas.instagram_transformed_schema import (
    InstagramPostSchema, 
    CommentSchema, 
    CommentOwnerSchema, 
    TaggedUserSchema, 
    ChildPostSchema,
    PostType
)

class DataTransformerService:
    """Service to transform raw scraped data to structured schemas"""
    
    def transform_instagram_post(self, raw_data: Dict[str, Any]) -> InstagramPostSchema:
        """Transform raw Instagram data to InstagramPostSchema - only map existing fields"""
        try:
            print(f"[Data Transformer] Transforming Instagram post: {raw_data.get('id', 'unknown')}")
            
            # Transform comments - only if they exist
            latest_comments = []
            if raw_data.get("latestComments"):
                for comment_data in raw_data.get("latestComments", []):
                    try:
                        comment = self._transform_comment(comment_data)
                        latest_comments.append(comment)
                    except Exception as e:
                        print(f"[Data Transformer] Error transforming comment: {e}")
            
            # Transform tagged users - only if they exist
            tagged_users = []
            if raw_data.get("taggedUsers"):
                for user_data in raw_data.get("taggedUsers", []):
                    try:
                        tagged_user = self._transform_tagged_user(user_data)
                        tagged_users.append(tagged_user)
                    except Exception as e:
                        print(f"[Data Transformer] Error transforming tagged user: {e}")
            
            # Transform child posts - only if they exist
            child_posts = []
            if raw_data.get("childPosts"):
                for child_data in raw_data.get("childPosts", []):
                    try:
                        child_post = self._transform_child_post(child_data)
                        child_posts.append(child_post)
                    except Exception as e:
                        print(f"[Data Transformer] Error transforming child post: {e}")
            
            # Map type to enum - only valid types
            post_type_str = raw_data.get("type", "Image")
            try:
                post_type = PostType(post_type_str)
            except ValueError:
                post_type = PostType.IMAGE
            
            # Parse timestamp - only if it exists
            timestamp = None
            timestamp_str = raw_data.get("timestamp")
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timestamp_str
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Get primary image URL (first image from images array or displayUrl)
            images = raw_data.get("images", [])
            primary_image_url = ""
            if images:
                primary_image_url = images[0]
            elif raw_data.get("displayUrl"):
                primary_image_url = raw_data.get("displayUrl")
            
            # Create the transformed post with ONLY existing data - no custom fields
            transformed_post = InstagramPostSchema(
                # Core identifiers - exact mapping
                id=raw_data.get("id", ""),
                type=post_type,
                shortCode=raw_data.get("shortCode", ""),
                
                # URLs and media - exact mapping
                url=raw_data.get("url", ""),
                displayUrl=raw_data.get("displayUrl", ""),
                images=images,
                
                # Content and metadata - exact mapping
                caption=raw_data.get("caption"),
                alt=raw_data.get("alt"),
                
                # Engagement metrics - exact mapping, handle negative values
                likesCount=max(0, raw_data.get("likesCount", 0)) if raw_data.get("likesCount", 0) >= 0 else 0,
                commentsCount=raw_data.get("commentsCount", 0),
                
                # Comments data - exact mapping
                firstComment=raw_data.get("firstComment"),
                latestComments=latest_comments,
                
                # Temporal information - exact mapping
                timestamp=timestamp,
                
                # Dimensions - exact mapping
                dimensionsHeight=raw_data.get("dimensionsHeight", 0),
                dimensionsWidth=raw_data.get("dimensionsWidth", 0),
                
                # Owner information - exact mapping
                ownerFullName=raw_data.get("ownerFullName", ""),
                ownerUsername=raw_data.get("ownerUsername", ""),
                ownerId=raw_data.get("ownerId", ""),
                
                # Tags and mentions - exact mapping
                hashtags=raw_data.get("hashtags", []),
                mentions=raw_data.get("mentions", []),
                taggedUsers=tagged_users,
                
                # Additional metadata - exact mapping
                isCommentsDisabled=raw_data.get("isCommentsDisabled", False),
                inputUrl=raw_data.get("inputUrl", ""),
                isSponsored=raw_data.get("isSponsored", False),
                
                # Carousel specific - exact mapping
                childPosts=child_posts,
                
                # Custom fields - leave empty as requested
                extracted_keywords=[],
                detected_objects=[],
                dominant_colors=[],
                style_attributes=[],
                product_type=None,
                brand_name=None,
                category=None,
                
                # System metadata
                scraped_date=datetime.now(),
                primary_image_url=primary_image_url
            )
            
            print(f"[Data Transformer] Successfully transformed Instagram post: {transformed_post.id}")
            return transformed_post
            
        except Exception as e:
            print(f"[Data Transformer] Error transforming Instagram post: {e}")
            # Return a minimal valid post in case of error
            return InstagramPostSchema(
                id=raw_data.get("id", "error"),
                type=PostType.IMAGE,
                shortCode=raw_data.get("shortCode", "error"),
                url=raw_data.get("url", ""),
                displayUrl=raw_data.get("displayUrl", ""),
                images=raw_data.get("images", []),
                ownerFullName=raw_data.get("ownerFullName", ""),
                ownerUsername=raw_data.get("ownerUsername", ""),
                ownerId=raw_data.get("ownerId", ""),
                inputUrl=raw_data.get("inputUrl", ""),
                timestamp=datetime.now(),
                scraped_date=datetime.now(),
                primary_image_url=raw_data.get("displayUrl", ""),
                extracted_keywords=[],
                detected_objects=[],
                dominant_colors=[],
                style_attributes=[],
                product_type=None,
                brand_name=None,
                category=None
            )
    
    def _transform_comment(self, comment_data: Dict[str, Any]) -> CommentSchema:
        """Transform raw comment data to CommentSchema - exact mapping only"""
        # Transform owner - exact mapping
        owner_data = comment_data.get("owner", {})
        owner = CommentOwnerSchema(
            id=owner_data.get("id", ""),
            username=owner_data.get("username", ""),
            is_verified=owner_data.get("is_verified", False),
            profile_pic_url=owner_data.get("profile_pic_url")
        )
        
        # Parse timestamp - exact mapping
        timestamp = datetime.now()
        timestamp_str = comment_data.get("timestamp")
        if timestamp_str:
            try:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
            except:
                pass
        
        return CommentSchema(
            id=comment_data.get("id", ""),
            text=comment_data.get("text", ""),
            ownerUsername=comment_data.get("ownerUsername", ""),
            ownerProfilePicUrl=comment_data.get("ownerProfilePicUrl"),
            timestamp=timestamp,
            repliesCount=comment_data.get("repliesCount", 0),
            replies=[],  # Only populate if exists in raw data
            likesCount=comment_data.get("likesCount", 0),
            owner=owner
        )
    
    def _transform_tagged_user(self, user_data: Dict[str, Any]) -> TaggedUserSchema:
        """Transform raw tagged user data to TaggedUserSchema - exact mapping only"""
        return TaggedUserSchema(
            full_name=user_data.get("full_name", ""),
            id=user_data.get("id", ""),
            is_verified=user_data.get("is_verified", False),
            profile_pic_url=user_data.get("profile_pic_url"),
            username=user_data.get("username", "")
        )
    
    def _transform_child_post(self, child_data: Dict[str, Any]) -> ChildPostSchema:
        """Transform raw child post data to ChildPostSchema - exact mapping only"""
        # Transform tagged users for child post - exact mapping
        tagged_users = []
        if child_data.get("taggedUsers"):
            for user_data in child_data.get("taggedUsers", []):
                try:
                    tagged_user = self._transform_tagged_user(user_data)
                    tagged_users.append(tagged_user)
                except Exception as e:
                    print(f"[Data Transformer] Error transforming child post tagged user: {e}")
        
        # Parse timestamp - exact mapping
        timestamp = None
        timestamp_str = child_data.get("timestamp")
        if timestamp_str:
            try:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
            except:
                pass
        
        return ChildPostSchema(
            id=child_data.get("id", ""),
            type=child_data.get("type", ""),
            caption=child_data.get("caption"),
            hashtags=child_data.get("hashtags", []),
            mentions=child_data.get("mentions", []),
            url=child_data.get("url", ""),
            commentsCount=child_data.get("commentsCount", 0),
            firstComment=child_data.get("firstComment"),
            latestComments=[],  # Only populate if exists
            dimensionsHeight=child_data.get("dimensionsHeight", 0),
            dimensionsWidth=child_data.get("dimensionsWidth", 0),
            displayUrl=child_data.get("displayUrl", ""),
            images=child_data.get("images", []),
            alt=child_data.get("alt"),
            likesCount=child_data.get("likesCount"),
            timestamp=timestamp,
            childPosts=child_data.get("childPosts", []),
            taggedUsers=tagged_users
        )
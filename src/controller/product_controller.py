"""
Product controller that handles business logic for product extraction.
Orchestrates between services and repositories.
"""

from typing import List, Optional, Dict
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from src.services.product_extraction_service import ProductExtractionService
from src.repository.product_repository import ProductRepository
from src.repository.instagram_post_repository import InstagramPostRepository
from src.schemas.product_schema import (
    ExtractProductRequest,
    ExtractProductResponse,
    BatchExtractProductRequest,
    BatchExtractProductResponse,
    ProductListResponse,
    ProductSchema,
    ProductSchemaMinimal,
    ProductItemSchema,
    ProductItemMinimalSchema
)


class ProductController:
    """Controller for product extraction operations"""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.extraction_service = ProductExtractionService()
        self.product_repository = ProductRepository(db_session)
        self.instagram_repository = InstagramPostRepository(db_session)
        print("[Product Controller] Initialized")
    
    async def extract_product_from_instagram_post(
        self, 
        request: ExtractProductRequest
    ) -> ExtractProductResponse:
        """
        Extract product from an Instagram post.
        Fetches the Instagram post from database, extracts products and items, and saves to database.
        """
        try:
            print(f"[Product Controller] Extracting product from Instagram post: {request.instagram_post_id}")
            
            # Fetch Instagram post from database
            instagram_post = await self.instagram_repository.get_post_by_id(request.instagram_post_id)
            
            if not instagram_post:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Instagram post not found with ID: {request.instagram_post_id}"
                )
            
            print(f"[Product Controller] Instagram post found: {instagram_post.url}")
            
            # Convert Instagram post model to dict for processing
            post_data = self._instagram_post_to_dict(instagram_post)
            
            # Extract product using the extraction service
            product_data = await self.extraction_service.process_instagram_post_to_product(post_data)
            
            # Save product to database
            saved_product = await self.product_repository.create_product(product_data)
            
            # Convert to Pydantic schema
            product_schema = self._product_model_to_schema(saved_product)
            
            response = ExtractProductResponse(
                success=True,
                message=f"Successfully extracted product with {len(saved_product.items)} items",
                product=product_schema,
                instagram_post_id=request.instagram_post_id,
                extracted_at=datetime.now()
            )
            
            print(f"[Product Controller] Product extraction completed: {saved_product.id}")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[Product Controller] Error extracting product: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error extracting product: {str(e)}"
            )
    
    async def batch_extract_products_from_instagram_posts(
        self,
        request: BatchExtractProductRequest
    ) -> BatchExtractProductResponse:
        """
        Extract products from multiple Instagram posts in batch.
        """
        try:
            print(f"[Product Controller] Batch extracting products from {len(request.instagram_post_ids)} posts")
            
            products = []
            errors = []
            successful_count = 0
            failed_count = 0
            
            for post_id in request.instagram_post_ids:
                try:
                    # Create individual request
                    individual_request = ExtractProductRequest(instagram_post_id=post_id)
                    
                    # Extract product
                    response = await self.extract_product_from_instagram_post(individual_request)
                    
                    if response.success and response.product:
                        products.append(response.product)
                        successful_count += 1
                    
                except HTTPException as e:
                    failed_count += 1
                    errors.append({
                        "post_id": post_id,
                        "error": str(e.detail)
                    })
                    print(f"[Product Controller] Failed to extract from post {post_id}: {e.detail}")
                except Exception as e:
                    failed_count += 1
                    errors.append({
                        "post_id": post_id,
                        "error": str(e)
                    })
                    print(f"[Product Controller] Failed to extract from post {post_id}: {str(e)}")
            
            response = BatchExtractProductResponse(
                success=successful_count > 0,
                message=f"Batch extraction completed: {successful_count} successful, {failed_count} failed",
                total_posts=len(request.instagram_post_ids),
                successful_extractions=successful_count,
                failed_extractions=failed_count,
                products=products,
                errors=errors,
                extracted_at=datetime.now()
            )
            
            print(f"[Product Controller] Batch extraction completed")
            return response
            
        except Exception as e:
            print(f"[Product Controller] Error in batch extraction: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error in batch extraction: {str(e)}"
            )
    
    async def get_product_by_id(self, product_id: int) -> ProductSchemaMinimal:
        """Get a product by ID with minimal schema (excludes embeddings)"""
        try:
            print(f"[Product Controller] Fetching product with ID: {product_id}")
            
            product = await self.product_repository.get_product_by_id(product_id)
            
            if not product:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Product not found with ID: {product_id}"
                )
            
            return self._product_model_to_minimal_schema(product)
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[Product Controller] Error fetching product: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching product: {str(e)}"
            )
    
    async def get_recent_products(self, limit: int = 50) -> ProductListResponse:
        """Get recent products with minimal schema (excludes embeddings)"""
        try:
            print(f"[Product Controller] Fetching {limit} recent products")
            
            products = await self.product_repository.get_recent_products(limit)
            
            product_schemas = [self._product_model_to_minimal_schema(p) for p in products]
            
            response = ProductListResponse(
                success=True,
                message=f"Retrieved {len(products)} products",
                total_products=len(products),
                products=product_schemas
            )
            
            return response
            
        except Exception as e:
            print(f"[Product Controller] Error fetching recent products: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching recent products: {str(e)}"
            )
    
    async def get_products_by_brand(self, brand: str, limit: int = 50) -> ProductListResponse:
        """Get products by brand with minimal schema (excludes embeddings)"""
        try:
            print(f"[Product Controller] Fetching products for brand: {brand}")
            
            products = await self.product_repository.get_products_by_brand(brand, limit)
            
            product_schemas = [self._product_model_to_minimal_schema(p) for p in products]
            
            response = ProductListResponse(
                success=True,
                message=f"Retrieved {len(products)} products for brand: {brand}",
                total_products=len(products),
                products=product_schemas
            )
            
            return response
            
        except Exception as e:
            print(f"[Product Controller] Error fetching products by brand: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching products by brand: {str(e)}"
            )
    
    def _instagram_post_to_dict(self, instagram_post) -> Dict:
        """Convert Instagram post model to dictionary for processing"""
        return {
            "id": instagram_post.id,
            "type": instagram_post.type,
            "shortCode": instagram_post.short_code,
            "url": instagram_post.url,
            "displayUrl": instagram_post.display_url,
            "images": instagram_post.images,
            "caption": instagram_post.caption,
            "alt": instagram_post.alt,
            "likesCount": instagram_post.likes_count,
            "commentsCount": instagram_post.comments_count,
            "timestamp": instagram_post.timestamp.isoformat() if instagram_post.timestamp else None,
            "dimensionsHeight": instagram_post.dimensions_height,
            "dimensionsWidth": instagram_post.dimensions_width,
            "ownerFullName": instagram_post.owner_full_name,
            "ownerUsername": instagram_post.owner_username,
            "ownerId": instagram_post.owner_id,
            "hashtags": instagram_post.hashtags,
            "mentions": instagram_post.mentions,
            "taggedUsers": instagram_post.tagged_users
        }
    
    def _product_model_to_schema(self, product) -> ProductSchema:
        """Convert Product model to ProductSchema (full schema)"""
        # Convert items to schemas
        item_schemas = []
        if product.items:  # Check if items relationship is loaded
            for item in product.items:
                item_schema = ProductItemSchema(
                    id=item.id,
                    product_id=item.product_id,
                    name=item.name,
                    brand=item.brand,
                    category=item.category,
                    style=item.style,
                    colors=item.colors,
                    product_type=item.product_type,
                    description=item.description,
                    visual_features=item.visual_features,
                    embedding=item.embedding,
                    text_embedding=item.text_embedding,
                    bounding_box=item.bounding_box,
                    confidence_score=item.confidence_score,
                    meta_info=item.meta_info,
                    created_at=item.created_at,
                    updated_at=item.updated_at
                )
                item_schemas.append(item_schema)
        
        return ProductSchema(
            id=product.id,
            name=product.name,
            description=product.description,
            image_url=product.image_url,
            source_url=product.source_url,
            brand=product.brand,
            category=product.category,
            style=product.style,
            colors=product.colors,
            caption=product.caption,
            embedding=product.embedding,
            text_embedding=product.text_embedding,
            meta_info=product.meta_info,
            items=item_schemas,
            created_at=product.created_at,
            updated_at=product.updated_at
        )
    
    def _product_model_to_minimal_schema(self, product) -> ProductSchemaMinimal:
        """Convert Product model to ProductSchemaMinimal (excludes embeddings)"""
        # Convert items to minimal schemas
        item_schemas = []
        if product.items:  # Check if items relationship is loaded
            for item in product.items:
                item_schema = ProductItemMinimalSchema(
                    id=item.id,
                    product_id=item.product_id,
                    name=item.name,
                    category=item.category,
                    style=item.style,
                    bounding_box=item.bounding_box,
                    confidence_score=item.confidence_score
                )
                item_schemas.append(item_schema)
        
        return ProductSchemaMinimal(
            id=product.id,
            name=product.name,
            description=product.description,
            image_url=product.image_url,
            source_url=product.source_url,
            brand=product.brand,
            category=product.category,
            style=product.style,
            colors=product.colors,
            caption=product.caption,
            meta_info=product.meta_info,
            items=item_schemas,
            created_at=product.created_at,
            updated_at=product.updated_at
        )
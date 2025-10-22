"""
Store item repository for database operations.
Handles all database interactions for store items.
"""

from typing import List, Optional, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_, text, bindparam
from sqlalchemy.exc import IntegrityError
import math

from src.models.store_item_model import StoreItem


class StoreItemRepository:
    """Repository for StoreItem database operations"""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        print("[Store Item Repository] Initialized")

    # ...existing code...
    async def create_store_item(self, item_data: dict) -> Optional[StoreItem]:
        """Create a new store item"""
        try:
            store_item = StoreItem(**item_data)
            self.db_session.add(store_item)
            await self.db_session.commit()
            await self.db_session.refresh(store_item)
            
            print(f"[Store Item Repository] Created store item: {store_item.sku_id}")
            return store_item

        except IntegrityError as e:
            await self.db_session.rollback()
            print(f"[Store Item Repository] Duplicate SKU: {item_data.get('sku_id')}")
            return None
        except Exception as e:
            await self.db_session.rollback()
            print(f"[Store Item Repository] Error creating store item: {str(e)}")
            raise e

    async def get_by_sku_id(self, sku_id: str) -> Optional[StoreItem]:
        """Get store item by SKU ID"""
        try:
            stmt = select(StoreItem).where(StoreItem.sku_id == sku_id)
            result = await self.db_session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            print(f"[Store Item Repository] Error fetching item by SKU: {str(e)}")
            raise e

    async def get_all_sku_ids(self) -> List[str]:
        """Get all existing SKU IDs"""
        try:
            stmt = select(StoreItem.sku_id)
            result = await self.db_session.execute(stmt)
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            print(f"[Store Item Repository] Error fetching SKU IDs: {str(e)}")
            raise e

    async def get_recent_items(self, limit: int = 50) -> List[StoreItem]:
        """Get recent store items"""
        try:
            stmt = (
                select(StoreItem)
                .order_by(StoreItem.created_at.desc())
                .limit(limit)
            )
            result = await self.db_session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            print(f"[Store Item Repository] Error fetching recent items: {str(e)}")
            raise e

    async def count_items(self) -> int:
        """Count total store items"""
        try:
            stmt = select(func.count(StoreItem.id))
            result = await self.db_session.execute(stmt)
            return result.scalar_one()
        except Exception as e:
            print(f"[Store Item Repository] Error counting items: {str(e)}")
            raise e
        
    async def get_paginated_items(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[StoreItem], int]:
        """
        Get paginated store items with filters.
        Returns (items, total_count)
        """
        try:
            # Build base query
            query = select(StoreItem)
            count_query = select(func.count(StoreItem.id))
            
            # Apply filters
            filter_conditions = []
            
            if filters:
                # Category filter
                if filters.get("category"):
                    filter_conditions.append(
                        StoreItem.category.in_(filters["category"])
                    )
                
                # Brand name filter
                if filters.get("brand_name"):
                    filter_conditions.append(
                        StoreItem.brand_name.in_(filters["brand_name"])
                    )
                
                # Product type filter
                if filters.get("product_type"):
                    filter_conditions.append(
                        StoreItem.product_type.in_(filters["product_type"])
                    )
                
                # Gender filter
                if filters.get("gender"):
                    filter_conditions.append(
                        StoreItem.gender.in_(filters["gender"])
                    )
                
                # Price range filter
                if filters.get("min_price") is not None:
                    filter_conditions.append(
                        StoreItem.lowest_price >= filters["min_price"]
                    )
                
                if filters.get("max_price") is not None:
                    filter_conditions.append(
                        StoreItem.lowest_price <= filters["max_price"]
                    )
                
                # Search query (title, description, tags)
                if filters.get("search_query"):
                    search_term = f"%{filters['search_query']}%"
                    filter_conditions.append(
                        or_(
                            StoreItem.title.ilike(search_term),
                            StoreItem.description.ilike(search_term),
                            StoreItem.tags.ilike(search_term)
                        )
                    )
            
            # Apply all filters
            if filter_conditions:
                query = query.where(and_(*filter_conditions))
                count_query = count_query.where(and_(*filter_conditions))
            
            # Get total count
            total_result = await self.db_session.execute(count_query)
            total_count = total_result.scalar() or 0
            
            # Apply sorting
            sort_column = getattr(StoreItem, sort_by, StoreItem.created_at)
            if sort_order.lower() == "asc":
                query = query.order_by(sort_column.asc())
            else:
                query = query.order_by(sort_column.desc())
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = await self.db_session.execute(query)
            items = result.scalars().all()
            
            return list(items), total_count
            
        except Exception as e:
            print(f"[Store Item Repository] Error in get_paginated_items: {str(e)}")
            raise
    
    async def get_unique_brands(self) -> List[str]:
        """Get all unique brand names"""
        try:
            query = select(StoreItem.brand_name).distinct().where(
                StoreItem.brand_name.isnot(None)
            ).order_by(StoreItem.brand_name)
            
            result = await self.db_session.execute(query)
            brands = [row[0] for row in result.fetchall() if row[0]]
            
            return brands
            
        except Exception as e:
            print(f"[Store Item Repository] Error getting unique brands: {str(e)}")
            return []
    
    async def get_price_range(self) -> Dict[str, Optional[float]]:
        """Get min and max price in the database"""
        try:
            query = select(
                func.min(StoreItem.lowest_price),
                func.max(StoreItem.lowest_price)
            ).where(StoreItem.lowest_price.isnot(None))
            
            result = await self.db_session.execute(query)
            min_price, max_price = result.one()
            
            return {
                "min_price": float(min_price) if min_price else None,
                "max_price": float(max_price) if max_price else None
            }
            
        except Exception as e:
            print(f"[Store Item Repository] Error getting price range: {str(e)}")
            return {"min_price": None, "max_price": None}

    async def find_similar_items_by_visual_embedding(
        self,
        visual_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[StoreItem, float]]:
        """
        Find similar store items based on visual embedding using cosine similarity.
        Returns list of tuples (StoreItem, similarity_score)
        """
        try:
            print(f"[Store Item Repository] Finding similar items with limit {limit}")
            print(f"[Store Item Repository] Query embedding dimension: {len(visual_embedding)}")
            print(f"[Store Item Repository] Similarity threshold: {similarity_threshold}")
            
            # First, check what embeddings exist in the database
            check_query = text("""
                SELECT 
                    COUNT(*) as total_items,
                    COUNT(visual_embedding) as items_with_embeddings,
                    COUNT(CASE WHEN visual_embedding IS NOT NULL THEN 1 END) as non_null_embeddings
                FROM store_items
            """)
            
            check_result = await self.db_session.execute(check_query)
            stats = check_result.fetchone()
            print(f"[Store Item Repository] Database stats: Total={stats.total_items}, With embeddings={stats.items_with_embeddings}")
            
            # Check embedding dimensions in database
            dim_query = text("""
                SELECT vector_dims(visual_embedding) as dims
                FROM store_items
                WHERE visual_embedding IS NOT NULL
                LIMIT 1
            """)
            
            try:
                dim_result = await self.db_session.execute(dim_query)
                dim_row = dim_result.fetchone()
                if dim_row:
                    print(f"[Store Item Repository] Database embedding dimension: {dim_row.dims}")
            except Exception as dim_error:
                print(f"[Store Item Repository] Could not check embedding dimension: {dim_error}")
            
            # Convert embedding to string format for pgvector
            embedding_str = "[" + ",".join(map(str, visual_embedding)) + "]"
            
            # Try without threshold first to see if there are ANY matches
            test_query_sql = f"""
                SELECT 
                    COUNT(*) as match_count,
                    MIN(1 - (visual_embedding <=> '{embedding_str}'::vector)) as min_similarity,
                    MAX(1 - (visual_embedding <=> '{embedding_str}'::vector)) as max_similarity,
                    AVG(1 - (visual_embedding <=> '{embedding_str}'::vector)) as avg_similarity
                FROM store_items
                WHERE visual_embedding IS NOT NULL
            """
            
            test_stmt = text(test_query_sql)
            test_result = await self.db_session.execute(test_stmt)
            test_row = test_result.fetchone()
            
            if test_row and test_row.match_count > 0:
                # Format safely with None checks
                min_sim = f"{test_row.min_similarity:.4f}" if test_row.min_similarity is not None else "N/A"
                max_sim = f"{test_row.max_similarity:.4f}" if test_row.max_similarity is not None else "N/A"
                avg_sim = f"{test_row.avg_similarity:.4f}" if test_row.avg_similarity is not None else "N/A"
                
                print(f"[Store Item Repository] Similarity stats: Count={test_row.match_count}, "
                      f"Min={min_sim}, Max={max_sim}, Avg={avg_sim}")
                
                # Dynamically adjust threshold if max similarity is below requested threshold
                if test_row.max_similarity is not None and test_row.max_similarity < similarity_threshold:
                    adjusted_threshold = max(0.0, test_row.max_similarity - 0.1)  # Use 90% of max similarity
                    print(f"[Store Item Repository] WARNING: Max similarity ({test_row.max_similarity:.4f}) is below threshold ({similarity_threshold:.4f})")
                    print(f"[Store Item Repository] Adjusting threshold to {adjusted_threshold:.4f} to return results")
                    similarity_threshold = adjusted_threshold
            else:
                print(f"[Store Item Repository] WARNING: No items with embeddings found or similarity calculation failed")
            
            # Build actual query - remove threshold constraint if it's too restrictive
            if similarity_threshold <= 0:
                # No threshold - just return top K by similarity
                query_sql = f"""
                    SELECT 
                        id, sku_id, title, slug, category, sub_category,
                        brand_name, product_type, gender, colorways, brand_sku, model,
                        lowest_price, description, is_d2c, is_active, is_certificate_required,
                        featured_image, pdp_url, quantity_left, wishlist_num, 
                        stock_claimed_percent, discount_percentage, note, tags,
                        release_date, csv_created_at, csv_updated_at, created_at, updated_at,
                        1 - (visual_embedding <=> '{embedding_str}'::vector) as similarity
                    FROM store_items
                    WHERE visual_embedding IS NOT NULL
                    ORDER BY visual_embedding <=> '{embedding_str}'::vector
                    LIMIT :limit
                """
                stmt = text(query_sql).bindparams(bindparam("limit"))
                params = {"limit": limit}
            else:
                # With threshold
                query_sql = f"""
                    SELECT 
                        id, sku_id, title, slug, category, sub_category,
                        brand_name, product_type, gender, colorways, brand_sku, model,
                        lowest_price, description, is_d2c, is_active, is_certificate_required,
                        featured_image, pdp_url, quantity_left, wishlist_num, 
                        stock_claimed_percent, discount_percentage, note, tags,
                        release_date, csv_created_at, csv_updated_at, created_at, updated_at,
                        1 - (visual_embedding <=> '{embedding_str}'::vector) as similarity
                    FROM store_items
                    WHERE visual_embedding IS NOT NULL
                      AND 1 - (visual_embedding <=> '{embedding_str}'::vector) >= :threshold
                    ORDER BY visual_embedding <=> '{embedding_str}'::vector
                    LIMIT :limit
                """
                stmt = text(query_sql).bindparams(
                    bindparam("threshold"),
                    bindparam("limit")
                )
                params = {
                    "threshold": similarity_threshold,
                    "limit": limit
                }
            
            # Execute with parameters
            result = await self.db_session.execute(stmt, params)
            
            rows = result.fetchall()
            
            # Convert rows to StoreItem objects with similarity scores
            similar_items = []
            for row in rows:
                # Access row fields by name
                item = StoreItem(
                    id=row.id,
                    sku_id=row.sku_id,
                    title=row.title,
                    slug=row.slug,
                    category=row.category,
                    sub_category=row.sub_category,
                    brand_name=row.brand_name,
                    product_type=row.product_type,
                    gender=row.gender,
                    colorways=row.colorways,
                    brand_sku=row.brand_sku,
                    model=row.model,
                    lowest_price=row.lowest_price,
                    description=row.description,
                    is_d2c=row.is_d2c,
                    is_active=row.is_active,
                    is_certificate_required=row.is_certificate_required,
                    featured_image=row.featured_image,
                    pdp_url=row.pdp_url,
                    quantity_left=row.quantity_left,
                    wishlist_num=row.wishlist_num,
                    stock_claimed_percent=row.stock_claimed_percent,
                    discount_percentage=row.discount_percentage,
                    note=row.note,
                    tags=row.tags,
                    release_date=row.release_date,
                    csv_created_at=row.csv_created_at,
                    csv_updated_at=row.csv_updated_at,
                    created_at=row.created_at,
                    updated_at=row.updated_at
                )
                
                similarity_score = float(row.similarity)
                similar_items.append((item, similarity_score))
            
            print(f"[Store Item Repository] Found {len(similar_items)} similar items (threshold: {similarity_threshold:.4f})")
            
            if len(similar_items) == 0:
                print(f"[Store Item Repository] WARNING: No items found. This might indicate:")
                print(f"  - Different embedding models used for product_items vs store_items")
                print(f"  - Embeddings not normalized")
                print(f"  - Query embedding has issues")
            
            return similar_items
            
        except Exception as e:
            print(f"[Store Item Repository] Error finding similar items: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise e
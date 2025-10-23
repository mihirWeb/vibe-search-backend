"""
Hybrid search service for store items.
Combines vector similarity search with keyword-based search and intelligent reranking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sqlalchemy import text, bindparam
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import ARRAY
from src.utils.query_parser import QwenQueryParser
from datetime import datetime
import asyncio
import logging

from src.utils.image_processing import (
    download_image,
    generate_visual_embedding,
    generate_text_embedding
)

logger = logging.getLogger(__name__)


class QueryExpander:
    """Handles query expansion and semantic understanding"""
    
    def __init__(self):
        self.synonym_mappings = {
            "beach": ["coastal", "shore", "summer", "vacation", "swim", "water"],
            "shorts": ["trunks", "swimwear", "bottoms", "casual"],
            "shoes": ["footwear", "sneakers", "boots", "trainers"],
            "tracking": ["hiking", "trekking", "trail", "outdoor", "mountain"],
            "running": ["jogging", "athletic", "fitness", "sport"],
            "casual": ["everyday", "relaxed", "comfortable", "informal"],
            "formal": ["business", "professional", "dress", "elegant"],
            "summer": ["warm", "lightweight", "breathable", "sunny"],
            "winter": ["warm", "insulated", "cold", "snow"],
            "shirt": ["tee", "top", "blouse"],
            "pants": ["trousers", "jeans", "slacks"],
            "jacket": ["coat", "outerwear", "hoodie"]
        }
    
    def expand_query(self, query: str) -> Dict:
        """Expand user query with synonyms and related terms"""
        query_lower = query.lower()
        expanded_terms = [query]
        extracted_keywords = []
        
        # Extract and expand keywords
        for keyword, synonyms in self.synonym_mappings.items():
            if keyword in query_lower:
                extracted_keywords.append(keyword)
                expanded_terms.extend(synonyms)
        
        # Remove duplicates and limit terms
        expanded_terms = list(set(expanded_terms))[:10]
        
        return {
            "original_query": query,
            "extracted_keywords": extracted_keywords,
            "expanded_terms": expanded_terms,
            "inferred_context": self._infer_context(query, extracted_keywords)
        }
    
    def _infer_context(self, query: str, keywords: List[str]) -> Dict:
        """Infer search context from query and keywords"""
        context = {
            "category": None,
            "use_case": None,
            "style": None,
            "season": None
        }
        
        query_lower = query.lower()
        
        # Category inference
        if any(word in query_lower for word in ["shorts", "trunks", "bottoms"]):
            context["category"] = "Bottoms"
        elif any(word in query_lower for word in ["shoes", "footwear", "sneakers", "boots"]):
            context["category"] = "Footwear"
        elif any(word in query_lower for word in ["shirt", "tee", "top"]):
            context["category"] = "Apparel"
        
        # Use case inference
        if any(word in query_lower for word in ["beach", "vacation", "swim", "summer"]):
            context["use_case"] = "beach/vacation/summer"
        elif any(word in query_lower for word in ["tracking", "hiking", "trekking", "outdoor"]):
            context["use_case"] = "outdoor/hiking"
        elif any(word in query_lower for word in ["running", "athletic", "fitness"]):
            context["use_case"] = "athletic/fitness"
        
        # Style inference
        if any(word in query_lower for word in ["casual", "comfortable", "relaxed"]):
            context["style"] = "casual"
        elif any(word in query_lower for word in ["formal", "business", "professional"]):
            context["style"] = "formal"
        
        # Season inference
        if any(word in query_lower for word in ["summer", "beach", "vacation"]):
            context["season"] = "summer"
        elif any(word in query_lower for word in ["winter", "cold", "snow"]):
            context["season"] = "winter"
        
        return context


class VectorSearchService:
    """Handles vector similarity search operations"""
    
    # Category and product type mappings for strict filtering
    CATEGORY_MAPPINGS = {
        "top": ["apparel", "streetwear", "clothing"],
        "bottom": ["apparel", "streetwear"],
        "shoes": ["sneakers", "shoes"],
        "hat": ["accessories"],
        "watch": ["watches"],
        "bag": ["handbags", "bag", "backpack", "accessories"]
    }
    
    PRODUCT_TYPE_MAPPINGS = {
        "top": ["clothing", "streetwear", "tops"],
        "bottom": ["clothing", "streetwear", "bottoms"],
        "shoes": ["sneakers", "shoes"],
        "hat": ["accessories", "hat"],
        "watch": ["watches", "accessories"],
        "bag": ["handbags", "accessories"]
    }
    
    async def search_by_text_embedding_with_prefilter(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        item_type: str,
        top_k: int = 10,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Two-stage search:
        1. Filter by category and product_type for the item_type
        2. Apply embedding search on filtered results
        
        Args:
            db: Database session
            query_embedding: Query embedding vector
            item_type: Type of item (top, bottom, shoes, hat, watch, bag)
            top_k: Number of results to return
            filters: Additional filters (brands, gender, price_range)
        """
        try:
            # Get allowed categories and product types for this item type
            allowed_categories = self.CATEGORY_MAPPINGS.get(item_type, [])
            allowed_product_types = self.PRODUCT_TYPE_MAPPINGS.get(item_type, [])
            
            if not allowed_categories and not allowed_product_types:
                logger.warning(f"[Pre-filter Search] No category/product_type mappings for: {item_type}")
                return []
            
            logger.debug(f"[Pre-filter Search] Item type: {item_type}")
            logger.debug(f"[Pre-filter Search] Allowed categories: {allowed_categories}")
            logger.debug(f"[Pre-filter Search] Allowed product types: {allowed_product_types}")
            
            # Convert embedding to PostgreSQL vector string format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Build query with category and product_type pre-filtering
            query_parts = [f"""
                SELECT 
                    id, sku_id, title, description, category, sub_category,
                    brand_name, product_type, gender, colorways, lowest_price,
                    featured_image, pdp_url, wishlist_num, tags,
                    1 - (textual_embedding <=> '{embedding_str}'::vector) as similarity_score
                FROM store_items
                WHERE textual_embedding IS NOT NULL
            """]
            
            bindparams_list = [bindparam("top_k")]
            params = {"top_k": top_k * 3}  # Get more results for filtering
            
            # Add category and product_type filters
            if allowed_categories:
                category_conditions = []
                for i, cat in enumerate(allowed_categories):
                    cat_param = f"cat_{i}"
                    category_conditions.append(f"category ILIKE :{cat_param}")
                    params[cat_param] = f"%{cat}%"
                    bindparams_list.append(bindparam(cat_param))
                
                # ✅ Category must be non-null and match allowed categories
                category_filter = f"(category IS NOT NULL AND ({' OR '.join(category_conditions)}))"
                query_parts.append(f" AND {category_filter}")

            if allowed_product_types:
                product_type_conditions = []
                for i, ptype in enumerate(allowed_product_types):
                    ptype_param = f"ptype_{i}"
                    product_type_conditions.append(f"product_type ILIKE :{ptype_param}")
                    params[ptype_param] = f"%{ptype}%"
                    bindparams_list.append(bindparam(ptype_param))
                
                # ✅ Product type can either match or be NULL
                product_type_filter = f"(({' OR '.join(product_type_conditions)}) OR product_type IS NULL)"
                query_parts.append(f" AND {product_type_filter}")
            
            # Add additional filters (brands, gender, price_range)
            if filters:
                if filters.get("brands"):
                    query_parts.append(" AND brand_name = ANY(:brands)")
                    params["brands"] = filters["brands"]
                    bindparams_list.append(bindparam("brands"))
                
                if filters.get("gender"):
                    query_parts.append(" AND gender = :gender")
                    params["gender"] = filters["gender"]
                    bindparams_list.append(bindparam("gender"))
                
                if filters.get("price_range") and len(filters["price_range"]) == 2:
                    query_parts.append(" AND lowest_price BETWEEN :min_price AND :max_price")
                    params["min_price"] = filters["price_range"][0]
                    params["max_price"] = filters["price_range"][1]
                    bindparams_list.extend([bindparam("min_price"), bindparam("max_price")])
            
            # Order by similarity and limit
            query_parts.append(" ORDER BY similarity_score DESC LIMIT :top_k")
            query_sql = "".join(query_parts)
            
            logger.debug(f"[Pre-filter Search] Executing filtered search for {item_type}")
            
            stmt = text(query_sql).bindparams(*bindparams_list)
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
            results = [dict(row._mapping) for row in rows]
            
            logger.debug(f"[Pre-filter Search] Found {len(results)} results for {item_type}")
            
            # Additional Python-level verification
            verified_results = []
            for result in results:
                result_category = (result.get("category") or "").lower()
                result_product_type = (result.get("product_type") or "").lower()
                
                # Check if category matches any allowed category (partial match)
                category_match = any(
                    cat.lower() in result_category or result_category in cat.lower()
                    for cat in allowed_categories
                )
                
                # Check if product_type matches any allowed product type (partial match)
                product_type_match = any(
                    ptype.lower() in result_product_type or result_product_type in ptype.lower()
                    for ptype in allowed_product_types
                )
                
                # Item must match EITHER category OR product_type
                if category_match or product_type_match:
                    verified_results.append(result)
                else:
                    logger.debug(f"[Pre-filter Search] Filtered out: {result.get('title')} - category: {result_category}, product_type: {result_product_type}")
            
            logger.debug(f"[Pre-filter Search] After verification: {len(verified_results)} results")
            
            return verified_results[:top_k]
            
        except Exception as e:
            logger.error(f"Pre-filter search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    # Keep the original search methods for backward compatibility
    async def search_by_text_embedding(
        self, 
        db: AsyncSession, 
        query_embedding: List[float], 
        top_k: int = 10, 
        filters: Dict = None
    ) -> List[Dict]:
        """Search using text embedding similarity with proper bindparam"""
        try:
            # Convert embedding to PostgreSQL vector string format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Build query - embed the vector string directly in SQL
            query_parts = [f"""
                SELECT 
                    id, sku_id, title, description, category, sub_category,
                    brand_name, product_type, gender, colorways, lowest_price,
                    featured_image, pdp_url, wishlist_num, tags,
                    1 - (textual_embedding <=> '{embedding_str}'::vector) as similarity_score
                FROM store_items
                WHERE textual_embedding IS NOT NULL
            """]
            
            # Prepare bindparams list
            bindparams_list = [bindparam("top_k")]
            params = {"top_k": top_k}
            
            # Add filters with named parameters
            if filters:
                if filters.get("category"):
                    categories = filters["category"]
                    # Build flexible category matching using ILIKE for partial matches
                    category_conditions = []
                    for i, cat in enumerate(categories):
                        cat_param = f"cat_{i}"
                        category_conditions.append(f"category ILIKE :{cat_param}")
                        params[cat_param] = f"%{cat}%"
                        bindparams_list.append(bindparam(cat_param))
                    
                    query_parts.append(f" AND ({' OR '.join(category_conditions)})")
                
                if filters.get("brands"):
                    query_parts.append(" AND brand_name = ANY(:brands)")
                    params["brands"] = filters["brands"]
                    bindparams_list.append(bindparam("brands"))
                
                if filters.get("gender"):
                    query_parts.append(" AND gender = :gender")
                    params["gender"] = filters["gender"]
                    bindparams_list.append(bindparam("gender"))
                
                if filters.get("price_range") and len(filters["price_range"]) == 2:
                    query_parts.append(" AND lowest_price BETWEEN :min_price AND :max_price")
                    params["min_price"] = filters["price_range"][0]
                    params["max_price"] = filters["price_range"][1]
                    bindparams_list.extend([bindparam("min_price"), bindparam("max_price")])
            
            query_parts.append(" ORDER BY similarity_score DESC LIMIT :top_k")
            query_sql = "".join(query_parts)
            
            # Use logger.debug instead of print for SQL queries
            logger.debug(f"Text embedding search query filters: {filters}")
            
            stmt = text(query_sql).bindparams(*bindparams_list)
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
            logger.debug(f"[Text Embedding Search] Found {len(rows)} results")
            
            return [dict(row._mapping) for row in rows]
            
        except Exception as e:
            logger.error(f"Text embedding search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    async def search_by_visual_embedding(
        self, 
        db: AsyncSession, 
        query_embedding: List[float],
        top_k: int = 10, 
        filters: Dict = None
    ) -> List[Dict]:
        """Search using visual embedding similarity with proper bindparam"""
        try:
            # Convert embedding to PostgreSQL vector string format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Build query - embed the vector string directly in SQL
            query_parts = [f"""
                SELECT 
                    id, sku_id, title, description, category, sub_category,
                    brand_name, product_type, gender, colorways, lowest_price,
                    featured_image, pdp_url, wishlist_num, tags,
                    1 - (visual_embedding <=> '{embedding_str}'::vector) as similarity_score
                FROM store_items
                WHERE visual_embedding IS NOT NULL
            """]
            
            bindparams_list = [bindparam("top_k")]
            params = {"top_k": top_k}
            
            # Add filters
            if filters:
                if filters.get("category"):
                    categories = filters["category"]
                    category_conditions = []
                    for i, cat in enumerate(categories):
                        cat_param = f"cat_{i}"
                        category_conditions.append(f"category ILIKE :{cat_param}")
                        params[cat_param] = f"%{cat}%"
                        bindparams_list.append(bindparam(cat_param))
                    query_parts.append(f" AND ({' OR '.join(category_conditions)})")
                
                if filters.get("brands"):
                    query_parts.append(" AND brand_name = ANY(:brands)")
                    params["brands"] = filters["brands"]
                    bindparams_list.append(bindparam("brands"))
                
                if filters.get("colors"):
                    color_conditions = []
                    for i, color in enumerate(filters["colors"]):
                        color_param = f"color_{i}"
                        color_conditions.append(f"colorways ILIKE :{color_param}")
                        params[color_param] = f"%{color}%"
                        bindparams_list.append(bindparam(color_param))
                    query_parts.append(f" AND ({' OR '.join(color_conditions)})")
                
                if filters.get("price_range") and len(filters["price_range"]) == 2:
                    query_parts.append(" AND lowest_price BETWEEN :min_price AND :max_price")
                    params["min_price"] = filters["price_range"][0]
                    params["max_price"] = filters["price_range"][1]
                    bindparams_list.extend([bindparam("min_price"), bindparam("max_price")])
            
            query_parts.append(" ORDER BY similarity_score DESC LIMIT :top_k")
            query_sql = "".join(query_parts)
            
            stmt = text(query_sql).bindparams(*bindparams_list)
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
            logger.debug(f"[Visual Embedding Search] Found {len(rows)} results")
            
            return [dict(row._mapping) for row in rows]
            
        except Exception as e:
            logger.error(f"Visual embedding search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


class BM25SearchService:
    """Handles BM25 keyword-based search"""
    
    async def search_by_keywords(
        self, 
        db: AsyncSession, 
        keywords: List[str], 
        top_k: int = 10, 
        filters: Dict = None
    ) -> List[Dict]:
        """Search using BM25-style keyword matching with proper bindparam"""
        try:
            if not keywords:
                return []
            
            # Limit to 5 keywords
            keywords = keywords[:5]
            
            # Build search conditions using named params
            search_conditions = []
            params = {"top_k": top_k}
            bindparams_list = [bindparam("top_k")]
            
            for i, keyword in enumerate(keywords):
                keyword_param = f"kw_{i}"
                search_conditions.extend([
                    f"title ILIKE :{keyword_param}",
                    f"description ILIKE :{keyword_param}",
                    f"tags ILIKE :{keyword_param}",
                    f"brand_name ILIKE :{keyword_param}"
                ])
                params[keyword_param] = f"%{keyword}%"
                bindparams_list.append(bindparam(keyword_param))
            
            # Build base query with CASE for scoring
            query_parts = [f"""
                SELECT 
                    id, sku_id, title, description, category, sub_category,
                    brand_name, product_type, gender, colorways, lowest_price,
                    featured_image, pdp_url, wishlist_num, tags,
                    CASE WHEN {' OR '.join(search_conditions)} THEN 1.0 ELSE 0.0 END as keyword_score
                FROM store_items
                WHERE ({' OR '.join(search_conditions)})
            """]
            
            # Add filters
            if filters:
                if filters.get("category"):
                    query_parts.append(" AND category = ANY(:categories)")
                    params["categories"] = filters["category"]
                    bindparams_list.append(bindparam("categories"))
                
                if filters.get("brands"):
                    query_parts.append(" AND brand_name = ANY(:brands)")
                    params["brands"] = filters["brands"]
                    bindparams_list.append(bindparam("brands"))
                
                if filters.get("gender"):
                    query_parts.append(" AND gender = :gender")
                    params["gender"] = filters["gender"]
                    bindparams_list.append(bindparam("gender"))
                
                if filters.get("price_range") and len(filters["price_range"]) == 2:
                    query_parts.append(" AND lowest_price BETWEEN :min_price AND :max_price")
                    params["min_price"] = filters["price_range"][0]
                    params["max_price"] = filters["price_range"][1]
                    bindparams_list.extend([bindparam("min_price"), bindparam("max_price")])
            
            # Add ordering and limit
            query_parts.append(" ORDER BY keyword_score DESC, wishlist_num DESC LIMIT :top_k")
            
            query_sql = "".join(query_parts)
            
            # Create text() with bindparams
            stmt = text(query_sql).bindparams(*bindparams_list)
            
            # Execute with dictionary parameters
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
            return [dict(row._mapping) for row in rows]
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


class ReRanker:
    """Handles reranking of search results"""
    
    def rerank_results(
        self, 
        results: List[Dict], 
        query_info: Dict, 
        mode: str = "hybrid"
    ) -> List[Dict]:
        """Rerank search results based on multiple factors"""
        try:
            for result in results:
                # Calculate combined score
                if mode == "hybrid":
                    # Combine semantic and keyword scores
                    semantic_score = result.get("similarity_score", 0)
                    keyword_score = result.get("keyword_score", 0)
                    popularity_score = min(result.get("wishlist_num", 0) / 1000.0, 1.0)
                    
                    # Weighted combination
                    result["combined_score"] = (
                        0.6 * semantic_score + 
                        0.3 * keyword_score + 
                        0.1 * popularity_score
                    )
                elif mode == "visual":
                    similarity_score = result.get("similarity_score", 0)
                    popularity_score = min(result.get("wishlist_num", 0) / 1000.0, 1.0)
                    result["combined_score"] = 0.9 * similarity_score + 0.1 * popularity_score
                elif mode == "textual":
                    result["combined_score"] = result.get("similarity_score", 0)
                
                # Generate match reasons
                result["match_reasons"] = self._generate_match_reasons(result, query_info)
            
            # Sort by combined score
            return sorted(results, key=lambda x: x.get("combined_score", 0), reverse=True)
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results
    
    def _generate_match_reasons(self, result: Dict, query_info: Dict) -> List[str]:
        """Generate human-readable match reasons"""
        reasons = []
        
        # Title matching
        if query_info.get("extracted_keywords"):
            title_lower = (result.get("title") or "").lower()
            for keyword in query_info["extracted_keywords"]:
                if keyword in title_lower:
                    reasons.append(f"Title contains '{keyword}'")
        
        # Context-based matching
        inferred_context = query_info.get("inferred_context", {})
        
        # Category matching
        if inferred_context.get("category") and result.get("category") == inferred_context["category"]:
            reasons.append(f"Matches category: {inferred_context['category']}")
        elif result.get("category"):
            reasons.append(f"Category: {result.get('category')}")
        
        # Use case matching
        if inferred_context.get("use_case"):
            description_lower = (result.get("description") or "").lower()
            product_type_lower = (result.get("product_type") or "").lower()
            if any(term in description_lower or term in product_type_lower 
                   for term in inferred_context["use_case"].split("/")):
                reasons.append(f"Suitable for: {inferred_context['use_case']}")
        
        # Style matching
        if inferred_context.get("style"):
            product_type_lower = (result.get("product_type") or "").lower()
            if inferred_context["style"] in product_type_lower:
                reasons.append(f"Style: {inferred_context['style']}")
        
        # Brand matching
        brand = result.get("brand_name")
        if brand:
            reasons.append(f"Brand: {brand}")
        
        # Price appropriateness
        price = result.get("lowest_price")
        if price:
            price_val = float(price)
            if price_val < 50:
                reasons.append("Budget-friendly")
            elif price_val < 150:
                reasons.append("Mid-range pricing")
            else:
                reasons.append("Premium product")
        
        # Popularity
        wishlist_count = result.get("wishlist_num", 0)
        if wishlist_count > 100:
            reasons.append("Popular item")
        
        return reasons if reasons else ["General match"]


class HybridSearchService:
    """Main service that orchestrates hybrid search"""
    
    def __init__(self):
        self.query_expander = QueryExpander()
        self.vector_search = VectorSearchService()
        self.bm25_search = BM25SearchService()
        self.reranker = ReRanker()
        self.query_parser = QwenQueryParser()  # Add AI query parser
        
        # Comprehensive clothing items mapping
        self.clothing_items = {
            "men": {
                "top": ["t-shirt", "polo shirt", "dress shirt", "casual shirt", "hoodie", "sweatshirt", "sweater", "vest", "jacket", "blazer", "coat", "tank top", "henley", "flannel shirt"],
                "bottom": ["jeans", "trousers", "chinos", "shorts", "sweatpants", "cargo pants", "khakis", "dress pants", "joggers", "track pants"],
                "hat": ["baseball cap", "beanie", "fedora", "flat cap", "bucket hat", "beret", "trilby", "top hat", "newsboy cap"],
                "shoes": ["sneakers", "dress shoes", "loafers", "boots", "sandals", "flip-flops", "oxfords", "brogues", "boat shoes", "slippers", "high-tops"],
                "watch": ["analog watch", "digital watch", "smartwatch", "chronograph", "dive watch", "dress watch", "field watch", "pilot watch", "fitness tracker"],
                "bag": ["backpack", "messenger bag", "briefcase", "duffel bag", "tote bag", "gym bag", "sling bag", "laptop bag", "waist bag"]
            },
            "women": {
                "top": ["t-shirt", "blouse", "tank top", "crop top", "sweater", "hoodie", "cardigan", "camisole", "bodysuit", "peplum top", "tube top", "halter top", "off-shoulder top"],
                "bottom": ["jeans", "trousers", "leggings", "skirt", "shorts", "culottes", "palazzo pants", "yoga pants", "jeggings", "capris", "skirt", "dress"],
                "hat": ["baseball cap", "beanie", "fedora", "sun hat", "beret", "fascinator", "wide-brimmed hat", "cloche", "visor", "turban"],
                "shoes": ["sneakers", "heels", "flats", "boots", "sandals", "wedges", "pumps", "stilettos", "platform shoes", "ballet flats", "mules", "espadrilles"],
                "watch": ["analog watch", "digital watch", "smartwatch", "bracelet watch", "pendant watch", "dress watch", "sport watch", "fitness tracker", "charm watch"],
                "bag": ["handbag", "clutch", "tote bag", "backpack", "crossbody bag", "shoulder bag", "hobo bag", "satchel", "bucket bag", "purse", "wristlet", "evening bag"]
            }
        }
                
        print("[Hybrid Search Service] Initialized")

    async def search_by_text(
        self, 
        db: AsyncSession, 
        query: str, 
        top_k: int = 10, 
        filters: Dict = None, 
        rerank: bool = True,
        use_ai_parser: bool = True
    ) -> Dict:
        """Perform text-based hybrid search with AI query parsing"""
        start_time = datetime.now()
        
        try:
            # print(f"[Hybrid Search] Text search for: {query}")
            
            # Step 1: AI-powered query parsing (if enabled)
            ai_parsed_query = None
            refined_query = query
            ai_filters = {}
            is_collection_query = False
            existing_items = {}
            
            if use_ai_parser:
                try:
                    ai_parsed_query = self.query_parser.parse_query(query)
                    refined_query = ai_parsed_query.get("refined_query", query)
                    ai_filters = ai_parsed_query.get("filters", {})
                    is_collection_query = ai_parsed_query.get("is_collection_query", False)
                    existing_items = ai_parsed_query.get("existing_items", {})
                    print(f"[Hybrid Search] AI parsed query: {ai_parsed_query}")
                    print(f"[Hybrid Search] Is collection query: {is_collection_query}")
                except Exception as e:
                    logger.error(f"[Hybrid Search] AI parsing failed, using original query: {e}")
                    refined_query = query
            
            # Step 2: Route to collection or normal search
            if is_collection_query:
                return await self._search_collection(
                    db=db,
                    refined_query=refined_query,
                    ai_parsed_query=ai_parsed_query,
                    filters=filters,
                    rerank=rerank,
                    start_time=start_time
                )
            else:
                # Normal search flow (existing code)
                return await self._search_normal(
                    db=db,
                    refined_query=refined_query,
                    ai_parsed_query=ai_parsed_query,
                    filters=filters,
                    top_k=top_k,
                    rerank=rerank,
                    start_time=start_time
                )
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "query_understanding": {"original_query": query, "error": str(e)},
                "matches": [],
                "search_strategy": "failed",
                "total_results": 0,
                "search_time_ms": 0,
                "is_collection_query": False
            }
    async def _search_normal(
        self,
        db: AsyncSession,
        refined_query: str,
        ai_parsed_query: Dict,
        filters: Dict,
        top_k: int,
        rerank: bool,
        start_time: datetime
    ) -> Dict:
        """Normal search flow with multi-step pre-filtering"""
        ai_filters = ai_parsed_query.get("filters", {}) if ai_parsed_query else {}
        
        # Merge filters
        merged_filters = self._merge_filters(filters, ai_filters)
        print(f"[Hybrid Search] Merged filters: {merged_filters}")
        
        # Query understanding and expansion
        query_analysis = self.query_expander.expand_query(refined_query)
        
        # Add AI parsing info
        if ai_parsed_query:
            query_analysis["ai_parsing"] = {
                "original_query": ai_parsed_query.get("original_query"),
                "refined_query": refined_query,
                "ai_filters": ai_filters,
                "explanation": ai_parsed_query.get("explanation")
            }
        
        print(f"[Hybrid Search] Query analysis: {query_analysis}")
        
        # Build enriched query
        enriched_query_text = self._build_enriched_query(refined_query, merged_filters, ai_filters)
        print(f"[Hybrid Search] Enriched query for embedding: {enriched_query_text}")
        
        # Generate query embedding
        query_embedding = generate_text_embedding(enriched_query_text)
        
        # Pad to 768 dimensions
        if len(query_embedding) < 768:
            query_embedding = query_embedding + [0.0] * (768 - len(query_embedding))
        query_embedding = query_embedding[:768]
        
        # ✅ NEW: Check if type filter exists for pre-filtering
        item_types = ai_filters.get("type", [])
        
        if item_types:
            # Use pre-filter search for each item type
            print(f"[Hybrid Search] Using pre-filter search for types: {item_types}")
            
            semantic_results = []
            for item_type in item_types:
                type_results = await self.vector_search.search_by_text_embedding_with_prefilter(
                    db=db,
                    query_embedding=query_embedding,
                    item_type=item_type,
                    top_k=top_k * 2,  # Get more for each type
                    filters=merged_filters
                )
                semantic_results.extend(type_results)
            
            # Remove duplicates based on sku_id
            seen_ids = set()
            unique_semantic_results = []
            for result in semantic_results:
                sku_id = result.get("sku_id")
                if sku_id and sku_id not in seen_ids:
                    seen_ids.add(sku_id)
                    unique_semantic_results.append(result)
            
            semantic_results = unique_semantic_results
            print(f"[Hybrid Search] Pre-filter semantic results: {len(semantic_results)}")
        else:
            # Use regular semantic search without pre-filtering
            print(f"[Hybrid Search] Using regular semantic search (no type filter)")
            semantic_results = await self.vector_search.search_by_text_embedding(
                db, query_embedding, top_k * 2, merged_filters
            )
            print(f"[Hybrid Search] Regular semantic results: {len(semantic_results)}")
        
        # Keyword search (unchanged)
        keyword_results = await self.bm25_search.search_by_keywords(
            db, query_analysis["expanded_terms"], top_k * 2, merged_filters
        )
        print(f"[Hybrid Search] Keyword results: {len(keyword_results)}")
        
        # Combine and deduplicate
        combined_results = self._combine_results(semantic_results) #removing keyword results
        print(f"[Hybrid Search] Combined results: {len(combined_results)}")
        
        # Apply exclusion filters
        combined_results = self._apply_exclusion_filters(combined_results, ai_filters)
        print(f"[Hybrid Search] After exclusions: {len(combined_results)}")
        
        # Rerank if requested
        if rerank:
            combined_results = self.reranker.rerank_results(
                combined_results, query_analysis, mode="hybrid"
            )
        
        # Format response
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "query_understanding": query_analysis,
            "matches": combined_results,
            "search_strategy": "AI-powered hybrid search with pre-filtering and query parsing",
            "total_results": len(combined_results),
            "search_time_ms": round(search_time, 2),
            "is_collection_query": False
        }
        
        
    async def _search_collection(
        self,
        db: AsyncSession,
        refined_query: str,
        ai_parsed_query: Dict,
        filters: Dict,
        rerank: bool,
        start_time: datetime
    ) -> Dict:
        """Collection search flow - generates 10 complete looks"""
        print("[Hybrid Search] Starting collection search")
        
        ai_filters = ai_parsed_query.get("filters", {})
        item_types = ai_filters.get("type", [])
        existing_items = ai_parsed_query.get("existing_items", {})
        
        # ✅ CRITICAL FIX: If only 1 item type, route to NORMAL search
        if len(item_types) <= 1:
            logger.info(f"[Collection] Only {len(item_types)} item type(s) detected, routing to normal search")
            return await self._search_normal(
                db=db,
                refined_query=refined_query,
                ai_parsed_query={
                    **ai_parsed_query,
                    "is_collection_query": False  # Override to false
                },
                filters=filters,
                top_k=10,
                rerank=rerank,
                start_time=start_time
            )
        
        # Merge filters
        merged_filters = self._merge_filters(filters, ai_filters)
        
        # Generate 10 complete looks
        looks = await self._generate_looks(
            db=db,
            refined_query=refined_query,
            item_types=item_types,
            existing_items=existing_items,
            merged_filters=merged_filters,
            ai_filters=ai_filters,
            num_looks=10
        )
        
        # ✅ ADDITIONAL FIX: If no looks generated, fallback to normal search
        if len(looks) == 0:
            logger.warning("[Collection] No looks generated, falling back to normal search")
            return await self._search_normal(
                db=db,
                refined_query=refined_query,
                ai_parsed_query={
                    **ai_parsed_query,
                    "is_collection_query": False
                },
                filters=filters,
                top_k=10,
                rerank=rerank,
                start_time=start_time
            )
        
        # Format response
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "query_understanding": {
                "original_query": ai_parsed_query.get("original_query"),
                "refined_query": refined_query,
                "item_types": item_types,
                "existing_items": existing_items,
                "explanation": ai_parsed_query.get("explanation")
            },
            "looks": looks,
            "search_strategy": "AI-powered collection search",
            "total_looks": len(looks),
            "search_time_ms": round(search_time, 2),
            "is_collection_query": True
        }
        
    async def _generate_looks(
        self,
        db: AsyncSession,
        refined_query: str,
        item_types: List[str],
        existing_items: Dict,
        merged_filters: Dict,
        ai_filters: Dict,
        num_looks: int = 10
    ) -> List[Dict]:
        """Generate complete looks by fetching top 2 items per sub-type (5 sub-types per type = 10 items per type)"""
        print(f"[Collection] Generating {num_looks} looks for types: {item_types}")
        
        # Determine gender for sub-type selection
        gender = ai_filters.get("gender", "").lower()
        if "men" in gender or "male" in gender:
            gender_key = "men"
        elif "women" in gender or "female" in gender:
            gender_key = "women"
        else:
            # Default to men if no gender specified (or handle both)
            gender_key = "men"
        
        # Get sub-type mappings for the determined gender
        if gender_key not in self.query_parser.sub_type_mappings:
            logger.error(f"[Collection] Gender key '{gender_key}' not found in sub_type_mappings")
            return []
        
        gender_mappings = self.query_parser.sub_type_mappings[gender_key]
        
        # Build search tasks for each type
        search_tasks = []
        
        for item_type in item_types:
            if item_type not in gender_mappings:
                logger.warning(f"[Collection] Item type '{item_type}' not found for gender '{gender_key}'")
                continue
            
            sub_types_dict = gender_mappings[item_type]
            
            # For each sub-type, create search task
            for sub_type, specific_items in sub_types_dict.items():
                # Create enriched query for this sub-type using all 5 specific items
                sub_query = f"{refined_query} {sub_type} {' '.join(specific_items)}"
                
                # Generate embedding
                embedding = generate_text_embedding(sub_query)
                if len(embedding) < 768:
                    embedding = embedding + [0.0] * (768 - len(embedding))
                embedding = embedding[:768]
                
                # Create search task to fetch top 2 results for this sub-type
                task = self._search_sub_type(
                    db=db,
                    embedding=embedding,
                    item_type=item_type,
                    sub_type=sub_type,
                    merged_filters=merged_filters,
                    ai_filters=ai_filters,
                    top_k=2  # Top 2 per sub-type
                )
                search_tasks.append(task)
        
        # Execute all searches in parallel
        print(f"[Collection] Executing {len(search_tasks)} parallel searches")
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Organize results by type (each type will have 10 items: 2 per sub-type × 5 sub-types)
        items_by_type = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Sub-type search failed: {result}")
                continue
            
            item_type = result["item_type"]
            if item_type not in items_by_type:
                items_by_type[item_type] = []
            items_by_type[item_type].extend(result["items"])
        
        print(f"[Collection] Found items: {[(k, len(v)) for k, v in items_by_type.items()]}")
        
        # print(f"[Collection] items by type: {items_by_type}")
        
        # Generate looks by combining items
        looks = self._combine_items_into_looks(
            items_by_type=items_by_type,
            item_types=item_types,
            existing_items=existing_items,
            num_looks=num_looks
        )
        
        return looks

    async def _search_sub_type(
        self,
        db: AsyncSession,
        embedding: List[float],
        item_type: str,
        sub_type: str,
        merged_filters: Dict,
        ai_filters: Dict,
        top_k: int = 2
    ) -> Dict:
        """
        Search for items of a specific sub-type using TWO-STAGE filtering:
        1. Pre-filter by category and product_type
        2. Apply embedding search on filtered results
        """
        try:
            logger.debug(f"[Sub-type Search] Type: {item_type}, Sub-type: {sub_type}")
            
            # Use the NEW pre-filter search method
            results = await self.vector_search.search_by_text_embedding_with_prefilter(
                db=db,
                query_embedding=embedding,
                item_type=item_type,
                top_k=top_k * 2,  # Get more for exclusion filtering
                filters=merged_filters
            )
            
            # Apply exclusion filters
            results = self._apply_exclusion_filters(results, ai_filters)
            
            logger.debug(f"[Sub-type Search] Found {len(results)} results for {item_type}/{sub_type} after all filtering")
            
            # Take top_k
            results = results[:top_k]
            
            return {
                "item_type": item_type,
                "sub_type": sub_type,
                "items": results
            }
        except Exception as e:
            logger.error(f"Sub-type search failed for {item_type}/{sub_type}: {e}")
            return {
                "item_type": item_type,
                "sub_type": sub_type,
                "items": []
            }

    def _combine_items_into_looks(
        self,
        items_by_type: Dict[str, List[Dict]],
        item_types: List[str],
        existing_items: Dict,
        num_looks: int = 10
    ) -> List[Dict]:
        """Combine items from different types into complete looks"""
        looks = []
        
        # Get existing items info
        existing_types = existing_items.get("type", [])
        existing_brands = existing_items.get("brands", [])
        
        print(f"[Collection] Combining items into looks - existing_types: {existing_types}, existing_brands: {existing_brands}")
        print(f"[Collection] Items by type keys: {list(items_by_type.keys())}")
        
        # ✅ CRITICAL CHECK: If only 1 item type with items, don't create looks
        available_types = [t for t in item_types if items_by_type.get(t)]
        if len(available_types) <= 1:
            logger.warning(f"[Collection] Only {len(available_types)} type(s) have items - insufficient for looks")
            return []
        
        # For each look, pick one item from each type
        for look_idx in range(num_looks):
            look = {
                "look_id": look_idx + 1,
                "items": {}
            }
            
            for item_type in item_types:
                # Skip if this is an existing item type
                if item_type in existing_types:
                    # Use existing item info
                    look["items"][item_type] = {
                        "product_id": None,
                        "title": f"Your {item_type}",
                        "brand": ", ".join(existing_brands) if existing_brands else "Your Item",
                        "category": None,
                        "price": None,
                        "image_url": None,
                        "pdp_url": None,
                        "type": item_type,
                        "is_existing": True,
                        "brands": existing_brands
                    }
                    continue
                
                # Get items for this type
                type_items = items_by_type.get(item_type, [])
                if not type_items:
                    print(f"[Collection] No items found for type: {item_type}")
                    continue
                
                # Pick item for this look (cycle through available items)
                item_idx = look_idx % len(type_items)
                item = type_items[item_idx]
                
                # Convert Decimal to float for price
                price = item.get("lowest_price")
                if price is not None:
                    try:
                        price = float(price)
                    except (ValueError, TypeError):
                        price = None
                
                # Format the item properly for frontend
                look["items"][item_type] = {
                    "product_id": item.get("sku_id"),
                    "title": item.get("title"),
                    "brand": item.get("brand_name"),
                    "category": item.get("category"),
                    "price": price,
                    "image_url": item.get("featured_image"),
                    "pdp_url": item.get("pdp_url"),
                    "type": item_type,
                    "is_existing": False,
                    "brands": None
                }
            
            # Only add look if it has at least 2 items
            if len(look["items"]) >= 2:
                looks.append(look)
                print(f"[Collection] Created look {look_idx + 1} with {len(look['items'])} items")
            else:
                print(f"[Collection] Skipped look {look_idx + 1} - only {len(look['items'])} items (need at least 2)")
        
        print(f"[Collection] Generated {len(looks)} complete looks")
        
        return looks[:num_looks]

    def _build_enriched_query(self, refined_query: str, merged_filters: Dict, ai_filters: Dict) -> str:
        """
        Build enriched query text by combining refined query with INCLUSION filter values.
        Uses weighted query approach: 
        - refined query (3x weight)
        - filter values (3x weight each: brands, colors, gender, category)
        - corresponding item type names (1x weight)
        
        Excludes exclusion filters (exclude_brands, exclude_colors, exclude_gender).
        
        Args:
            refined_query: The refined query text
            merged_filters: Merged filters from user and AI (only used for price_range)
            ai_filters: AI-parsed filters (used for category, brands, colors, gender, type)
            
        Returns:
            Enriched query string for embedding generation with weighted terms
        """
        query_parts = []
        
        # Step 1: Add refined query 3 times for weighting
        query_parts.extend([refined_query] * 3)
        
        # Step 2: Add filter values 3 times each for weighting
        if ai_filters:
            # Add category filter values from AI (3x weight)
            if ai_filters.get("category"):
                categories = ai_filters["category"]
                if isinstance(categories, list):
                    category_str = " ".join(categories)
                    query_parts.extend([category_str] * 3)
                else:
                    query_parts.extend([str(categories)] * 3)
            
            # Add brand filter values from AI (3x weight, INCLUSION only)
            if ai_filters.get("brands"):
                brands = ai_filters["brands"]
                if isinstance(brands, list):
                    brands_str = " ".join(brands)
                    query_parts.extend([brands_str] * 3)
                else:
                    query_parts.extend([str(brands)] * 3)
            
            # Add color filter values from AI (3x weight, INCLUSION only)
            if ai_filters.get("colors"):
                colors = ai_filters["colors"]
                if isinstance(colors, list):
                    colors_str = " ".join(colors)
                    query_parts.extend([colors_str] * 3)
                else:
                    query_parts.extend([str(colors)] * 3)
            
            # Add gender filter value from AI (3x weight, INCLUSION only)
            if ai_filters.get("gender"):
                query_parts.extend([str(ai_filters["gender"])] * 3)
        
        # Step 3: Add corresponding item type names based on gender and type (1x weight)
        if ai_filters and ai_filters.get("type"):
            item_types = ai_filters["type"]
            gender = ai_filters.get("gender", "").lower()
            
            # Determine which gender mappings to use
            if "men" in gender or "male" in gender:
                gender_key = "men"
                type_specific_items = self._get_type_specific_items([gender_key], item_types)
                query_parts.extend(type_specific_items)
            elif "women" in gender or "female" in gender:
                gender_key = "women"
                type_specific_items = self._get_type_specific_items([gender_key], item_types)
                query_parts.extend(type_specific_items)
            else:
                # No gender specified - include both men and women items
                type_specific_items = self._get_type_specific_items(["men", "women"], item_types)
                query_parts.extend(type_specific_items)
        
        # Step 4: Add price range context from merged_filters (1x weight)
        if merged_filters and merged_filters.get("price_range"):
            price_range = merged_filters["price_range"]
            if len(price_range) == 2:
                min_price, max_price = price_range
                if min_price == 0:
                    query_parts.append(f"affordable under ${max_price}")
                elif max_price >= 500:
                    query_parts.append(f"premium over ${min_price}")
                else:
                    query_parts.append(f"mid-range ${min_price} to ${max_price}")
        
        # Join all parts with spaces
        enriched_query = " ".join(filter(None, query_parts))
        
        print(f"[Build Enriched Query] Final enriched query: {enriched_query}")
        
        return enriched_query
    
    def _get_type_specific_items(self, genders: List[str], item_types: List[str]) -> List[str]:
        """
        Get clothing item names for specific genders and types.
        
        Args:
            genders: List of genders ['men', 'women']
            item_types: List of item types ['top', 'bottom', 'hat', 'shoes', 'watch', 'bag']
            
        Returns:
            List of specific clothing item names
        """
        items = []
        
        for gender in genders:
            if gender not in self.clothing_items:
                continue
            
            for item_type in item_types:
                if item_type not in self.clothing_items[gender]:
                    continue
                
                # Get all items for this gender and type
                type_items = self.clothing_items[gender][item_type]
                items.extend(type_items)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        
        return unique_items
 
    def _merge_filters(self, user_filters: Dict, ai_filters: Dict) -> Dict:
        """Merge user-provided filters with AI-parsed filters, handling None values properly"""
        merged = {}
        
        # Helper function to add filter if valid
        def add_filter(key: str, value):
            if value is None:
                return
            if isinstance(value, list):
                # Filter out None and empty strings
                cleaned = [v for v in value if v is not None and v != ""]
                if cleaned:
                    merged[key] = cleaned
            elif isinstance(value, str) and value.strip():
                merged[key] = value.strip()
            elif key == "price_range" and isinstance(value, list) and len(value) == 2:
                if all(v is not None and isinstance(v, (int, float)) for v in value):
                    merged[key] = value
        
        # User filters take precedence
        if user_filters:
            for key, value in user_filters.items():
                add_filter(key, value)
        
        # Add AI filters - ONLY for price_range
        if ai_filters:
            # Only merge price_range from AI filters if not already present
            if "price_range" in ai_filters and "price_range" not in merged:
                add_filter("price_range", ai_filters["price_range"])
            
            # Comment out other filter merging - keep AI exclusion filters separate
            # if key not in merged:
            #     add_filter(key, value)
        
        return merged if merged else None

    def _apply_exclusion_filters(self, results: List[Dict], ai_filters: Dict) -> List[Dict]:
        """Apply exclusion filters from AI parsing with case-insensitive word matching"""
        if not ai_filters:
            return results
        
        filtered_results = []
        
        exclude_brands = ai_filters.get("exclude_brands", [])
        exclude_colors = ai_filters.get("exclude_colors", [])
        exclude_gender = ai_filters.get("exclude_gender", [])
        
        for result in results:
            should_exclude = False
            
            # Check gender exclusion with word boundary matching
            if exclude_gender:
                gender = result.get("gender")
                title = result.get("title")
                tags = result.get("tags")
                
                for excluded in exclude_gender:
                    if not excluded:
                        continue
                    
                    excluded_lower = excluded.lower()
                    
                    # Check in gender field
                    if gender and self._contains_word(gender.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded gender in gender field: {excluded}")
                        should_exclude = True
                        break
                    
                    # Check in title
                    if title and self._contains_word(title.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded gender in title: {excluded}")
                        should_exclude = True
                        break
                    
                    # Check in tags
                    if tags and self._contains_word(tags.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded gender in tags: {excluded}")
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
            
            # Check brand exclusion with word boundary matching
            if exclude_brands:
                brand = result.get("brand_name")
                title = result.get("title")
                tags = result.get("tags")
                
                for excluded in exclude_brands:
                    if not excluded:
                        continue
                    
                    excluded_lower = excluded.lower()
                    
                    # Check in brand_name
                    if brand and self._contains_word(brand.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded brand in brand_name: {excluded}")
                        should_exclude = True
                        break
                    
                    # Check in title
                    if title and self._contains_word(title.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded brand in title: {excluded}")
                        should_exclude = True
                        break
                    
                    # Check in tags
                    if tags and self._contains_word(tags.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded brand in tags: {excluded}")
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
            
            # Check color exclusion with word boundary matching
            if exclude_colors:
                colorways = result.get("colorways")
                title = result.get("title")
                tags = result.get("tags")
                
                for excluded in exclude_colors:
                    if not excluded:
                        continue
                    
                    excluded_lower = excluded.lower()
                    
                    # Check in colorways
                    if colorways and self._contains_word(colorways.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded color in colorways: {excluded}")
                        should_exclude = True
                        break
                    
                    # Check in title
                    if title and self._contains_word(title.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded color in title: {excluded}")
                        should_exclude = True
                        break
                    
                    # Check in tags
                    if tags and self._contains_word(tags.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered out {result.get('title')} - excluded color in tags: {excluded}")
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _contains_word(self, text: str, word: str) -> bool:
        """
        Check if a word exists in text as a whole word (case-insensitive).
        Handles word boundaries properly to avoid partial matches.
        
        Args:
            text: The text to search in (should be lowercase)
            word: The word to search for (should be lowercase)
            
        Returns:
            True if word exists as a complete word in text
        """
        import re
        # Use word boundary regex to match whole words only
        # \b ensures we match word boundaries
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text))

   
    async def search_by_image(
        self, 
        db: AsyncSession, 
        image_url: str, 
        top_k: int = 10,
        filters: Dict = None, 
        rerank: bool = True
    ) -> Dict:
        """Perform image-based search"""
        start_time = datetime.now()
        
        try:
            print(f"[Hybrid Search] Image search for: {image_url}")
            
            # Step 1: Download image and generate embedding
            image = download_image(image_url)
            query_embedding = generate_visual_embedding(image)
            
            if query_embedding is None or len(query_embedding) == 0:
                raise ValueError("Failed to generate image embedding")
            
            print(f"[Hybrid Search] Visual embedding dimension: {len(query_embedding)}")
            
            # Step 2: Perform visual search
            visual_results = await self.vector_search.search_by_visual_embedding(
                db, query_embedding, top_k * 2, filters
            )
            
            print(f"[Hybrid Search] Visual search results: {len(visual_results)}")
            
            # Step 3: Analyze image content
            image_analysis = self._analyze_image_content(visual_results[:5])
            
            # Step 4: Rerank if requested
            if rerank:
                visual_results = self.reranker.rerank_results(
                    visual_results, {}, mode="visual"
                )
            
            # Step 5: Format response
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "query_analysis": {
                    "detected_items": image_analysis["detected_items"],
                    "extracted_from_image": image_analysis
                },
                "matches": visual_results[:top_k],
                "total_results": len(visual_results),
                "search_time_ms": round(search_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "query_analysis": {"error": str(e)},
                "matches": [],
                "total_results": 0,
                "search_time_ms": 0
            }
    
    def _combine_results(
        self, 
        semantic_results: List[Dict], 
        keyword_results: List[Dict]
    ) -> List[Dict]:
        """Combine and deduplicate results from different search methods"""
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            product_id = result.get("sku_id")
            if product_id and product_id not in combined:
                combined[product_id] = {
                    **result,
                    "semantic_score": result.get("similarity_score", 0),
                    "keyword_score": 0
                }
        
        # Add keyword results
        for result in keyword_results:
            product_id = result.get("sku_id")
            if product_id:
                if product_id in combined:
                    combined[product_id]["keyword_score"] = result.get("keyword_score", 0)
                else:
                    combined[product_id] = {
                        **result,
                        "semantic_score": 0,
                        "keyword_score": result.get("keyword_score", 0)
                    }
        
        return list(combined.values())
    
    def _analyze_image_content(self, top_results: List[Dict]) -> Dict:
        """Basic image content analysis based on search results"""
        analysis = {
            "detected_items": [],
            "dominant_colors": [],
            "inferred_style": [],
            "detected_category": None
        }
        
        # Analyze top results to infer image content
        categories = [r.get("category") for r in top_results if r.get("category")]
        if categories:
            analysis["detected_category"] = max(set(categories), key=categories.count)
        
        # Extract brands as detected items
        brands = [r.get("brand_name") for r in top_results if r.get("brand_name")]
        analysis["detected_items"] = list(set(brands))[:3]
        
        # Infer style from product types
        product_types = [r.get("product_type") for r in top_results if r.get("product_type")]
        product_types_str = " ".join(product_types).lower()
        
        if "casual" in product_types_str:
            analysis["inferred_style"].append("casual")
        if "athletic" in product_types_str or "sport" in product_types_str:
            analysis["inferred_style"].append("athletic")
        if "formal" in product_types_str:
            analysis["inferred_style"].append("formal")
        
        return analysis
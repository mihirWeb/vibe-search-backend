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
import traceback

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
    
    async def search_by_visual_embedding_with_prefilter(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        item_type: str,
        top_k: int = 10,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Visual search with pre-filtering by category and product_type
        
        Args:
            db: Database session
            query_embedding: Visual embedding vector
            item_type: Type of item (top, bottom, shoes, hat, watch, bag)
            top_k: Number of results to return
            filters: Additional filters
        """
        try:
            # Get allowed categories and product types for this item type
            allowed_categories = self.CATEGORY_MAPPINGS.get(item_type, [])
            allowed_product_types = self.PRODUCT_TYPE_MAPPINGS.get(item_type, [])
            
            if not allowed_categories and not allowed_product_types:
                logger.warning(f"[Visual Pre-filter Search] No category/product_type mappings for: {item_type}")
                return []
            
            logger.debug(f"[Visual Pre-filter Search] Item type: {item_type}")
            logger.debug(f"[Visual Pre-filter Search] Allowed categories: {allowed_categories}")
            logger.debug(f"[Visual Pre-filter Search] Allowed product types: {allowed_product_types}")
            
            # Convert embedding to PostgreSQL vector string format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Build query with category and product_type pre-filtering
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
            params = {"top_k": top_k * 3}  # Get more results for filtering
            
            # Add category and product_type filters
            if allowed_categories:
                category_conditions = []
                for i, cat in enumerate(allowed_categories):
                    param_name = f"cat_{i}"
                    category_conditions.append(f"LOWER(category) LIKE LOWER(:{param_name})")
                    params[param_name] = f"%{cat}%"
                    bindparams_list.append(bindparam(param_name))
                
                category_filter = f"(category IS NOT NULL AND ({' OR '.join(category_conditions)}))"
                query_parts.append(f" AND {category_filter}")

            if allowed_product_types:
                product_type_conditions = []
                for i, ptype in enumerate(allowed_product_types):
                    param_name = f"ptype_{i}"
                    product_type_conditions.append(f"LOWER(product_type) LIKE LOWER(:{param_name})")
                    params[param_name] = f"%{ptype}%"
                    bindparams_list.append(bindparam(param_name))
                
                product_type_filter = f"(({' OR '.join(product_type_conditions)}) OR product_type IS NULL)"
                query_parts.append(f" AND {product_type_filter}")
            
            # Add additional filters
            if filters:
                if filters.get("price_range") and len(filters["price_range"]) == 2:
                    query_parts.append(" AND lowest_price BETWEEN :min_price AND :max_price")
                    params["min_price"] = filters["price_range"][0]
                    params["max_price"] = filters["price_range"][1]
                    bindparams_list.append(bindparam("min_price"))
                    bindparams_list.append(bindparam("max_price"))
            
            # Order by similarity and limit
            query_parts.append(" ORDER BY similarity_score DESC LIMIT :top_k")
            query_sql = "".join(query_parts)
            
            logger.debug(f"[Visual Pre-filter Search] Executing filtered search for {item_type}")
            
            stmt = text(query_sql).bindparams(*bindparams_list)
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
            results = [dict(row._mapping) for row in rows]
            
            logger.debug(f"[Visual Pre-filter Search] Found {len(results)} results for {item_type}")
            
            # Additional Python-level verification
            verified_results = []
            for result in results:
                result_category = (result.get("category") or "").lower()
                result_product_type = (result.get("product_type") or "").lower()
                
                category_match = any(
                    cat.lower() in result_category or result_category in cat.lower()
                    for cat in allowed_categories
                )
                
                product_type_match = any(
                    ptype.lower() in result_product_type or result_product_type in ptype.lower()
                    for ptype in allowed_product_types
                )
                
                if category_match or product_type_match:
                    verified_results.append(result)
            
            logger.debug(f"[Visual Pre-filter Search] After verification: {len(verified_results)} results")
            
            return verified_results[:top_k]
            
        except Exception as e:
            logger.error(f"Visual pre-filter search failed: {e}")
            logger.error(traceback.format_exc())
            return []

    
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
                # if filters.get("brands"):
                #     query_parts.append(" AND brand_name = ANY(:brands)")
                #     params["brands"] = filters["brands"]
                #     bindparams_list.append(bindparam("brands"))
                
                # if filters.get("gender"):
                #     query_parts.append(" AND gender = :gender")
                #     params["gender"] = filters["gender"]
                #     bindparams_list.append(bindparam("gender"))
                
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
                
                # if filters.get("brands"):
                #     query_parts.append(" AND brand_name = ANY(:brands)")
                #     params["brands"] = filters["brands"]
                #     bindparams_list.append(bindparam("brands"))
                
                # if filters.get("gender"):
                #     query_parts.append(" AND gender = :gender")
                #     params["gender"] = filters["gender"]
                #     bindparams_list.append(bindparam("gender"))
                
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
            logger.error(f"BM25 search without prefilter failed: {e}")
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
        
        # ✅ Check if type filter exists for pre-filtering
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
                    top_k=top_k * 2,
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
        
        # ✅ UPDATED: Keyword search with expanded query and pre-filtering
        keyword_results = await self.bm25_search.search_by_keywords(
            db=db,
            refined_query=refined_query,
            ai_filters=ai_filters,
            top_k=top_k * 2,
            filters=merged_filters
        )
        print(f"[Hybrid Search] Keyword results: {len(keyword_results)}")
        
        # Combine and deduplicate
        combined_results = self._combine_results(semantic_results, keyword_results)
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
                    "is_collection_query": False
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
        """Generate complete looks by fetching top 2 items per sub-type"""
        print(f"[Collection] Generating {num_looks} looks for types: {item_types}")
        
        # Determine gender for sub-type selection
        gender = ai_filters.get("gender", "").lower()
        if "men" in gender or "male" in gender:
            gender_key = "men"
        elif "women" in gender or "female" in gender:
            gender_key = "women"
        else:
            gender_key = "men"
        
        # Get sub-type mappings
        if gender_key not in self.query_parser.sub_type_mappings:
            logger.error(f"[Collection] Gender key '{gender_key}' not found")
            return []
        
        gender_mappings = self.query_parser.sub_type_mappings[gender_key]
        
        # Build search tasks
        search_tasks = []
        
        for item_type in item_types:
            if item_type not in gender_mappings:
                logger.warning(f"[Collection] Item type '{item_type}' not found")
                continue
            
            sub_types_dict = gender_mappings[item_type]
            
            for sub_type, specific_items in sub_types_dict.items():
                sub_query = f"{refined_query} {sub_type} {' '.join(specific_items)}"
                
                embedding = generate_text_embedding(sub_query)
                if len(embedding) < 768:
                    embedding = embedding + [0.0] * (768 - len(embedding))
                embedding = embedding[:768]
                
                task = self._search_sub_type(
                    db=db,
                    embedding=embedding,
                    item_type=item_type,
                    sub_type=sub_type,
                    merged_filters=merged_filters,
                    ai_filters=ai_filters,
                    top_k=2
                )
                search_tasks.append(task)
        
        print(f"[Collection] Executing {len(search_tasks)} parallel searches")
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Organize results by type
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
        
        # Generate looks
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
        """Search for items of a specific sub-type"""
        try:
            logger.debug(f"[Sub-type Search] Type: {item_type}, Sub-type: {sub_type}")
            
            results = await self.vector_search.search_by_text_embedding_with_prefilter(
                db=db,
                query_embedding=embedding,
                item_type=item_type,
                top_k=top_k * 2,
                filters=merged_filters
            )
            
            results = self._apply_exclusion_filters(results, ai_filters)
            
            logger.debug(f"[Sub-type Search] Found {len(results)} results for {item_type}/{sub_type}")
            
            return {
                "item_type": item_type,
                "sub_type": sub_type,
                "items": results[:top_k]
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
        
        existing_types = existing_items.get("type", [])
        existing_brands = existing_items.get("brands", [])
        
        print(f"[Collection] Combining items - existing_types: {existing_types}")
        
        available_types = [t for t in item_types if items_by_type.get(t)]
        if len(available_types) <= 1:
            logger.warning(f"[Collection] Only {len(available_types)} type(s) have items")
            return []
        
        for look_idx in range(num_looks):
            look = {
                "look_id": look_idx + 1,
                "items": {}
            }
            
            for item_type in item_types:
                if item_type in existing_types:
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
                
                type_items = items_by_type.get(item_type, [])
                if not type_items:
                    continue
                
                item_idx = look_idx % len(type_items)
                item = type_items[item_idx]
                
                price = item.get("lowest_price")
                if price is not None:
                    try:
                        price = float(price)
                    except (ValueError, TypeError):
                        price = None
                
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
            
            if len(look["items"]) >= 2:
                looks.append(look)
        
        print(f"[Collection] Generated {len(looks)} complete looks")
        
        return looks[:num_looks]

    def _build_enriched_query(self, refined_query: str, merged_filters: Dict, ai_filters: Dict) -> str:
        """Build enriched query with weighted terms: refined_query (3x), brands (5x), other filters (3x)"""
        query_parts = []
        
        # Add refined query 3 times
        query_parts.extend([refined_query] * 3)
        
        # Add filter values with weights
        if ai_filters:
            # Category (3x)
            if ai_filters.get("category"):
                categories = ai_filters["category"]
                if isinstance(categories, list):
                    category_str = " ".join(categories)
                    query_parts.extend([category_str] * 3)
                else:
                    query_parts.extend([str(categories)] * 3)
            
            # Brands (5x)
            if ai_filters.get("brands"):
                brands = ai_filters["brands"]
                if isinstance(brands, list):
                    brands_str = " ".join(brands)
                    query_parts.extend([brands_str] * 5)
                else:
                    query_parts.extend([str(brands)] * 5)
            
            # Colors (3x)
            if ai_filters.get("colors"):
                colors = ai_filters["colors"]
                if isinstance(colors, list):
                    colors_str = " ".join(colors)
                    query_parts.extend([colors_str] * 3)
                else:
                    query_parts.extend([str(colors)] * 3)
            
            # Gender (3x)
            if ai_filters.get("gender"):
                query_parts.extend([str(ai_filters["gender"])] * 3)
        
        # Add item type specific names (1x)
        if ai_filters and ai_filters.get("type"):
            item_types = ai_filters["type"]
            gender = ai_filters.get("gender", "").lower()
            
            if "men" in gender or "male" in gender:
                gender_key = "men"
                type_specific_items = self._get_type_specific_items([gender_key], item_types)
                query_parts.extend(type_specific_items)
            elif "women" in gender or "female" in gender:
                gender_key = "women"
                type_specific_items = self._get_type_specific_items([gender_key], item_types)
                query_parts.extend(type_specific_items)
            else:
                type_specific_items = self._get_type_specific_items(["men", "women"], item_types)
                query_parts.extend(type_specific_items)
        
        # Add price context (1x)
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
        
        enriched_query = " ".join(filter(None, query_parts))
        
        print(f"[Build Enriched Query] Final: {enriched_query}")
        
        return enriched_query
    
    def _get_type_specific_items(self, genders: List[str], item_types: List[str]) -> List[str]:
        """Get clothing item names for specific genders and types"""
        items = []
        
        for gender in genders:
            if gender not in self.clothing_items:
                continue
            
            for item_type in item_types:
                if item_type not in self.clothing_items[gender]:
                    continue
                
                type_items = self.clothing_items[gender][item_type]
                items.extend(type_items)
        
        # Remove duplicates
        seen = set()
        unique_items = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        
        return unique_items
 
    def _merge_filters(self, user_filters: Dict, ai_filters: Dict) -> Dict:
        """Merge user-provided filters with AI-parsed filters"""
        merged = {}
        
        def add_filter(key: str, value):
            if value is None:
                return
            if isinstance(value, list):
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
        
        # Add AI filters - only price_range if not already present
        if ai_filters:
            if "price_range" in ai_filters and "price_range" not in merged:
                add_filter("price_range", ai_filters["price_range"])
        
        return merged if merged else None

    def _apply_exclusion_filters(self, results: List[Dict], ai_filters: Dict) -> List[Dict]:
        """Apply exclusion filters with case-insensitive word matching"""
        if not ai_filters:
            return results
        
        filtered_results = []
        
        exclude_brands = ai_filters.get("exclude_brands", [])
        exclude_colors = ai_filters.get("exclude_colors", [])
        exclude_gender = ai_filters.get("exclude_gender", [])
        
        for result in results:
            should_exclude = False
            
            # Check gender exclusion
            if exclude_gender:
                gender = result.get("gender")
                title = result.get("title")
                tags = result.get("tags")
                
                for excluded in exclude_gender:
                    if not excluded:
                        continue
                    
                    excluded_lower = excluded.lower()
                    
                    if gender and self._contains_word(gender.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - gender: {excluded}")
                        should_exclude = True
                        break
                    
                    if title and self._contains_word(title.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - title gender: {excluded}")
                        should_exclude = True
                        break
                    
                    if tags and self._contains_word(tags.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - tags gender: {excluded}")
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
            
            # Check brand exclusion
            if exclude_brands:
                brand = result.get("brand_name")
                title = result.get("title")
                tags = result.get("tags")
                
                for excluded in exclude_brands:
                    if not excluded:
                        continue
                    
                    excluded_lower = excluded.lower()
                    
                    if brand and self._contains_word(brand.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - brand: {excluded}")
                        should_exclude = True
                        break
                    
                    if title and self._contains_word(title.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - title brand: {excluded}")
                        should_exclude = True
                        break
                    
                    if tags and self._contains_word(tags.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - tags brand: {excluded}")
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
            
            # Check color exclusion
            if exclude_colors:
                colorways = result.get("colorways")
                title = result.get("title")
                tags = result.get("tags")
                
                for excluded in exclude_colors:
                    if not excluded:
                        continue
                    
                    excluded_lower = excluded.lower()
                    
                    if colorways and self._contains_word(colorways.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - color: {excluded}")
                        should_exclude = True
                        break
                    
                    if title and self._contains_word(title.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - title color: {excluded}")
                        should_exclude = True
                        break
                    
                    if tags and self._contains_word(tags.lower(), excluded_lower):
                        logger.debug(f"[Exclusion] Filtered: {result.get('title')} - tags color: {excluded}")
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _contains_word(self, text: str, word: str) -> bool:
        """Check if word exists as a whole word in text (case-insensitive)"""
        import re
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text))

   
    async def search_by_image(
        self, 
        db: AsyncSession, 
        image_url: str,
        query: Optional[str] = None,
        top_k: int = 10,
        filters: Dict = None, 
        rerank: bool = True,
        use_ai_parser: bool = True
    ) -> Dict:
        """
        Perform image-based search with optional text query
        Follows same process as search_by_text with AI parsing and collection support
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"[Image Search] Starting image search with query: {query}")
            
            # Download image and generate visual embedding
            image = download_image(image_url)
            visual_embedding = generate_visual_embedding(image)
            
            if visual_embedding is None or len(visual_embedding) == 0:
                raise ValueError("Failed to generate visual embedding from image")
            
            logger.info(f"[Image Search] Visual embedding dimension: {len(visual_embedding)}")
            
            # Step 1: AI-powered query parsing (if query provided)
            ai_parsed_query = None
            refined_query = query or ""
            ai_filters = {}
            is_collection_query = False
            existing_items = {}
            
            if query and use_ai_parser:
                try:
                    logger.info(f"[Image Search] Parsing query with AI: {query}")
                    ai_parsed_query = self.query_parser.parse_query(query)
                    refined_query = ai_parsed_query.get("refined_query", query)
                    ai_filters = ai_parsed_query.get("filters", {})
                    is_collection_query = ai_parsed_query.get("is_collection_query", False)
                    existing_items = ai_parsed_query.get("existing_items", {})
                    
                    logger.info(f"[Image Search] AI parsed - is_collection: {is_collection_query}, types: {ai_filters.get('type', [])}")
                except Exception as e:
                    logger.warning(f"[Image Search] AI parsing failed, using original query: {e}")
                    ai_parsed_query = None
            
            # Step 2: Route to collection or normal image search
            if is_collection_query:
                return await self._search_image_collection(
                    db=db,
                    visual_embedding=visual_embedding,
                    refined_query=refined_query,
                    ai_parsed_query=ai_parsed_query,
                    filters=filters,
                    rerank=rerank,
                    start_time=start_time
                )
            else:
                return await self._search_image_normal(
                    db=db,
                    visual_embedding=visual_embedding,
                    refined_query=refined_query,
                    ai_parsed_query=ai_parsed_query,
                    filters=filters,
                    top_k=top_k,
                    rerank=rerank,
                    start_time=start_time
                )
            
        except Exception as e:
            logger.error(f"[Image Search] Search failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "query_analysis": {"error": str(e)},
                "matches": [],
                "total_results": 0,
                "search_time_ms": 0,
                "is_collection_query": False
            }
    
    async def _search_image_normal(
        self,
        db: AsyncSession,
        visual_embedding: List[float],
        refined_query: str,
        ai_parsed_query: Dict,
        filters: Dict,
        top_k: int,
        rerank: bool,
        start_time: datetime
    ) -> Dict:
        """Normal image search with pre-filtering"""
        ai_filters = ai_parsed_query.get("filters", {}) if ai_parsed_query else {}
        
        # Merge filters
        merged_filters = self._merge_filters(filters, ai_filters)
        logger.info(f"[Image Search Normal] Merged filters: {merged_filters}")
        
        # Analyze image content for query understanding
        image_analysis = {
            "search_method": "visual_embedding",
            "has_text_query": bool(refined_query),
            "text_query": refined_query if refined_query else None
        }
        
        # Add AI parsing info
        if ai_parsed_query:
            image_analysis["ai_parsing"] = {
                "original_query": ai_parsed_query.get("original_query"),
                "refined_query": refined_query,
                "ai_filters": ai_filters,
                "explanation": ai_parsed_query.get("explanation")
            }
        
        # Check if type filter exists for pre-filtering
        item_types = ai_filters.get("type", [])
        
        if item_types:
            # Use pre-filter search for each item type
            logger.info(f"[Image Search Normal] Using pre-filter search for types: {item_types}")
            
            visual_results = []
            for item_type in item_types:
                type_results = await self.vector_search.search_by_visual_embedding_with_prefilter(
                    db=db,
                    query_embedding=visual_embedding,
                    item_type=item_type,
                    top_k=top_k * 2,
                    filters=merged_filters
                )
                visual_results.extend(type_results)
            
            # Remove duplicates based on sku_id
            seen_ids = set()
            unique_visual_results = []
            for result in visual_results:
                sku_id = result.get("sku_id")
                if sku_id and sku_id not in seen_ids:
                    seen_ids.add(sku_id)
                    unique_visual_results.append(result)
            
            visual_results = unique_visual_results
            logger.info(f"[Image Search Normal] Pre-filter visual results: {len(visual_results)}")
        else:
            # Use regular visual search without pre-filtering
            logger.info(f"[Image Search Normal] Using regular visual search (no type filter)")
            visual_results = await self.vector_search.search_by_visual_embedding(
                db, visual_embedding, top_k * 2, merged_filters
            )
            logger.info(f"[Image Search Normal] Regular visual results: {len(visual_results)}")
        
        # Apply exclusion filters
        visual_results = self._apply_exclusion_filters(visual_results, ai_filters)
        logger.info(f"[Image Search Normal] After exclusions: {len(visual_results)}")
        
        # Rerank if requested
        if rerank and refined_query:
            query_info = {"original_query": refined_query, "extracted_keywords": []}
            visual_results = self.reranker.rerank_results(
                visual_results, query_info, mode="visual"
            )
        
        # Add detected items from top results
        if len(visual_results) > 0:
            detected_analysis = self._analyze_image_content(visual_results[:5])
            image_analysis.update(detected_analysis)
        
        # Format response
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "query_analysis": image_analysis,
            "matches": visual_results[:top_k],
            "total_results": len(visual_results),
            "search_time_ms": round(search_time, 2),
            "is_collection_query": False
        }
    
    async def _search_image_collection(
        self,
        db: AsyncSession,
        visual_embedding: List[float],
        refined_query: str,
        ai_parsed_query: Dict,
        filters: Dict,
        rerank: bool,
        start_time: datetime
    ) -> Dict:
        """Collection search flow for image search"""
        logger.info("[Image Search Collection] Starting collection search")
        
        ai_filters = ai_parsed_query.get("filters", {})
        item_types = ai_filters.get("type", [])
        existing_items = ai_parsed_query.get("existing_items", {})
        
        # If only 1 item type, route to normal search
        if len(item_types) <= 1:
            logger.info(f"[Image Search Collection] Only {len(item_types)} item type(s) detected, routing to normal search")
            return await self._search_image_normal(
                db=db,
                visual_embedding=visual_embedding,
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
        
        # Merge filters
        merged_filters = self._merge_filters(filters, ai_filters)
        
        # Generate 10 complete looks using visual embedding
        looks = await self._generate_visual_looks(
            db=db,
            visual_embedding=visual_embedding,
            refined_query=refined_query,
            item_types=item_types,
            existing_items=existing_items,
            merged_filters=merged_filters,
            ai_filters=ai_filters,
            num_looks=10
        )
        
        # If no looks generated, fallback to normal search
        if len(looks) == 0:
            logger.warning("[Image Search Collection] No looks generated, falling back to normal search")
            return await self._search_image_normal(
                db=db,
                visual_embedding=visual_embedding,
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
            "query_analysis": {
                "search_method": "visual_embedding_collection",
                "original_query": ai_parsed_query.get("original_query"),
                "refined_query": refined_query,
                "item_types": item_types,
                "existing_items": existing_items,
                "explanation": ai_parsed_query.get("explanation")
            },
            "looks": looks,
            "total_looks": len(looks),
            "search_time_ms": round(search_time, 2),
            "is_collection_query": True,
            "matches": [],
            "total_results": 0
        }
    
    async def _generate_visual_looks(
        self,
        db: AsyncSession,
        visual_embedding: List[float],
        refined_query: str,
        item_types: List[str],
        existing_items: Dict,
        merged_filters: Dict,
        ai_filters: Dict,
        num_looks: int = 10
    ) -> List[Dict]:
        """Generate complete looks using visual embedding"""
        logger.info(f"[Visual Collection] Generating {num_looks} looks for types: {item_types}")
        
        # Build search tasks for each item type
        search_tasks = []
        
        for item_type in item_types:
            task = self._search_visual_sub_type(
                db=db,
                visual_embedding=visual_embedding,
                item_type=item_type,
                merged_filters=merged_filters,
                ai_filters=ai_filters,
                top_k=10  # Get more items per type
            )
            search_tasks.append(task)
        
        logger.info(f"[Visual Collection] Executing {len(search_tasks)} parallel searches")
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Organize results by type
        items_by_type = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Visual sub-type search failed: {result}")
                continue
            
            item_type = result["item_type"]
            if item_type not in items_by_type:
                items_by_type[item_type] = []
            items_by_type[item_type].extend(result["items"])
        
        logger.info(f"[Visual Collection] Found items: {[(k, len(v)) for k, v in items_by_type.items()]}")
        
        # Generate looks
        looks = self._combine_items_into_looks(
            items_by_type=items_by_type,
            item_types=item_types,
            existing_items=existing_items,
            num_looks=num_looks
        )
        
        return looks
    
    async def _search_visual_sub_type(
        self,
        db: AsyncSession,
        visual_embedding: List[float],
        item_type: str,
        merged_filters: Dict,
        ai_filters: Dict,
        top_k: int = 10
    ) -> Dict:
        """Search for items of a specific type using visual embedding"""
        try:
            logger.debug(f"[Visual Sub-type Search] Type: {item_type}")
            
            results = await self.vector_search.search_by_visual_embedding_with_prefilter(
                db=db,
                query_embedding=visual_embedding,
                item_type=item_type,
                top_k=top_k * 2,
                filters=merged_filters
            )
            
            results = self._apply_exclusion_filters(results, ai_filters)
            
            logger.debug(f"[Visual Sub-type Search] Found {len(results)} results for {item_type}")
            
            return {
                "item_type": item_type,
                "items": results[:top_k]
            }
        except Exception as e:
            logger.error(f"Visual sub-type search failed for {item_type}: {e}")
            return {
                "item_type": item_type,
                "items": []
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
    """Handles BM25 keyword-based search with query expansion"""
    
    # Category and product type mappings (same as VectorSearchService)
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
    
    async def search_by_keywords(
        self, 
        db: AsyncSession, 
        refined_query: str,
        ai_filters: Dict,
        top_k: int = 10, 
        filters: Dict = None
    ) -> List[Dict]:
        """
        Search using BM25-style keyword matching with query expansion and pre-filtering.
        Expands query with: brand (5x), refined_query (3x), other filters/context (3x).
        Uses same category/product_type filtering as search_by_text_embedding_with_prefilter.
        """
        try:
            # Step 1: Build expanded query terms
            expanded_terms = []
            
            # Add refined query 3 times
            expanded_terms.extend([refined_query] * 3)
            
            # Add brand 5 times (if available)
            if ai_filters and ai_filters.get("brands"):
                brands = ai_filters["brands"]
                if isinstance(brands, list):
                    for brand in brands:
                        expanded_terms.extend([brand] * 5)
                else:
                    expanded_terms.extend([str(brands)] * 5)
            
            # Add other filters 3 times each
            if ai_filters:
                # Category
                if ai_filters.get("category"):
                    categories = ai_filters["category"]
                    if isinstance(categories, list):
                        for cat in categories:
                            expanded_terms.extend([cat] * 3)
                    else:
                        expanded_terms.extend([str(categories)] * 3)
                
                # Colors
                if ai_filters.get("colors"):
                    colors = ai_filters["colors"]
                    if isinstance(colors, list):
                        for color in colors:
                            expanded_terms.extend([color] * 3)
                    else:
                        expanded_terms.extend([str(colors)] * 3)
                
                # Gender
                if ai_filters.get("gender"):
                    expanded_terms.extend([str(ai_filters["gender"])] * 3)
            
            # Remove empty terms and limit to reasonable number
            expanded_terms = [term for term in expanded_terms if term and term.strip()]
            expanded_terms = expanded_terms[:20]  # Limit to 20 terms max
            
            logger.debug(f"[BM25 Search] Expanded terms: {expanded_terms}")
            
            if not expanded_terms:
                return []
            
            # Step 2: Get item types for pre-filtering
            item_types = ai_filters.get("type", []) if ai_filters else []
            
            # Step 3: Build search query with pre-filtering
            if item_types:
                # Use pre-filtering similar to search_by_text_embedding_with_prefilter
                results = []
                for item_type in item_types:
                    type_results = await self._search_with_prefilter(
                        db=db,
                        expanded_terms=expanded_terms,
                        item_type=item_type,
                        top_k=top_k * 2,
                        filters=filters
                    )
                    results.extend(type_results)
                
                # Remove duplicates
                seen_ids = set()
                unique_results = []
                for result in results:
                    sku_id = result.get("sku_id")
                    if sku_id and sku_id not in seen_ids:
                        seen_ids.add(sku_id)
                        unique_results.append(result)
                
                return unique_results[:top_k]
            else:
                # No item type filtering - use regular keyword search
                return await self._search_without_prefilter(
                    db=db,
                    expanded_terms=expanded_terms,
                    top_k=top_k,
                    filters=filters
                )
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _search_with_prefilter(
        self,
        db: AsyncSession,
        expanded_terms: List[str],
        item_type: str,
        top_k: int,
        filters: Dict = None
    ) -> List[Dict]:
        """Search with category and product_type pre-filtering"""
        try:
            # Get allowed categories and product types
            allowed_categories = self.CATEGORY_MAPPINGS.get(item_type, [])
            allowed_product_types = self.PRODUCT_TYPE_MAPPINGS.get(item_type, [])
            
            if not allowed_categories and not allowed_product_types:
                logger.warning(f"[BM25 Pre-filter] No mappings for: {item_type}")
                return []
            
            # Build keyword search conditions
            search_conditions = []
            params = {"top_k": top_k}
            bindparams_list = [bindparam("top_k")]
            
            for i, keyword in enumerate(expanded_terms):
                keyword_param = f"kw_{i}"
                search_conditions.extend([
                    f"title ILIKE :{keyword_param}",
                    f"description ILIKE :{keyword_param}",
                    f"tags ILIKE :{keyword_param}",
                    f"brand_name ILIKE :{keyword_param}"
                ])
                params[keyword_param] = f"%{keyword}%"
                bindparams_list.append(bindparam(keyword_param))
            
            # Build query with pre-filtering
            query_parts = [f"""
                SELECT 
                    id, sku_id, title, description, category, sub_category,
                    brand_name, product_type, gender, colorways, lowest_price,
                    featured_image, pdp_url, wishlist_num, tags,
                    CASE WHEN {' OR '.join(search_conditions)} THEN 1.0 ELSE 0.0 END as keyword_score
                FROM store_items
                WHERE ({' OR '.join(search_conditions)})
            """]
            
            # Add category and product_type filters (SAME AS search_by_text_embedding_with_prefilter)
            if allowed_categories:
                category_conditions = []
                for i, cat in enumerate(allowed_categories):
                    cat_param = f"cat_{i}"
                    category_conditions.append(f"category ILIKE :{cat_param}")
                    params[cat_param] = f"%{cat}%"
                    bindparams_list.append(bindparam(cat_param))
                
                category_filter = f"(category IS NOT NULL AND ({' OR '.join(category_conditions)}))"
                query_parts.append(f" AND {category_filter}")

            if allowed_product_types:
                product_type_conditions = []
                for i, ptype in enumerate(allowed_product_types):
                    ptype_param = f"ptype_{i}"
                    product_type_conditions.append(f"product_type ILIKE :{ptype_param}")
                    params[ptype_param] = f"%{ptype}%"
                    bindparams_list.append(bindparam(ptype_param))
                
                product_type_filter = f"(({' OR '.join(product_type_conditions)}) OR product_type IS NULL)"
                query_parts.append(f" AND {product_type_filter}")
            
            # Add additional filters
            if filters:
                # if filters.get("brands"):
                #     query_parts.append(" AND brand_name = ANY(:brands)")
                #     params["brands"] = filters["brands"]
                #     bindparams_list.append(bindparam("brands"))
                
                # if filters.get("gender"):
                #     query_parts.append(" AND gender = :gender")
                #     params["gender"] = filters["gender"]
                #     bindparams_list.append(bindparam("gender"))
                
                if filters.get("price_range") and len(filters["price_range"]) == 2:
                    query_parts.append(" AND lowest_price BETWEEN :min_price AND :max_price")
                    params["min_price"] = filters["price_range"][0]
                    params["max_price"] = filters["price_range"][1]
                    bindparams_list.extend([bindparam("min_price"), bindparam("max_price")])
            
            query_parts.append(" ORDER BY keyword_score DESC, wishlist_num DESC LIMIT :top_k")
            query_sql = "".join(query_parts)
            
            stmt = text(query_sql).bindparams(*bindparams_list)
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
            results = [dict(row._mapping) for row in rows]
            
            # Python-level verification (same as vector search)
            verified_results = []
            for result in results:
                result_category = (result.get("category") or "").lower()
                result_product_type = (result.get("product_type") or "").lower()
                
                category_match = any(
                    cat.lower() in result_category or result_category in cat.lower()
                    for cat in allowed_categories
                )
                
                product_type_match = any(
                    ptype.lower() in result_product_type or result_product_type in ptype.lower()
                    for ptype in allowed_product_types
                )
                
                if category_match or product_type_match:
                    verified_results.append(result)
            
            logger.debug(f"[BM25 Pre-filter] {item_type}: {len(verified_results)} verified results")
            
            return verified_results
            
        except Exception as e:
            logger.error(f"BM25 pre-filter search failed: {e}")
            return []
    
    async def _search_without_prefilter(
        self,
        db: AsyncSession,
        expanded_terms: List[str],
        top_k: int,
        filters: Dict = None
    ) -> List[Dict]:
        """Regular keyword search without item type pre-filtering"""
        try:
            search_conditions = []
            params = {"top_k": top_k}
            bindparams_list = [bindparam("top_k")]
            
            for i, keyword in enumerate(expanded_terms):
                keyword_param = f"kw_{i}"
                search_conditions.extend([
                    f"title ILIKE :{keyword_param}",
                    f"description ILIKE :{keyword_param}",
                    f"tags ILIKE :{keyword_param}",
                    f"brand_name ILIKE :{keyword_param}"
                ])
                params[keyword_param] = f"%{keyword}%"
                bindparams_list.append(bindparam(keyword_param))
            
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
            
            query_parts.append(" ORDER BY keyword_score DESC, wishlist_num DESC LIMIT :top_k")
            query_sql = "".join(query_parts)
            
            stmt = text(query_sql).bindparams(*bindparams_list)
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
            return [dict(row._mapping) for row in rows]

        except Exception as e:
            logger.error(f"Regular search failed: {e}")
            return []
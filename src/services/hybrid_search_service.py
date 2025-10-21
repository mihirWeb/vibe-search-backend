"""
Hybrid search service for store items.
Combines vector similarity search with keyword-based search and intelligent reranking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sqlalchemy import text, bindparam
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import ARRAY
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
            # Since we're using string substitution for the vector, don't bind it as a parameter
            query_parts = [f"""
                SELECT 
                    id, sku_id, title, description, category, sub_category,
                    brand_name, product_type, gender, colorways, lowest_price,
                    featured_image, pdp_url, wishlist_num, tags,
                    1 - (textual_embedding <=> '{embedding_str}'::vector) as similarity_score
                FROM store_items
                WHERE textual_embedding IS NOT NULL
            """]
            
            # Prepare bindparams list (without embedding since it's in the query string)
            bindparams_list = [bindparam("top_k")]
            
            # Use dictionary for parameters (without embedding)
            params = {"top_k": top_k}
            
            # Add filters with named parameters
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
            
            query_parts.append(" ORDER BY similarity_score DESC LIMIT :top_k")
            
            query_sql = "".join(query_parts)
            
            # Create text() with bindparams
            stmt = text(query_sql).bindparams(*bindparams_list)
            
            # Execute with dictionary parameters
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
            # Convert rows to dictionaries
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
            # Since we're using string substitution for the vector, don't bind it as a parameter
            query_parts = [f"""
                SELECT 
                    id, sku_id, title, description, category, sub_category,
                    brand_name, product_type, gender, colorways, lowest_price,
                    featured_image, pdp_url, wishlist_num, tags,
                    1 - (visual_embedding <=> '{embedding_str}'::vector) as similarity_score
                FROM store_items
                WHERE visual_embedding IS NOT NULL
            """]
            
            # Prepare bindparams list (without embedding since it's in the query string)
            bindparams_list = [bindparam("top_k")]
            
            # Use dictionary for parameters (without embedding)
            params = {"top_k": top_k}
            
            # Add filters with named parameters
            if filters:
                if filters.get("category"):
                    query_parts.append(" AND category = ANY(:categories)")
                    params["categories"] = filters["category"]
                    bindparams_list.append(bindparam("categories"))
                
                if filters.get("brands"):
                    query_parts.append(" AND brand_name = ANY(:brands)")
                    params["brands"] = filters["brands"]
                    bindparams_list.append(bindparam("brands"))
                
                if filters.get("colors"):
                    # For multiple colors, use OR conditions
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
            
            # Create text() with bindparams
            stmt = text(query_sql).bindparams(*bindparams_list)
            
            # Execute with dictionary parameters
            result = await db.execute(stmt, params)
            rows = result.fetchall()
            
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
        logger.info("[Hybrid Search Service] Initialized")
    
    async def search_by_text(
        self, 
        db: AsyncSession, 
        query: str, 
        top_k: int = 10, 
        filters: Dict = None, 
        rerank: bool = True
    ) -> Dict:
        """Perform text-based hybrid search with query expansion and context inference"""
        start_time = datetime.now()
        
        try:
            logger.info(f"[Hybrid Search] Text search for: {query}")
            
            # Step 1: Query understanding and expansion
            query_analysis = self.query_expander.expand_query(query)
            
            logger.info(f"[Hybrid Search] Query analysis: {query_analysis}")
            
            # Step 2: Generate query embedding
            query_embedding = generate_text_embedding(query)
            
            # Pad to 768 dimensions if needed
            if len(query_embedding) < 768:
                query_embedding = query_embedding + [0.0] * (768 - len(query_embedding))
            query_embedding = query_embedding[:768]
            
            # logger.info(f"[Hybrid Search] Embedding dimension: {len(query_embedding)}")
            
            # Step 3: Perform parallel searches
            semantic_task = self.vector_search.search_by_text_embedding(
                db, query_embedding, top_k * 2, filters
            )
            
            keyword_task = self.bm25_search.search_by_keywords(
                db, query_analysis["expanded_terms"], top_k * 2, filters
            )
            
            semantic_results, keyword_results = await asyncio.gather(
                semantic_task, keyword_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(semantic_results, Exception):
                logger.error(f"Semantic search failed: {semantic_results}")
                semantic_results = []
            else:
                logger.info(f"[Hybrid Search] Semantic results: {len(semantic_results)}")
            
            if isinstance(keyword_results, Exception):
                logger.error(f"Keyword search failed: {keyword_results}")
                keyword_results = []
            else:
                logger.info(f"[Hybrid Search] Keyword results: {len(keyword_results)}")
            
            # Step 4: Combine and deduplicate results
            combined_results = self._combine_results(semantic_results, keyword_results)
            logger.info(f"[Hybrid Search] Combined results: {len(combined_results)}")
            
            # Step 5: Rerank if requested with query context
            if rerank:
                combined_results = self.reranker.rerank_results(
                    combined_results, query_analysis, mode="hybrid"
                )
            
            # Step 6: Format response
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "query_understanding": query_analysis,
                "matches": combined_results[:top_k],
                "search_strategy": "Hybrid: BM25 keyword search + semantic embeddings with query expansion",
                "total_results": len(combined_results),
                "search_time_ms": round(search_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "query_understanding": {"original_query": query, "error": str(e)},
                "matches": [],
                "search_strategy": "failed",
                "total_results": 0,
                "search_time_ms": 0
            }
    
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
            logger.info(f"[Hybrid Search] Image search for: {image_url}")
            
            # Step 1: Download image and generate embedding
            image = download_image(image_url)
            query_embedding = generate_visual_embedding(image)
            
            if query_embedding is None or len(query_embedding) == 0:
                raise ValueError("Failed to generate image embedding")
            
            logger.info(f"[Hybrid Search] Visual embedding dimension: {len(query_embedding)}")
            
            # Step 2: Perform visual search
            visual_results = await self.vector_search.search_by_visual_embedding(
                db, query_embedding, top_k * 2, filters
            )
            
            logger.info(f"[Hybrid Search] Visual search results: {len(visual_results)}")
            
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
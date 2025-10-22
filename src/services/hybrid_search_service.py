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
            print(f"[Hybrid Search] Text search for: {query}")
            
            # Step 1: AI-powered query parsing (if enabled)
            ai_parsed_query = None
            refined_query = query
            ai_filters = {}
            
            if use_ai_parser:
                try:
                    ai_parsed_query = self.query_parser.parse_query(query)
                    refined_query = ai_parsed_query.get("refined_query", query)
                    ai_filters = ai_parsed_query.get("filters", {})
                    print(f"[Hybrid Search] AI parsed query: {ai_parsed_query}")
                except Exception as e:
                    logger.error(f"[Hybrid Search] AI parsing failed, using original query: {e}")
                    refined_query = query
            
            # Step 2: Merge AI filters with provided filters
            merged_filters = self._merge_filters(filters, ai_filters)
            print(f"[Hybrid Search] Merged filters: {merged_filters}")
            
            # Step 3: Query understanding and expansion on refined query
            query_analysis = self.query_expander.expand_query(refined_query)
            
            # Add AI parsing info to query analysis
            if ai_parsed_query:
                query_analysis["ai_parsing"] = {
                    "original_query": ai_parsed_query.get("original_query"),
                    "refined_query": refined_query,
                    "ai_filters": ai_filters,
                    "explanation": ai_parsed_query.get("explanation")
                }
            
            print(f"[Hybrid Search] Query analysis: {query_analysis}")
            
            # Step 4: Build enriched query text with filter values for embedding generation
            enriched_query_text = self._build_enriched_query(refined_query, merged_filters, ai_filters)
            print(f"[Hybrid Search] Enriched query for embedding: {enriched_query_text}")
            
            # Step 5: Generate query embedding from enriched query text
            query_embedding = generate_text_embedding(enriched_query_text)
            
            # Pad to 768 dimensions if needed
            if len(query_embedding) < 768:
                query_embedding = query_embedding + [0.0] * (768 - len(query_embedding))
            query_embedding = query_embedding[:768]
            
            # Step 6: Perform parallel searches with merged filters
            semantic_task = self.vector_search.search_by_text_embedding(
                db, query_embedding, top_k * 2, merged_filters
            )
            
            keyword_task = self.bm25_search.search_by_keywords(
                db, query_analysis["expanded_terms"], top_k * 2, merged_filters
            )
            
            semantic_results, keyword_results = await asyncio.gather(
                semantic_task, keyword_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(semantic_results, Exception):
                logger.error(f"Semantic search failed: {semantic_results}")
                semantic_results = []
            else:
                print(f"[Hybrid Search] Semantic results: {len(semantic_results)}")
            
            if isinstance(keyword_results, Exception):
                logger.error(f"Keyword search failed: {keyword_results}")
                keyword_results = []
            else:
                print(f"[Hybrid Search] Keyword results: {len(keyword_results)}")
            
            # Step 7: Combine and deduplicate results
            combined_results = self._combine_results(semantic_results, keyword_results)
            print(f"[Hybrid Search] Combined results: {len(combined_results)}")
            
            # Step 8: Apply exclusion filters
            combined_results = self._apply_exclusion_filters(combined_results, ai_filters)
            print(f"[Hybrid Search] After exclusions: {len(combined_results)}")
            
            # Step 9: Rerank if requested
            if rerank:
                combined_results = self.reranker.rerank_results(
                    combined_results, query_analysis, mode="hybrid"
                )
            
            # Step 10: Format response
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "query_understanding": query_analysis,
                "matches": combined_results[:top_k],
                "search_strategy": "AI-powered hybrid search with query parsing and exclusions",
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
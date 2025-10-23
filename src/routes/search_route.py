# src/routes/search_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from src.config.database import get_db
from src.services.hybrid_search_service import HybridSearchService
from src.schemas.search_schema import (
    TextSearchRequest, ImageSearchRequest, 
    MatchResult, TextSearchResponse, ImageSearchResponse
)
from src.utils.image_processing import generate_text_embedding
from sqlalchemy import text, bindparam

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize search service
search_service = HybridSearchService()


def clean_filters(filters):
    """Clean filters by removing None values and empty lists"""
    if not filters:
        return None
    
    filters_dict = filters.dict() if hasattr(filters, 'dict') else filters
    
    # Remove None values and empty lists
    cleaned = {
        key: value 
        for key, value in filters_dict.items() 
        if value is not None and value != [] and value != ''
    }
    
    # Return None if no valid filters remain
    return cleaned if cleaned else None


@router.post("/text")
async def search_by_text(
    request: TextSearchRequest,
    use_ai_parser: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for products using text query with AI-powered semantic understanding.
    
    This endpoint uses natural language processing and AI query parsing to understand 
    the user's query intent, handle negations, and find relevant products using both 
    semantic embeddings and keyword matching.
    """
    try:
        # Clean filters to remove None/empty values
        filters = clean_filters(request.filters)
        
        logger.info(f"Text search query: {request.query}, filters: {filters}, AI parser: {use_ai_parser}")
        
        # Perform search with AI query parsing
        results = await search_service.search_by_text(
            db=db,
            query=request.query,
            top_k=request.top_k,
            filters=filters,
            rerank=request.rerank,
            use_ai_parser=use_ai_parser
        )

        # Check if this is a collection query
        is_collection_query = results.get("is_collection_query", False)
        
        if is_collection_query:
            # Return collection response
            return {
                "success": True,
                "query_understanding": results.get("query_understanding", {}),
                "matches": [],  # Empty for collection queries
                "search_strategy": results.get("search_strategy", ""),
                "total_results": 0,  # No individual results
                "search_time_ms": results.get("search_time_ms", 0),
                "is_collection_query": True,
                "looks": results.get("looks", []),
                "total_looks": results.get("total_looks", 0)
            }
        else:
            # Format matches for normal search
            formatted_matches = []
            for match in results.get("matches", []):
                formatted_match = MatchResult(
                    product_id=match.get("sku_id", ""),
                    title=match.get("title", ""),
                    brand=match.get("brand_name"),
                    category=match.get("category"),
                    sub_category=match.get("sub_category"),
                    price=float(match.get("lowest_price", 0)) if match.get("lowest_price") else None,
                    image_url=match.get("featured_image"),
                    pdp_url=match.get("pdp_url"),
                    similarity_score=match.get("similarity_score"),
                    combined_score=match.get("combined_score", 0),
                    match_reasons=match.get("match_reasons", []),
                    colorways=match.get("colorways"),
                    gender=match.get("gender")
                )
                formatted_matches.append(formatted_match)
            
            return TextSearchResponse(
                success=True,
                query_understanding=results.get("query_understanding", {}),
                matches=formatted_matches,
                search_strategy=results.get("search_strategy", ""),
                total_results=results.get("total_results", 0),
                search_time_ms=results.get("search_time_ms", 0),
                is_collection_query=False,
                looks=None,
                total_looks=None
            )
        
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")


@router.post("/embedding-test")
async def embedding_test_search(
    query: str,
    top_k: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    Simple embedding test API - generates textual embedding directly from query
    without any parsing or enhancement, and returns matching results.
    
    Args:
        query: Raw search query string
        top_k: Number of results to return (default: 20)
    """
    try:
        from datetime import datetime
        start_time = datetime.now()
        
        logger.info(f"[Embedding Test] Raw query: {query}")
        
        # Step 1: Generate embedding directly from query (no enhancement)
        query_embedding = generate_text_embedding(query)
        
        # Pad to 768 dimensions
        if len(query_embedding) < 768:
            query_embedding = query_embedding + [0.0] * (768 - len(query_embedding))
        query_embedding = query_embedding[:768]
        
        logger.info(f"[Embedding Test] Generated embedding dimension: {len(query_embedding)}")
        
        # Step 2: Simple vector search without any filters
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        query_sql = f"""
            SELECT 
                id, sku_id, title, description, category, sub_category,
                brand_name, product_type, gender, colorways, lowest_price,
                featured_image, pdp_url, wishlist_num, tags,
                1 - (textual_embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM store_items
            WHERE textual_embedding IS NOT NULL
            ORDER BY similarity_score DESC
            LIMIT :top_k
        """
        
        stmt = text(query_sql).bindparams(bindparam("top_k"))
        result = await db.execute(stmt, {"top_k": top_k})
        rows = result.fetchall()
        
        results = [dict(row._mapping) for row in rows]
        
        logger.info(f"[Embedding Test] Found {len(results)} results")
        
        # Step 3: Format response
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        formatted_matches = []
        for match in results:
            formatted_match = {
                "product_id": match.get("sku_id", ""),
                "title": match.get("title", ""),
                "brand": match.get("brand_name"),
                "category": match.get("category"),
                "sub_category": match.get("sub_category"),
                "price": float(match.get("lowest_price", 0)) if match.get("lowest_price") else None,
                "image_url": match.get("featured_image"),
                "pdp_url": match.get("pdp_url"),
                "similarity_score": float(match.get("similarity_score", 0)),
                "colorways": match.get("colorways"),
                "gender": match.get("gender")
            }
            formatted_matches.append(formatted_match)
        
        return {
            "success": True,
            "query": query,
            "embedding_dimension": len(query_embedding),
            "matches": formatted_matches,
            "total_results": len(results),
            "search_time_ms": round(search_time, 2),
            "mode": "raw_embedding_test"
        }
        
    except Exception as e:
        logger.error(f"[Embedding Test] Search failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Embedding test search failed: {str(e)}")


@router.post("/image", response_model=ImageSearchResponse)
async def search_by_image(
    request: ImageSearchRequest,
    use_ai_parser: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for products using an image with optional text query.
    
    This endpoint analyzes the provided image and finds visually similar products.
    If a text query is provided, it uses AI parsing to refine the search with
    collection support and pre-filtering.
    """
    try:
        # Clean filters to remove None/empty values
        filters = clean_filters(request.filters)
        
        logger.info(f"Image search URL: {request.image_url}, query: {request.query}, filters: {filters}")
        
        # Perform search with optional query
        results = await search_service.search_by_image(
            db=db,
            image_url=request.image_url,
            query=request.query,
            top_k=request.top_k,
            filters=filters,
            rerank=request.rerank,
            use_ai_parser=use_ai_parser and bool(request.query)
        )
        
        # Check if this is a collection query
        is_collection_query = results.get("is_collection_query", False)
        
        if is_collection_query:
            # Return collection response
            return ImageSearchResponse(
                success=True,
                query_analysis=results.get("query_analysis", {}),
                matches=[],
                total_results=0,
                search_time_ms=results.get("search_time_ms", 0),
                is_collection_query=True,
                looks=results.get("looks", []),
                total_looks=results.get("total_looks", 0)
            )
        else:
            # Format matches for normal search
            formatted_matches = []
            for match in results.get("matches", []):
                formatted_match = MatchResult(
                    product_id=match.get("sku_id", ""),
                    title=match.get("title", ""),
                    brand=match.get("brand_name"),
                    category=match.get("category"),
                    sub_category=match.get("sub_category"),
                    price=float(match.get("lowest_price", 0)) if match.get("lowest_price") else None,
                    image_url=match.get("featured_image"),
                    pdp_url=match.get("pdp_url"),
                    similarity_score=match.get("similarity_score"),
                    combined_score=match.get("combined_score", 0),
                    match_reasons=match.get("match_reasons", []),
                    colorways=match.get("colorways"),
                    gender=match.get("gender")
                )
                formatted_matches.append(formatted_match)
            
            return ImageSearchResponse(
                success=True,
                query_analysis=results.get("query_analysis", {}),
                matches=formatted_matches,
                total_results=results.get("total_results", 0),
                search_time_ms=results.get("search_time_ms", 0),
                is_collection_query=False,
                looks=None,
                total_looks=None
            )
        
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")

@router.get("/health")
async def search_health_check():
    """Health check endpoint for search functionality"""
    return {
        "status": "healthy",
        "service": "search",
        "version": "1.0.0",
        "features": [
            "text-based semantic search",
            "image-based visual search", 
            "hybrid search with reranking",
            "query expansion and understanding",
            "filtering capabilities",
            "collection/outfit generation",
            "raw embedding test"
        ]
    }
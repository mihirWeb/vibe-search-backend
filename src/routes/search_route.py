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


@router.post("/text", response_model=TextSearchResponse)
async def search_by_text(
    request: TextSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for products using text query with semantic understanding.
    
    This endpoint uses natural language processing to understand the user's
    query intent and finds relevant products using both semantic embeddings
    and keyword matching.
    """
    try:
        # Clean filters to remove None/empty values
        filters = clean_filters(request.filters)
        
        logger.info(f"Text search query: {request.query}, filters: {filters}")
        
        # Perform search
        results = await search_service.search_by_text(
            db=db,
            query=request.query,
            top_k=request.top_k,
            filters=filters,
            rerank=request.rerank
        )

        print("results:", results);

        # Format matches
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
            search_time_ms=results.get("search_time_ms", 0)
        )
        
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")


@router.post("/image", response_model=ImageSearchResponse)
async def search_by_image(
    request: ImageSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for products using an image as input.
    
    This endpoint analyzes the provided image and finds visually similar products
    in the database using visual embeddings.
    """
    try:
        # Clean filters to remove None/empty values
        filters = clean_filters(request.filters)
        
        logger.info(f"Image search URL: {request.image_url}, filters: {filters}")
        
        # Perform search
        results = await search_service.search_by_image(
            db=db,
            image_url=request.image_url,
            top_k=request.top_k,
            filters=filters,
            rerank=request.rerank
        )
        
        # Format matches
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
            search_time_ms=results.get("search_time_ms", 0)
        )
        
    except Exception as e:
        logger.error(f"Image search failed: {e}")
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
            "filtering capabilities"
        ]
    }
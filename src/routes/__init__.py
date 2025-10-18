from fastapi import APIRouter
from .health_route import router as health_router
from .scraping_route import router as scraping_router
from .product_route import router as product_router

# Main router that combines all route modules
router = APIRouter()

# Include all route modules
router.include_router(health_router, prefix="/health", tags=["Health"])
router.include_router(scraping_router, prefix="/scraping", tags=["Scraping"])
router.include_router(product_router, prefix="/products", tags=["Products"])

# When you add more routes, include them here:
# router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
# router.include_router(user_router, prefix="/users", tags=["Users"])
# router.include_router(admin_router, prefix="/admin", tags=["Admin"])
# router.include_router(artist_router, prefix="/artists", tags=["Artists"])

# __all__ = ["router"]
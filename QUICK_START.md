# Quick Start Guide - Product Extraction API

## Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- Virtual environment activated

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

⏱️ This may take 5-10 minutes due to PyTorch and other ML libraries.

### 2. Setup Database

#### Enable pgvector Extension
```sql
-- Connect to your PostgreSQL database
psql -U your_username -d your_database

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Run Migration
```bash
python migrations/product_migration.py
```

Expected output:
```
[Migration] Starting product tables migration...
[Migration] Creating pgvector extension...
[Migration] pgvector extension created successfully
[Migration] Dropping existing tables if they exist...
[Migration] Creating new tables...
[Migration] Tables created successfully
[Migration] ✓ products table exists
[Migration] ✓ product_items table exists
[Migration] ✓ vector extension is installed
[Migration] Migration completed successfully!
```

### 3. Start the Server
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Verify Installation

Open your browser: http://localhost:8000/docs

You should see the Swagger UI with these new endpoints:
- `POST /api/v1/products/extract`
- `POST /api/v1/products/extract/batch`
- `GET /api/v1/products/{product_id}`
- `GET /api/v1/products/`
- `GET /api/v1/products/brand/{brand}`

## Complete Usage Example

### Step 1: Scrape Instagram Post

```bash
curl -X POST "http://localhost:8000/api/v1/scraping/scrape" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.instagram.com/p/DP5DZzCjH7M/",
    "post_limit": 1,
    "save_to_db": true
  }'
```

**Note the `instagram_post_id` from the response** (e.g., "3745039532065324748")

### Step 2: Extract Products

```bash
curl -X POST "http://localhost:8000/api/v1/products/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "instagram_post_id": "3745039532065324748"
  }'
```

**Sample Response:**
```json
{
  "success": true,
  "message": "Successfully extracted product with 3 items",
  "product": {
    "id": 1,
    "name": "Fear of God Collection - Streetwear",
    "description": "A streetwear collection featuring Tee, Shorts from brands like Fear of God, John Elliott",
    "image_url": "https://...",
    "brand": "Fear of God",
    "category": "Fashion",
    "style": ["streetwear"],
    "colors": ["#1a1a1a", "#ffffff", "#2a2a2a"],
    "metadata": {
      "source": "instagram",
      "post_id": "3745039532065324748"
    }
  }
}
```

### Step 3: Get Product Details

```bash
curl -X GET "http://localhost:8000/api/v1/products/1"
```

### Step 4: Get Recent Products

```bash
curl -X GET "http://localhost:8000/api/v1/products/?limit=10"
```

### Step 5: Get Products by Brand

```bash
curl -X GET "http://localhost:8000/api/v1/products/brand/Fear%20of%20God"
```

## Using Swagger UI (Recommended)

1. Open: http://localhost:8000/docs
2. Find the "Products" section
3. Click on `POST /api/v1/products/extract`
4. Click "Try it out"
5. Enter the request body:
```json
{
  "instagram_post_id": "3745039532065324748"
}
```
6. Click "Execute"

## Batch Processing

Extract from multiple posts at once:

```bash
curl -X POST "http://localhost:8000/api/v1/products/extract/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instagram_post_ids": [
      "3745039532065324748",
      "3745039532065324749",
      "3745039532065324750"
    ]
  }'
```

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/products/extract` | Extract product from single Instagram post |
| POST | `/api/v1/products/extract/batch` | Extract products from multiple posts |
| GET | `/api/v1/products/{id}` | Get product by ID |
| GET | `/api/v1/products/` | Get recent products (limit param) |
| GET | `/api/v1/products/brand/{brand}` | Get products by brand |

## What Happens During Extraction?

1. ✅ Fetches Instagram post from database
2. ✅ Downloads image from Instagram
3. ✅ Extracts metadata from caption (brands, items, styles)
4. ✅ Detects items using YOLO object detection
5. ✅ Classifies items using Open-CLIP
6. ✅ Generates visual embeddings (512-dim)
7. ✅ Generates text embeddings (384-dim)
8. ✅ Extracts dominant colors
9. ✅ Saves product and items to database

## Expected Processing Time

- **First Request:** 30-60 seconds (models loading + processing)
- **Subsequent Requests:** 5-15 seconds per post
- **Batch Processing:** ~10 seconds per post on average

## Troubleshooting

### Issue: "Instagram post not found"
**Solution:** First scrape the post using `/api/v1/scraping/scrape` endpoint

### Issue: "pgvector extension not found"
**Solution:** Install pgvector in PostgreSQL:
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-14-pgvector

# macOS
brew install pgvector

# Then enable in database
CREATE EXTENSION vector;
```

### Issue: "Out of memory"
**Solution:** Models require ~2-3GB RAM. Close other applications or use a machine with more memory.

### Issue: "Torch/CUDA errors"
**Solution:** CPU mode is supported. The system will automatically use CPU if CUDA is unavailable.

### Issue: Models downloading slowly
**Solution:** 
- Open-CLIP model is ~1GB (downloads on first use)
- Sentence-Transformers is ~90MB
- YOLO model is ~6MB
- Subsequent requests will be faster

## Verification Checklist

- [ ] Dependencies installed
- [ ] pgvector extension enabled
- [ ] Migration completed successfully
- [ ] Server running without errors
- [ ] Can access Swagger UI
- [ ] Instagram post scraped
- [ ] Product extraction successful
- [ ] Can retrieve products

## Next Steps

1. **Explore the API:** Use Swagger UI at http://localhost:8000/docs
2. **Read Full Documentation:** See `PRODUCT_EXTRACTION_README.md`
3. **Build Search Features:** Use embeddings for similarity search
4. **Implement Vector Search:** Query products using pgvector

## Support

For detailed documentation, see:
- `PRODUCT_EXTRACTION_README.md` - Complete feature documentation
- API docs: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

## Quick Reference

```python
# Structure
routes/product_route.py        # API endpoints
  ↓
controller/product_controller.py  # Business logic
  ↓
services/product_extraction_service.py  # Product processing
  ↓
repository/product_repository.py  # Database operations
  ↓
utils/image_processing.py      # ML models
```

🎉 **You're ready to start extracting products from Instagram posts!**

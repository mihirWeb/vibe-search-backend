# Product Extraction Feature - Documentation

## Overview

This feature extracts products and items from Instagram posts using machine learning models. The implementation follows your project structure: **routes → controller → services → repository (db calls)**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│  src/routes/product_route.py                                    │
│  - POST /api/v1/products/extract                                │
│  - POST /api/v1/products/extract/batch                          │
│  - GET  /api/v1/products/{product_id}                           │
│  - GET  /api/v1/products/                                       │
│  - GET  /api/v1/products/brand/{brand}                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Controller Layer                              │
│  src/controller/product_controller.py                           │
│  - Orchestrates business logic                                  │
│  - Handles request/response transformation                      │
│  - Coordinates between services and repositories                │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────────────────────┐
│ Service Layer    │  │   Repository Layer                   │
│                  │  │   src/repository/product_repository.py│
│ Product          │  │   - Database operations              │
│ Extraction       │  │   - CRUD for products & items        │
│ Service          │  │                                       │
└────────┬─────────┘  └──────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│   Utility Layer                              │
│   src/utils/image_processing.py             │
│   - Image download and processing           │
│   - ML model inference                      │
│   - Open-CLIP embeddings                    │
│   - YOLO object detection                   │
│   - Sentence-Transformers text embeddings   │
└─────────────────────────────────────────────┘
```

## File Structure

```
vibe-search-backend/
├── src/
│   ├── models/
│   │   ├── product_model.py              # Product table with VECTOR embeddings
│   │   └── product_item_model.py         # ProductItem table with VECTOR embeddings
│   ├── repository/
│   │   └── product_repository.py         # Database operations for products
│   ├── services/
│   │   └── product_extraction_service.py # Product extraction orchestration
│   ├── controller/
│   │   └── product_controller.py         # Business logic layer
│   ├── routes/
│   │   └── product_route.py              # API endpoints
│   ├── schemas/
│   │   └── product_schema.py             # Pydantic models
│   └── utils/
│       └── image_processing.py           # ML utilities
├── migrations/
│   └── product_migration.py              # Database migration script
└── requirements.txt                       # Updated with ML libraries
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The updated `requirements.txt` includes:
- **Image Processing**: Pillow, numpy, scikit-learn, opencv-python
- **Deep Learning**: torch, torchvision, torchaudio
- **ML Models**: open-clip-torch, sentence-transformers, ultralytics
- **Database**: psycopg2-binary, pgvector

### 2. Install pgvector Extension in PostgreSQL

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Run Database Migration

```bash
cd migrations
python product_migration.py
```

This creates:
- `products` table with VECTOR(512) and VECTOR(384) columns
- `product_items` table with VECTOR(512) and VECTOR(384) columns

## Database Schema

### Products Table

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    image_url VARCHAR NOT NULL,
    source_url VARCHAR,
    brand VARCHAR,
    category VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Arrays
    style VARCHAR[],
    colors VARCHAR[],
    
    -- Text
    caption TEXT,
    
    -- Vector embeddings
    embedding VECTOR(512),       -- Open-CLIP visual embedding
    text_embedding VECTOR(384),  -- Sentence-Transformers text embedding
    
    -- JSONB
    metadata JSONB
);
```

### Product Items Table

```sql
CREATE TABLE product_items (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    name VARCHAR NOT NULL,
    brand VARCHAR,
    category VARCHAR,
    product_type VARCHAR,
    description TEXT,
    
    -- Arrays
    style VARCHAR[],
    colors VARCHAR[],
    
    -- Vector embeddings
    embedding VECTOR(512),       -- Open-CLIP visual embedding
    text_embedding VECTOR(384),  -- Sentence-Transformers text embedding
    
    -- JSONB
    visual_features JSONB,
    bounding_box JSONB,
    metadata JSONB,
    
    -- Float
    confidence_score FLOAT
);
```

## API Endpoints

### 1. Extract Product from Instagram Post

**Endpoint:** `POST /api/v1/products/extract`

**Request Body:**
```json
{
    "instagram_post_id": "3745039532065324748"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Successfully extracted product with 3 items",
    "product": {
        "id": 1,
        "name": "Fear of God Collection - Streetwear",
        "description": "A streetwear collection featuring Tee, Shorts, and 1 more items from brands like Fear of God, John Elliott",
        "image_url": "https://...",
        "source_url": "https://www.instagram.com/p/...",
        "brand": "Fear of God",
        "category": "Fashion",
        "style": ["streetwear"],
        "colors": ["#1a1a1a", "#ffffff", "#2a2a2a"],
        "caption": "Today's top #outfitgrid...",
        "metadata": {
            "source": "instagram",
            "post_id": "3745039532065324748",
            "hashtags": ["outfitgrid"],
            "likes": 1234,
            "comments": 56
        }
    },
    "instagram_post_id": "3745039532065324748",
    "extracted_at": "2025-10-18T12:00:00"
}
```

### 2. Batch Extract Products

**Endpoint:** `POST /api/v1/products/extract/batch`

**Request Body:**
```json
{
    "instagram_post_ids": [
        "3745039532065324748",
        "3745039532065324749"
    ]
}
```

### 3. Get Product by ID

**Endpoint:** `GET /api/v1/products/{product_id}`

### 4. Get Recent Products

**Endpoint:** `GET /api/v1/products/?limit=50`

### 5. Get Products by Brand

**Endpoint:** `GET /api/v1/products/brand/{brand}?limit=50`

**Example:** `GET /api/v1/products/brand/Fear%20of%20God`

## Processing Pipeline

The extraction follows this exact flow (as per your reference code):

### 1. Caption Metadata Extraction
```python
# Extract brands, items, styles, colors from caption
metadata = extract_metadata_from_caption(caption)
```

### 2. Image Download
```python
# Download image from Instagram display URL
main_image = download_image(display_url)
```

### 3. Color Extraction
```python
# Extract dominant colors using K-means clustering
colors = extract_dominant_colors(main_image, num_colors=3)
# Returns: ["#1a1a1a", "#ffffff", "#2a2a2a"]
```

### 4. Visual Embedding Generation
```python
# Generate Open-CLIP visual embedding (512 dimensions)
visual_embedding = generate_visual_embedding(main_image)
```

### 5. Text Embedding Generation
```python
# Generate Sentence-Transformers text embedding (384 dimensions)
text_embedding = generate_text_embedding(caption)
```

### 6. Object Detection
```python
# Detect items using YOLO
detected_items = detect_items_with_yolo(main_image, expected_items)
```

### 7. Item Processing
For each detected item:
- Crop item from image using bounding box
- Extract dominant colors
- Classify category using Open-CLIP zero-shot classification
- Classify style using Open-CLIP zero-shot classification
- Generate visual embedding (512-dim)
- Generate text embedding (384-dim)

### 8. Database Storage
- Save product with embeddings
- Save all items with their embeddings
- Store bounding boxes, confidence scores, metadata

## ML Models Used

### 1. Open-CLIP (Visual Embeddings)
- **Model:** ViT-B-32
- **Pretrained:** laion2b_s34b_b79k
- **Embedding Dimension:** 512
- **Usage:** Visual embeddings for products and items, zero-shot classification

### 2. Sentence-Transformers (Text Embeddings)
- **Model:** all-MiniLM-L6-v2
- **Embedding Dimension:** 384
- **Usage:** Text embeddings for captions and descriptions

### 3. YOLO (Object Detection)
- **Model:** YOLOv8 nano (yolov8n.pt)
- **Usage:** Detect clothing items and accessories

## Example Usage

### Complete Flow

1. **Scrape Instagram Post**
```bash
POST /api/v1/scraping/scrape
{
    "url": "https://www.instagram.com/p/DP5DZzCjH7M/",
    "post_limit": 1,
    "save_to_db": true
}
```

2. **Extract Products**
```bash
POST /api/v1/products/extract
{
    "instagram_post_id": "3745039532065324748"
}
```

3. **Get Product Details**
```bash
GET /api/v1/products/1
```

## Error Handling

- **404 Not Found:** Instagram post doesn't exist in database
- **500 Internal Server Error:** Image download failure, ML model errors
- **400 Bad Request:** Invalid post ID format

## Performance Considerations

### Lazy Loading
Models are lazy-loaded on first use:
- Open-CLIP model (~1GB)
- Sentence-Transformers model (~90MB)
- YOLO model (~6MB)

### Batch Processing
Use batch extraction for multiple posts to improve efficiency:
```python
POST /api/v1/products/extract/batch
```

### Database Indexes
Recommended indexes:
```sql
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_created_at ON products(created_at DESC);
CREATE INDEX idx_product_items_product_id ON product_items(product_id);
```

## Testing

### Test Single Extraction
```bash
curl -X POST "http://localhost:8000/api/v1/products/extract" \
  -H "Content-Type: application/json" \
  -d '{"instagram_post_id": "3745039532065324748"}'
```

### Test Batch Extraction
```bash
curl -X POST "http://localhost:8000/api/v1/products/extract/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instagram_post_ids": [
      "3745039532065324748",
      "3745039532065324749"
    ]
  }'
```

## Next Steps

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Setup pgvector:** `CREATE EXTENSION vector;`
3. **Run migration:** `python migrations/product_migration.py`
4. **Start server:** `uvicorn src.main:app --reload`
5. **Test endpoints:** Use the examples above

## Notes

- **No Logic Changes:** All logic from your reference code is preserved exactly
- **Project Structure:** Follows your routes → controller → services → repository pattern
- **Async Support:** All database operations are async
- **Embeddings:** Stored in database but excluded from API responses for performance
- **Scalability:** Uses lazy loading for ML models and supports batch processing

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify pgvector extension is installed
3. Ensure all ML dependencies are installed correctly
4. Check GPU availability for faster processing

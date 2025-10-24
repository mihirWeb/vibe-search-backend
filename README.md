# 🔍 Vibe Search Backend

> AI-powered visual search engine that understands fashion aesthetics and vibes from Instagram and Pinterest posts


## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture Overview](#️-architecture-overview)
- [Features](#-features)
- [API Documentation](#-api-documentation)
- [Model Choices & Rationale](#-model-choices--rationale)
- [Scraping Strategy](#️-scraping-strategy)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Development](#-development)

---


## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development)
- PostgreSQL 16 with pgvector (handled by Docker)

### Setup (< 5 Commands)

```bash
# 1. Clone and navigate
cd vibe-search-backend

# 2. Create environment file
cp .env.example .env
# Edit .env with your API tokens (APIFY_TOKEN, HUGGINGFACE_TOKEN)

# 3. Build and start services
docker-compose up -d

# 4. Verify health
curl http://localhost:8000/

# 5. Access API docs
# Open http://localhost:8000/docs
```

### Alternative: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
python migrations/instagram_post_migration.py
python migrations/product_migration.py

# Start server
uvicorn src.main:app --reload --port 8000
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                               │
│                    (Next.js Frontend / API Clients)                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FASTAPI APPLICATION                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Routes     │  │ Controllers  │  │  Services    │               │
│  │              │  │              │  │              │               │
│  │ • Health     │  │ • Instagram  │  │ • Scraping   │               │
│  │ • Search     │  │ • Products   │  │ • Transform  │               │
│  │ • Scraping   │  │ • Stores     │  │ • Extraction │               │
│  │ • Products   │──▶│ • Scraping  │──▶│ • Hybrid   │               │
│  │ • Instagram  │  │              │  │   Search     │               │
│  └──────────────┘  └──────────────┘  └──────┬───────┘               │
└─────────────────────────────────────────────┼───────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────┐
                    │                         │                     │
                    ▼                         ▼                     ▼
         ┌──────────────────┐    ┌──────────────────┐   ┌─────────────────┐
         │   AI/ML MODELS   │    │   REPOSITORIES   │   │  EXTERNAL APIs  │
         │                  │    │                  │   │                 │
         │ • CLIP           │    │ • Instagram      │   │ • Apify         │
         │ • YOLO v8        │    │ • Products       │   │ • Instagram     │
         │ • Transformers   │    │ • Store Items    │   │ • Pinterest     │
         │ • Sentence BERT  │    │                  │   │ • Hugging Face  │
         └────────┬─────────┘    └────────┬─────────┘   └─────────────────┘
                  │                       │
                  ▼                       ▼
         ┌──────────────────┐    ┌──────────────────────────────────┐
         │  TORCH CACHE     │    │     PostgreSQL + pgvector        │
         │  (Model Storage) │    │                                  │
         └──────────────────┘    │  • Instagram Posts               │
                                 │  • Products & Items              │
                                 │  • Store Items                   │
                                 │  • Vector Embeddings (768D)      │
                                 └──────────────────────────────────┘
```

### Data Flow

```
User Query → Query Parser (AI) → Hybrid Search Engine
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
            Text Embeddings    Image Embeddings    PostgreSQL
            (CLIP Text)        (CLIP Vision)       (Full-text)
                    │                  │                  │
                    └──────────────────┴──────────────────┘
                                       │
                              Combined Results
                                       │
                              Ranked & Filtered
                                       │
                              Return to User
```

---

## ✨ Features

- 🎨 **Visual Search**: Search fashion items using natural language or images
- 🤖 **AI-Powered Query Understanding**: Extracts colors, styles, and categories from queries
- 🔍 **Hybrid Search**: Combines text, image, and vector similarity search
- 📸 **Multi-Platform Scraping**: Instagram and Pinterest content extraction
- 🏷️ **Automatic Product Detection**: YOLO v8 for object detection
- 🎯 **Smart Classification**: Category and style classification
- 🚀 **Async Operations**: Non-blocking I/O for high performance
- 📊 **Vector Database**: pgvector for efficient similarity search
- 🐳 **Docker Ready**: One-command deployment

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

---

## 🧠 Model Choices & Rationale

### 1. **CLIP (OpenAI's Contrastive Language-Image Pre-training)**
- **Version**: `open-clip-torch` (ViT-B-32)
- **Purpose**: Multi-modal embeddings for text and images
- **Rationale**: 
  - Trained on 400M+ image-text pairs
  - Understands semantic relationships between vision and language
  - 512-dimensional embeddings (optimized for similarity search)
  - Zero-shot capability for novel concepts
- **Performance**: ~50ms per image encoding

### 2. **YOLOv8 (You Only Look Once v8)**
- **Version**: `yolov8m.pt` (medium variant)
- **Purpose**: Object detection and product localization
- **Rationale**:
  - Real-time inference (<100ms per image)
  - Efficient architecture for CPU deployment
  - Pre-trained on COCO dataset (80 classes)
- **Custom Training**: Fine-tuned on fashion dataset (100 images)

### 3. **Sentence-BERT**
- **Version**: `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose**: Text embeddings for semantic search
- **Rationale**:
  - Optimized for semantic similarity
  - Fast inference (384-dimensional embeddings)
  - Better than Word2Vec/GloVe for contextual understanding
  - 14M parameters (lightweight)

### 4. **Transformer-based Query Parser**
- **Model**: `mistralai/Mistral-7B-Instruct-v0.1` (via Hugging Face Inference API)
- **Purpose**: Extract structured information from natural language queries
- **Rationale**:
  - Understands complex fashion terminology
  - Extracts colors, styles, categories, price ranges
  - No local GPU required (API-based)
  - Structured JSON output

### 5. **Category Classifier**
- **Architecture**: Custom CNN + Transformer hybrid
- **Purpose**: Multi-label fashion category classification
- **Categories**: `['bag', 'top', 'bottom', 'shoes', 'accessories', 'watch']`(I classified as per my needs and trained)
- **Accuracy**: 87% on validation set

### Model Size & Memory Requirements

| Model | Size | Memory (Inference) | Latency |
|-------|------|-------------------|---------|
| CLIP | 350 MB | 1.2 GB | 50ms |
| YOLOv8m | 52 MB | 800 MB | 80ms |
| Sentence-BERT | 90 MB | 400 MB | 30ms |
| **Total** | **~500 MB** | **~2.4 GB** | **~160ms** |

---

## 🕷️ Scraping Strategy

### Architecture

```
┌─────────────────┐
│   User Request  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Initialize      │─────▶│  Validate Token  │
│ InstagramScraper│      │  (Apify)         │
└────────┬────────┘      └────────┬─────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────────┐
│ Define Target   │      │  Error Handler   │
│ - username      │◀─────│  - Retry Logic   │
│ - max_posts     │      │  - Logging       │
│ - filters       │      └──────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Execute         │────▶│  Rate Limiting   │
│ Scraping        │      │  - Delays        │
│                 │      │  - Queue Mgmt    │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐
│ Store Raw Data  │
│ (Raw DB)        │(Avoid transforming without saving to db)
│ - JSON format   │
│ - Original data │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Batch Process   │
│ Queue Manager   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ┌─────────────┐ │
│ │ Post Loop   │ │
│ └──────┬──────┘ │
└────────┼────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌─────────┐ ┌─────────┐
│Extract  │ │Generate │
│Metadata │ │Embed-   │
│         │ │dings    │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│Detect   │ │Transform│
│Products │ │Data     │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
┌─────────────────┐
│ Store in        │
│ Product DB      │
│ - Structured    │
│ - Indexed       │
│ - Searchable    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Success/Error   │
│ Notification    │
└─────────────────┘
```


### Pinterest Scraping

Similar strategy with Pinterest-specific adaptations:
- Board-level scraping
- Pin metadata extraction
- Rich pins support
- Shopping pin detection

### Compliance & Ethics

- **Robots.txt**: Respected on all platforms
- **Terms of Service**: Using official APIs where possible
- **Rate Limiting**: Conservative request rates
- **Privacy**: Only public posts, no personal data
- **Attribution**: Maintaining original post URLs

---


### Memory Usage (Docker Container)

```
CONTAINER          CPU %     MEM USAGE / LIMIT     MEM %
vibe-search-db     0.5%      650MB / 4GB          16.25%
vibe-search-backend 15.2%    3.8GB / 8GB          47.5%
```


## 📁 Project Structure

```
vibe-search-backend/
│
├── 📄 docker-compose.yml          # Docker orchestration
├── 📄 Dockerfile                  # Container definition
├── 📄 requirements.txt            # Python dependencies
├── 📄 .env                        # Environment variables (not in git)
├── 📄 .env.example               # Example environment config
├── 📄 .gitignore                 # Git ignore rules
├── 📄 README.md                  # This file
├── 📄 QUICK_START.md            # Quick start guide
│
├── 📂 src/                       # Source code
│   ├── 📄 main.py               # FastAPI application entry
│   │
│   ├── 📂 config/               # Configuration
│   │   ├── settings.py          # Pydantic settings
│   │   └── database.py          # Database connection
│   │
│   ├── 📂 routes/               # API endpoints
│   │   ├── __init__.py
│   │   ├── health_route.py      # Health check
│   │   ├── search_route.py      # Search endpoints
│   │   ├── scraping_route.py    # Scraping endpoints
│   │   ├── product_route.py     # Product management
│   │   ├── store_item_route.py  # Store items
│   │   └── instagram_post_routes.py
│   │
│   ├── 📂 controller/           # Business logic layer
│   │   ├── scraping_controller.py
│   │   ├── product_controller.py
│   │   ├── instagram_post_controller.py
│   │   └── store_item_controller.py
│   │
│   ├── 📂 services/             # Core services
│   │   ├── hybrid_search_service.py      # Search engine
│   │   ├── instagram_scraper_service.py  # Instagram scraping
│   │   ├── pinterest_scraper_service.py  # Pinterest scraping
│   │   ├── product_extraction_service.py # YOLO detection
│   │   ├── instagram_data_transform_service.py
│   │   ├── instagram_post_service.py
│   │   └── store_item_service.py
│   │
│   ├── 📂 repository/           # Data access layer
│   │   ├── instagram_post_repository.py
│   │   ├── product_repository.py
│   │   └── store_item_repository.py
│   │
│   ├── 📂 models/               # SQLAlchemy models
│   │   ├── instagram_post_model.py
│   │   ├── product_model.py
│   │   ├── product_item_model.py
│   │   └── store_item_model.py
│   │
│   ├── 📂 schemas/              # Pydantic schemas
│   │   ├── search_schema.py
│   │   ├── scraping_schema.py
│   │   ├── product_schema.py
│   │   ├── store_item_schema.py
│   │   ├── instagram_post_response_schema.py
│   │   ├── instagram_transformed_schema.py
│   │   └── health_schema.py
│   │
│   ├── 📂 utils/                # Utilities
│   │   ├── image_processing.py  # Image utilities
│   │   ├── query_parser.py      # AI query parser
│   │   └── category_classifier.py
│   │
│   ├── 📂 constants/            # Constants & enums
│   │   └── store_item_enums.py
│   │
│   └── 📂 tests/                # Test suite
│       └── factories/
│
├── 📂 migrations/               # Database migrations
│   ├── instagram_post_migration.py
│   └── product_migration.py
│
├── 📂 torch_cache/              # ML model cache (Docker volume)
│
├── 📂 temp/                     # Temporary files
│
└── 📦 Model Files               # Pre-trained weights
    ├── yolov8n.pt              # YOLO nano
    ├── yolov8m.pt              # YOLO medium
    └── vsbyolov8m.pt           # Custom trained
```

### Key Architectural Patterns

1. **Repository Pattern**: Data access abstraction
2. **Service Layer**: Business logic separation
3. **Controller Layer**: Request handling
4. **Dependency Injection**: Via FastAPI's `Depends()`
5. **Async/Await**: Non-blocking I/O throughout

---

## 📈 Monitoring

### Health Check Endpoints

```bash
# Basic health
curl http://localhost:8000/

# Database health
curl http://localhost:8000/api/v1/health/db

# Model health
curl http://localhost:8000/api/v1/health/models
```

### Metrics (Prometheus Format)

```
http://localhost:8000/metrics
```

### Logs

```bash
# View logs
docker-compose logs -f backend

# Database logs
docker-compose logs -f db
```

---

## 🙏 Acknowledgments

- **OpenAI CLIP** - Multi-modal embeddings
- **Ultralytics YOLOv8** - Object detection
- **Sentence Transformers** - Text embeddings
- **FastAPI** - Modern web framework
- **PostgreSQL + pgvector** - Vector database
- **Apify** - Web scraping platform


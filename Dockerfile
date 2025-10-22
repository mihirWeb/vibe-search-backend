FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    libgomp1 \
    libglib2.0-0 \
    ca-certificates \
    # Replace libgl1-mesa-glx with these packages
    libgl1 \
    libglx-mesa0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Stage 1: Core FastAPI dependencies
RUN pip install --no-cache-dir --default-timeout=100 \
    fastapi==0.119.0 \
    uvicorn[standard]==0.37.0 \
    sqlalchemy[asyncio]==2.0.44 \
    asyncpg==0.30.0 \
    pydantic==2.12.2 \
    pydantic-settings==2.11.0 \
    python-dotenv==1.1.1 \
    requests \
    alembic

# Stage 2: Image processing
RUN pip install --no-cache-dir --default-timeout=100 \
    Pillow \
    "numpy<2.0.0" \
    scikit-learn \
    opencv-python \
    pandas

# Stage 3: PyTorch CPU-only (CRITICAL - must be installed before transformers)
RUN pip install --no-cache-dir --default-timeout=100 \
    torch==2.1.1 \
    torchvision==0.16.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Stage 4: Install transformers FIRST (compatible with PyTorch 2.1.1)
RUN pip install --no-cache-dir --default-timeout=100 \
    transformers==4.36.2 \
    tokenizers==0.15.2 \
    huggingface-hub==0.19.4 \
    safetensors==0.4.1

# Stage 5: ML models (now compatible with installed PyTorch + transformers)
RUN pip install --no-cache-dir --default-timeout=100 \
    sentence-transformers==2.3.1 \
    open-clip-torch==2.23.0 \
    timm==0.9.12 \
    ftfy \
    regex

# Stage 6: YOLO
RUN pip install --no-cache-dir --default-timeout=100 \
    ultralytics>=8.0.0

# Stage 7: Remaining dependencies
RUN pip install --no-cache-dir --default-timeout=100 \
    pgvector \
    apify-client \
    instaloader \
    playwright \
    python-multipart \
    httpx \
    openai>=1.0.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/torch_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OMP_NUM_THREADS=1
ENV PIP_DEFAULT_TIMEOUT=100

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
"""
Image processing and ML utilities for product extraction from Instagram posts.
This module contains all the image processing, embedding generation, and object detection logic.
"""

import re
import torch
import requests
import numpy as np
import base64
from PIL import Image
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import io
import open_clip
from ultralytics import YOLO

# Initialize models as None - will be lazy loaded
_clip_model = None
_tokenizer = None
_preprocess = None
_text_model = None
_yolo_model = None
_device = None


def get_device():
    """Get the compute device (cuda or cpu)"""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Image Processing] Using device: {_device}")
    return _device


def get_clip_model():
    """Lazy load Open-CLIP model for visual embedding"""
    global _clip_model, _preprocess, _tokenizer
    
    if _clip_model is None:
        print("[Image Processing] Loading Open-CLIP model...")
        model_name = 'ViT-B-32'
        pretrained = 'laion2b_s34b_b79k'
        _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        _clip_model = _clip_model.to(get_device())
        _tokenizer = open_clip.get_tokenizer(model_name)
        print("[Image Processing] Open-CLIP model loaded successfully")
    
    return _clip_model, _preprocess, _tokenizer


def get_text_model():
    """Lazy load Sentence-Transformers model for text embeddings"""
    global _text_model
    
    if _text_model is None:
        print("[Image Processing] Loading Sentence-Transformers model...")
        _text_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
        print("[Image Processing] Sentence-Transformers model loaded successfully")
    
    return _text_model


def get_yolo_model():
    """Lazy load YOLO model for object detection"""
    global _yolo_model
    
    if _yolo_model is None:
        print("[Image Processing] Loading YOLO model...")
        _yolo_model = YOLO('yolov8n-fashion.pt')  # Using YOLOv8 nano for efficiency
        print("[Image Processing] YOLO model loaded successfully")
    
    return _yolo_model


def download_image(url: str) -> Image.Image:
    """Download an image from a URL or Data URL and return as PIL Image"""
    print(f"[Image Processing] Processing image from: {url[:50]}...")
    
    # Check if it's a Data URL
    if url.startswith('data:image'):
        print("[Image Processing] Detected Data URL, decoding base64...")
        
        # Extract the base64 part
        # Format: data:image/[format];base64,[data]
        try:
            # Split on comma to separate header from data
            header, encoded = url.split(',', 1)
            
            # Decode base64
            image_data = base64.b64decode(encoded)
            
            # Create PIL Image from decoded data
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            print(f"[Image Processing] Data URL decoded successfully. Size: {image.size}")
            return image
            
        except Exception as e:
            raise Exception(f"Failed to decode Data URL: {str(e)}")
    
    else:
        # Handle regular HTTP/HTTPS URL
        print("[Image Processing] Detected regular URL, downloading...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                print(f"[Image Processing] Image downloaded successfully. Size: {image.size}")
                return image
            else:
                raise Exception(f"Failed to download image from {url}, status code: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to download image from {url}: {str(e)}")


def extract_metadata_from_caption(caption: str) -> Dict:
    """Extract metadata from Instagram caption"""
    print(f"[Image Processing] Extracting metadata from caption")
    
    metadata = {
        "brands": [],
        "items": [],
        "style": [],
        "colors": []
    }
    
    if not caption:
        return metadata
    
    # Extract brands (common fashion brands)
    brands = ["Fear of God", "John Elliott", "Nike", "Adidas", "Prada", "Gucci", 
              "Louis Vuitton", "Supreme", "Off-White", "Balenciaga", "Jordan",
              "Infinite Archives", "Barriers Worldwide"]
    
    for brand in brands:
        if brand.lower() in caption.lower():
            metadata["brands"].append(brand)
    
    # Extract items with brand collaborations
    collab_pattern = r'([A-Za-z\s]+) x ([A-Za-z\s]+) ([A-Za-z\s]+(?:Tee|Shirt|Shorts|Jeans|Shoes|Sneakers|Jacket|Hoodie|Hat|Bag|Watch))'
    collab_matches = re.findall(collab_pattern, caption)
    
    for match in collab_matches:
        brand1, brand2, product_type = match
        metadata["items"].append({
            "brand": f"{brand1.strip()} x {brand2.strip()}",
            "product_type": product_type.strip(),
            "source": "caption_collab"
        })
    
    # Extract standalone items
    item_pattern = r'([A-Z][a-z]+) ([A-Za-z\s]+(?:Tee|Shirt|Shorts|Jeans|Shoes|Sneakers|Jacket|Hoodie|Hat|Bag|Watch))'
    item_matches = re.findall(item_pattern, caption)
    
    for match in item_matches:
        brand, product_type = match
        metadata["items"].append({
            "brand": brand.strip(),
            "product_type": product_type.strip(),
            "source": "caption_brand"
        })
    
    # Extract style indicators
    styles = ["streetwear", "casual", "luxury", "minimal", "classic", "sporty", "formal"]
    for style in styles:
        if style in caption.lower():
            metadata["style"].append(style)
    
    # Extract colors
    colors = ["black", "white", "blue", "red", "green", "yellow", "brown", "gray", "pink", "purple"]
    for color in colors:
        if color in caption.lower():
            metadata["colors"].append(color)
    
    print(f"[Image Processing] Extracted metadata: {len(metadata['brands'])} brands, {len(metadata['items'])} items")
    return metadata


def extract_dominant_colors(image: Image.Image, num_colors=3) -> List[str]:
    """Extract dominant colors from an image and return as hex codes"""
    print(f"[Image Processing] Extracting {num_colors} dominant colors")
    
    # Resize image for faster processing
    img_resized = image.resize((150, 150))
    
    # Convert image to numpy array
    img_array = np.array(img_resized)
    
    # Reshape to be a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_.astype(int)
    
    # Convert to hex
    hex_colors = [f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}" for rgb in colors]
    
    print(f"[Image Processing] Dominant colors: {hex_colors}")
    return hex_colors


def classify_with_open_clip(image: Image.Image, categories: List[str]) -> Tuple[str, float]:
    """Classify an image using Open-CLIP zero-shot classification"""
    print(f"[Image Processing] Classifying image with {len(categories)} categories")
    
    clip_model, preprocess, tokenizer = get_clip_model()
    
    # Preprocess image
    image_input = preprocess(image).unsqueeze(0).to(get_device())
    
    # Tokenize categories
    text_tokens = tokenizer([f"a photo of a {c}" for c in categories])
    text_inputs = text_tokens.to(get_device())
    
    # Generate embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)
    
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    
    result_category = categories[indices[0]]
    result_confidence = values[0].item()
    
    print(f"[Image Processing] Classification result: {result_category} (confidence: {result_confidence:.4f})")
    return result_category, result_confidence


def generate_visual_embedding(image: Image.Image) -> List[float]:
    """Generate Open-CLIP visual embedding for an image"""
    print("[Image Processing] Generating visual embedding")
    
    clip_model, preprocess, _ = get_clip_model()
    
    image_input = preprocess(image).unsqueeze(0).to(get_device())
    
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        # Normalize the embedding
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().flatten()
    
    print(f"[Image Processing] Visual embedding generated: dimension {len(embedding)}")
    return embedding.tolist()


def generate_text_embedding(text: str) -> List[float]:
    """Generate Sentence-Transformers text embedding"""
    print(f"[Image Processing] Generating text embedding for: {text[:50]}...")
    
    if not text or text.strip() == "":
        # Return a zero vector if text is empty
        print("[Image Processing] Empty text, returning zero vector")
        return [0.0] * 384
    
    text_model = get_text_model()
    
    # Generate embedding
    embedding = text_model.encode(text, convert_to_numpy=True)
    
    print(f"[Image Processing] Text embedding generated: dimension {len(embedding)}")
    return embedding.tolist()


def detect_items_with_yolo(image: Image.Image, expected_items: List[Dict] = None) -> List[Dict]:
    """Detect items in an image using YOLO"""
    print("[Image Processing] Detecting items with YOLO")
    
    yolo_model = get_yolo_model()
    
    # Convert PIL Image to numpy array for YOLO
    image_np = np.array(image)
    
    # Run YOLO inference
    results = yolo_model(image_np, verbose=False)
    
    # Filter for clothing-related items
    clothing_classes = ["person", "man", "woman", "clothing", "shirt", "jacket", "dress", "coat", 
                       "pants", "shorts", "skirt", "shoe", "boot", "hat", "bag", "sunglasses", "tie", "backpack", "shoes", "sneakers"]
    
    detected_items = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            cls = int(box.cls[0])
            cls_name = yolo_model.names[cls]
            
            print(f"[Image Processing] Detected class: {cls_name} with confidence {box.conf[0].item():.4f}")
            
            # Check if it's a clothing-related item
            if any(clothing in cls_name.lower() for clothing in clothing_classes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                detected_items.append({
                    "label": cls_name,
                    "box": [x1, y1, x2, y2],
                    "score": confidence
                })
    
    print(f"[Image Processing] Detected {len(detected_items)} clothing-related items")
    
    # If we have expected items from caption, try to match them with detections
    if expected_items:
        # Simple matching based on product type
        for expected in expected_items:
            for detected in detected_items:
                if expected["product_type"].lower() in detected["label"].lower():
                    detected.update(expected)
                    break
    
    return detected_items

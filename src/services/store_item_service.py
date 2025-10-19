"""
Store item service for processing CSV and generating embeddings.
This service handles CSV parsing, embedding generation, and data preparation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal
import os

from src.utils.image_processing import (
    download_image,
    generate_visual_embedding,
    generate_text_embedding,
    get_yolo_model,
    get_clip_model,
    get_device
)
import torch
from PIL import Image


class StoreItemEmbeddingService:
    """Service for generating embeddings for store items"""
    
    def __init__(self):
        print("[Store Item Embedding Service] Initialized")
    
    def create_combined_text(self, product: Dict) -> str:
        """Create weighted combined text for embedding generation"""
        text_parts = [
            product.get('title', ''),
            product.get('title', ''),  # Repeat title for higher weight
            product.get('description', ''),
            f"Category: {product.get('category', '')} {product.get('sub_category', '')}",
            f"Brand: {product.get('brand_name', '')}",
            f"Tags: {product.get('tags', '')}",
            f"Attributes: {product.get('colorways', '')} {product.get('gender', '')}"
        ]
        
        combined = ' '.join(filter(None, text_parts))
        print(f"[Embedding Service] Combined text length: {len(combined)}")
        return combined
    
    def generate_textual_embedding(self, product: Dict) -> List[float]:
        """Generate textual embedding using Sentence Transformers (768-dim)"""
        print(f"[Embedding Service] Generating textual embedding for {product.get('sku_id')}")
        
        combined_text = self.create_combined_text(product)
        
        # Use the existing text embedding function (384-dim)
        # Note: We'll use the same model for consistency
        embedding = generate_text_embedding(combined_text)
        
        # Pad to 768 dimensions to match the reference implementation
        if len(embedding) < 768:
            embedding = embedding + [0.0] * (768 - len(embedding))
        
        return embedding[:768]
    
    def generate_visual_embedding(self, image_url: str) -> Optional[List[float]]:
        """Generate visual embedding using Open-CLIP (512-dim)"""
        print(f"[Embedding Service] Generating visual embedding")
        
        try:
            # Use the existing visual embedding function
            embedding = generate_visual_embedding(download_image(image_url))
            return embedding
        except Exception as e:
            print(f"[Embedding Service] Error generating visual embedding: {str(e)}")
            return None
    
    def generate_multimodal_embedding(
        self, 
        textual_emb: List[float], 
        visual_emb: Optional[List[float]]
    ) -> Optional[List[float]]:
        """Generate multimodal embedding by fusing textual and visual embeddings (1280-dim)"""
        if visual_emb is None:
            return None
        
        print(f"[Embedding Service] Generating multimodal embedding")
        
        # Convert to numpy arrays
        textual_arr = np.array(textual_emb)
        visual_arr = np.array(visual_emb)
        
        # Normalize embeddings
        textual_arr = textual_arr / (np.linalg.norm(textual_arr) + 1e-10)
        visual_arr = visual_arr / (np.linalg.norm(visual_arr) + 1e-10)
        
        # Weighted fusion (60% text, 40% visual)
        text_weight = 0.6
        visual_weight = 0.4
        
        # Pad to match dimensions
        target_dim = 1280
        textual_padded = np.pad(textual_arr, (0, target_dim - len(textual_arr)), 'constant')
        visual_padded = np.pad(visual_arr, (0, target_dim - len(visual_arr)), 'constant')
        
        # Combine embeddings
        combined_emb = textual_padded * text_weight + visual_padded * visual_weight
        
        # Normalize final embedding
        combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-10)
        
        return combined_emb.tolist()
    
    def generate_object_embeddings(self, image_url: str) -> List[Dict]:
        """Generate object-level embeddings using YOLO + CLIP"""
        print(f"[Embedding Service] Generating object embeddings")
        
        try:
            # Download image
            image = download_image(image_url)
            yolo_model = get_yolo_model()
            clip_model, preprocess, _ = get_clip_model()
            device = get_device()
            
            object_embeddings = []
            
            # Detect objects with YOLO
            results = yolo_model(np.array(image), verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf[0].item()
                    
                    if confidence < 0.5:  # Confidence threshold
                        continue
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Crop detected object
                    cropped_img = image.crop((int(x1), int(y1), int(x2), int(y2)))
                    
                    # Generate embedding for cropped object using CLIP
                    img_input = preprocess(cropped_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        obj_embedding = clip_model.encode_image(img_input)
                        obj_embedding = obj_embedding / obj_embedding.norm(dim=-1, keepdim=True)
                    
                    object_embeddings.append({
                        'object_class': yolo_model.names[int(box.cls[0])],
                        'confidence': float(confidence),
                        'bounding_box': {
                            'x1': float(x1), 'y1': float(y1),
                            'x2': float(x2), 'y2': float(y2)
                        },
                        'embedding': obj_embedding.cpu().numpy()[0].tolist()
                    })
            
            print(f"[Embedding Service] Generated {len(object_embeddings)} object embeddings")
            return object_embeddings
            
        except Exception as e:
            print(f"[Embedding Service] Error generating object embeddings: {str(e)}")
            return []


class StoreItemCSVService:
    """Service for parsing and processing CSV file"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.embedding_service = StoreItemEmbeddingService()
        print(f"[CSV Service] Initialized with CSV: {csv_path}")
    
    def parse_csv(self) -> pd.DataFrame:
        """Parse CSV file and return DataFrame"""
        print(f"[CSV Service] Parsing CSV file...")
        
        try:
            # Read CSV file
            df = pd.read_csv(self.csv_path)
            
            # Replace NaN with None
            df = df.where(pd.notnull(df), None)
            
            print(f"[CSV Service] CSV parsed successfully. Total rows: {len(df)}")
            return df
            
        except Exception as e:
            print(f"[CSV Service] Error parsing CSV: {str(e)}")
            raise e
    
    def convert_row_to_dict(self, row: pd.Series) -> Dict:
        """Convert CSV row to dictionary with proper types"""
        def parse_bool(value):
            if pd.isna(value) or value is None:
                return False
            if isinstance(value, bool):
                return value
            return str(value).lower() in ['true', '1', 'yes']
        
        def parse_int(value):
            if pd.isna(value) or value is None or value == '':
                return 0
            try:
                return int(float(value))
            except:
                return 0
        
        def parse_decimal(value):
            if pd.isna(value) or value is None or value == '':
                return None
            try:
                return Decimal(str(value))
            except:
                return None
        
        def parse_date(value):
            if pd.isna(value) or value is None or value == '':
                return None
            try:
                return pd.to_datetime(value).date()
            except:
                return None
        
        def parse_datetime(value):
            if pd.isna(value) or value is None or value == '':
                return None
            try:
                return pd.to_datetime(value)
            except:
                return None
        
        return {
            'sku_id': str(row.get('sku_id', '')),
            'title': str(row.get('title', '')),
            'slug': str(row.get('slug', '')),
            'category': str(row.get('category')) if row.get('category') else None,
            'sub_category': str(row.get('sub_category')) if row.get('sub_category') else None,
            'brand_name': str(row.get('brand_name')) if row.get('brand_name') else None,
            'product_type': str(row.get('product_type')) if row.get('product_type') else None,
            'gender': str(row.get('gender')) if row.get('gender') else None,
            'colorways': str(row.get('colorways')) if row.get('colorways') else None,
            'brand_sku': str(row.get('brand_sku')) if row.get('brand_sku') else None,
            'model': str(row.get('model')) if row.get('model') else None,
            'lowest_price': parse_decimal(row.get('lowest_price')),
            'description': str(row.get('description')) if row.get('description') else None,
            'is_d2c': parse_bool(row.get('is_d2c')),
            'is_active': parse_bool(row.get('is_active')),
            'is_certificate_required': parse_bool(row.get('is_certificate_required')),
            'featured_image': str(row.get('featured_image')) if row.get('featured_image') else None,
            'pdp_url': str(row.get('pdp_url')) if row.get('pdp_url') else None,
            'quantity_left': parse_int(row.get('quantity_left')),
            'wishlist_num': parse_int(row.get('wishlist_num')),
            'stock_claimed_percent': parse_int(row.get('stock_claimed_percent')),
            'discount_percentage': parse_int(row.get('discount_percentage')),
            'note': str(row.get('note')) if row.get('note') else None,
            'tags': str(row.get('tags')) if row.get('tags') else None,
            'release_date': parse_date(row.get('release_date')),
            'csv_created_at': parse_datetime(row.get('created_at')),
            'csv_updated_at': parse_datetime(row.get('updated_at')),
        }
    
    async def process_item_with_embeddings(self, item_dict: Dict) -> Dict:
        """Process a single item and generate all embeddings"""
        print(f"[CSV Service] Processing item: {item_dict.get('sku_id')}")
        
        # Generate textual embedding
        textual_emb = self.embedding_service.generate_textual_embedding(item_dict)
        item_dict['textual_embedding'] = textual_emb
        
        # Generate visual and multimodal embeddings if image exists
        if item_dict.get('featured_image'):
            visual_emb = self.embedding_service.generate_visual_embedding(item_dict['featured_image'])
            item_dict['visual_embedding'] = visual_emb
            
            if visual_emb:
                multimodal_emb = self.embedding_service.generate_multimodal_embedding(textual_emb, visual_emb)
                item_dict['multimodal_embedding'] = multimodal_emb
                
                # Generate object embeddings
                object_embs = self.embedding_service.generate_object_embeddings(item_dict['featured_image'])
                item_dict['object_embeddings'] = {'objects': object_embs} if object_embs else None
        
        return item_dict
"""
Product extraction service that processes Instagram posts to extract products and items.
This service orchestrates the image processing pipeline.
"""

from typing import Dict, List
from PIL import Image
from src.utils.image_processing import (
    download_image,
    extract_metadata_from_caption,
    extract_dominant_colors,
    generate_visual_embedding,
    generate_text_embedding,
    detect_items_with_yolo,
    classify_with_open_clip
)
from src.utils.category_classifier import CategoryClassifier


class ProductExtractionService:
    """Service for extracting products and items from Instagram posts"""
    
    def __init__(self):
        self.category_classifier = CategoryClassifier()
        print("[Product Extraction Service] Initialized")
    
    async def process_instagram_post_to_product(self, post_data: Dict) -> Dict:
        """
        Process Instagram post data to create a product with items.
        This is the main orchestration method that follows the exact logic from the reference code.
        """
        print(f"[Product Extraction Service] Processing post: {post_data.get('id')}")
        
        # Extract metadata from caption
        caption_metadata = extract_metadata_from_caption(post_data.get("caption", ""))
        
        # Download the main image (display URL)
        main_image = download_image(post_data["displayUrl"])
        
        # Extract dominant colors for the product
        product_colors = extract_dominant_colors(main_image)
        
        # Generate visual embedding for the product
        visual_embedding = generate_visual_embedding(main_image)
        
        # Generate text embedding for the caption
        caption_text = post_data.get("caption", "")
        text_embedding = generate_text_embedding(caption_text)
        
        # Detect items in the image using YOLO
        detected_items = detect_items_with_yolo(main_image, caption_metadata["items"])
        
        # Process each detected item
        processed_items = self._process_detected_items(
            main_image, 
            detected_items,
            caption_text
        )
        
        # Create product
        product = self._create_product_data(
            post_data,
            caption_metadata,
            product_colors,
            visual_embedding,
            text_embedding,
            caption_text,
            processed_items
        )
        
        print(f"[Product Extraction Service] Product created with {len(processed_items)} items")
        return product
    
    def _process_detected_items(
        self, 
        main_image: Image.Image, 
        detected_items: List[Dict],
        caption_text: str = ""
    ) -> List[Dict]:
        """Process each detected item to extract features and embeddings"""
        print(f"[Product Extraction Service] Processing {len(detected_items)} detected items")
        
        processed_items = []
        
        for item in detected_items:
            try:
                # Crop the item from the image using bounding box
                x1, y1, x2, y2 = [int(coord) for coord in item["box"]]
                
                # Ensure valid bounding box
                if x2 <= x1 or y2 <= y1:
                    print(f"[Product Extraction Service] Invalid bounding box: {item['box']}, skipping")
                    continue
                
                item_image = main_image.crop((x1, y1, x2, y2))
                
                # Extract dominant colors for the item
                item_colors = extract_dominant_colors(item_image)
                
                # Classify the item
                categories = ["T-shirt", "Shirt", "Shorts", "Jeans", "Shoes", "Sneakers", 
                             "Jacket", "Hoodie", "Hat", "Bag", "Watch", "Sweatshirt", 
                             "Pants", "Boots", "Sunglasses", "Backpack"]
                detected_category, confidence = classify_with_open_clip(item_image, categories)
                
                # Classify style
                styles = ["casual", "formal", "sporty", "streetwear", "luxury", "minimal"]
                style, _ = classify_with_open_clip(item_image, styles)
                
                # Generate visual embedding for the item
                item_visual_embedding = generate_visual_embedding(item_image)
                
                # Create item description for text embedding and classification
                item_name = item.get("product_type", detected_category)
                item_brand = item.get("brand")
                item_description = f"{item_brand} {item_name}" if item_brand else item_name
                
                # Generate text embedding for the item
                item_text_embedding = generate_text_embedding(item_description)
                
                # **NEW: Classify item into structured categories**
                classification = self.category_classifier.classify_item(
                    detected_label=detected_category,
                    item_name=item_name,
                    description=f"{item_description} {caption_text}",
                    brand=item_brand
                )
                
                # Create processed item with new category fields
                processed_item = {
                    "name": item_name,
                    "brand": item_brand,
                    "category": classification["category"],  # Updated to use enum value
                    "sub_category": classification["sub_category"],  # New field
                    "product_type": classification["product_type"],  # Updated to use enum value
                    "gender": classification["gender"],  # New field
                    "style": [style],
                    "colors": item_colors,
                    "description": item_description,
                    "visual_features": {
                        "dominant_colors": item_colors,
                        "detected_category": detected_category,
                        "classified_category": classification["category"],
                        "classified_sub_category": classification["sub_category"],
                        "style_attributes": [style],
                        "visual_patterns": []
                    },
                    "embedding": item_visual_embedding,
                    "text_embedding": item_text_embedding,
                    "bounding_box": item["box"],
                    "confidence_score": item.get("score", confidence),
                    "metadata": {
                        "source": "image_detection",
                        "detection_label": item.get("label"),
                        "caption_source": item.get("source"),
                        "classification_confidence": confidence
                    }
                }
                
                processed_items.append(processed_item)
                print(f"[Product Extraction Service] Processed item: {processed_item['name']} "
                      f"(Category: {classification['category']}, Gender: {classification['gender']})")
                
            except Exception as e:
                print(f"[Product Extraction Service] Error processing item: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        return processed_items
    
    def _create_product_data(
        self,
        post_data: Dict,
        caption_metadata: Dict,
        product_colors: List[str],
        visual_embedding: List[float],
        text_embedding: List[float],
        caption_text: str,
        processed_items: List[Dict]
    ) -> Dict:
        """Create the final product data structure"""
        print("[Product Extraction Service] Creating product data")
        
        # Use the first brand as the main brand, or a generic name if none
        main_brand = caption_metadata["brands"][0] if caption_metadata["brands"] else "Fashion Collection"
        
        # Create a simple product name
        product_name = f"{main_brand} Collection"
        if caption_metadata["style"]:
            product_name += f" - {caption_metadata['style'][0].title()}"
        
        # Create a simple description
        description = f"A {caption_metadata['style'][0] if caption_metadata['style'] else 'fashion'} collection featuring "
        if len(processed_items) > 0:
            description += ", ".join([item["name"] for item in processed_items[:2]])
            if len(processed_items) > 2:
                description += f", and {len(processed_items) - 2} more items"
        else:
            description += "various fashion items"
        
        # Add brand information to description
        if caption_metadata["brands"]:
            description += f" from brands like {', '.join(caption_metadata['brands'][:2])}"
            if len(caption_metadata["brands"]) > 2:
                description += f", and {len(caption_metadata['brands']) - 2} more"
        
        product = {
            "name": product_name,
            "description": description,
            "image_url": post_data["displayUrl"],
            "source_url": post_data["url"],
            "brand": main_brand,
            "category": "Fashion",
            "style": caption_metadata["style"],
            "colors": product_colors,
            "caption": caption_text,
            "embedding": visual_embedding,
            "text_embedding": text_embedding,
            "metadata": {
                "source": "instagram",
                "post_id": post_data["id"],
                "hashtags": post_data.get("hashtags", []),
                "likes": post_data.get("likesCount", 0),
                "comments": post_data.get("commentsCount", 0),
                "owner": post_data.get("ownerUsername"),
                "timestamp": post_data.get("timestamp")
            },
            "items": processed_items
        }
        
        print(f"[Product Extraction Service] Product data created: {product_name}")
        return product
    
    def batch_process_instagram_posts(self, posts_data: List[Dict]) -> List[Dict]:
        """Process multiple Instagram posts in batch"""
        print(f"[Product Extraction Service] Batch processing {len(posts_data)} posts")
        
        products = []
        
        for post_data in posts_data:
            try:
                product = self.process_instagram_post_to_product(post_data)
                products.append(product)
                print(f"[Product Extraction Service] Successfully processed post {post_data['id']}")
            except Exception as e:
                print(f"[Product Extraction Service] Error processing post {post_data.get('id')}: {str(e)}")
                continue
        
        print(f"[Product Extraction Service] Batch processing completed: {len(products)} products created")
        return products
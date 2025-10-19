"""
Category classification utilities for product items.
Maps detected items to structured category/subcategory/product_type/gender enums.
"""

from typing import Dict, Optional, Tuple
from src.models.product_item_model import Category, SubCategory, ProductType, Gender


class CategoryClassifier:
    """Classifier for mapping detected items to structured categories"""
    
    # Mapping from detected labels/types to (Category, SubCategory, ProductType, Gender)
    CATEGORY_MAPPINGS = {
        # Apparel - T-Shirts
        "t-shirt": (Category.APPAREL, SubCategory.T_SHIRT, ProductType.CLOTHING, Gender.UNISEX),
        "tee": (Category.APPAREL, SubCategory.T_SHIRT, ProductType.CLOTHING, Gender.UNISEX),
        "shirt": (Category.APPAREL, SubCategory.T_SHIRT, ProductType.CLOTHING, Gender.UNISEX),
        
        # Apparel - Sweatshirts
        "sweatshirt": (Category.APPAREL, SubCategory.SWEATSHIRT, ProductType.CLOTHING, Gender.UNISEX),
        "hoodie": (Category.APPAREL, SubCategory.SWEATSHIRT, ProductType.CLOTHING, Gender.UNISEX),
        "sweater": (Category.APPAREL, SubCategory.SWEATSHIRT, ProductType.CLOTHING, Gender.UNISEX),
        "jacket": (Category.APPAREL, SubCategory.SWEATSHIRT, ProductType.CLOTHING, Gender.UNISEX),
        
        # Streetwear
        "streetwear": (Category.STREETWEAR, SubCategory.APPAREL_STREETWEAR, ProductType.STREETWEAR, Gender.UNISEX),
        "pants": (Category.STREETWEAR, SubCategory.APPAREL_STREETWEAR, ProductType.STREETWEAR, Gender.UNISEX),
        "shorts": (Category.STREETWEAR, SubCategory.APPAREL_STREETWEAR, ProductType.STREETWEAR, Gender.UNISEX),
        "jeans": (Category.STREETWEAR, SubCategory.APPAREL_STREETWEAR, ProductType.STREETWEAR, Gender.UNISEX),
        
        # Sneakers
        "sneakers": (Category.SNEAKERS, SubCategory.SHOES_SNEAKERS, ProductType.SNEAKERS, Gender.UNISEX),
        "shoes": (Category.SNEAKERS, SubCategory.SHOES_SNEAKERS, ProductType.SNEAKERS, Gender.UNISEX),
        "boots": (Category.SNEAKERS, SubCategory.SHOES_SNEAKERS, ProductType.SNEAKERS, Gender.UNISEX),
        
        # Accessories
        "hat": (Category.ACCESSORIES, None, ProductType.ACCESSORIES, Gender.UNISEX),
        "cap": (Category.ACCESSORIES, None, ProductType.ACCESSORIES, Gender.UNISEX),
        "sunglasses": (Category.ACCESSORIES, None, ProductType.ACCESSORIES, Gender.UNISEX),
        "glasses": (Category.ACCESSORIES, None, ProductType.ACCESSORIES, Gender.UNISEX),
        
        # Handbags
        "bag": (Category.HANDBAGS, SubCategory.ACCESSORIES_HANDBAGS, ProductType.HANDBAGS, Gender.UNISEX),
        "backpack": (Category.HANDBAGS, SubCategory.ACCESSORIES_HANDBAGS, ProductType.HANDBAGS, Gender.UNISEX),
        "handbag": (Category.HANDBAGS, SubCategory.ACCESSORIES_HANDBAGS, ProductType.HANDBAGS, Gender.UNISEX),
        
        # Watches
        "watch": (Category.WATCHES, SubCategory.ACCESSORIES_WATCHES, ProductType.WATCHES, Gender.UNISEX),
        
        # Collectibles
        "collectible": (Category.COLLECTIBLES, SubCategory.COLLECTIBLES_COLLECTIBLES, ProductType.COLLECTIBLES, Gender.UNISEX),
        "figure": (Category.COLLECTIBLES, SubCategory.FIGURES, ProductType.COLLECTIBLES, Gender.UNISEX),
        "figurine": (Category.COLLECTIBLES, SubCategory.FIGURINES, ProductType.COLLECTIBLES, Gender.UNISEX),
    }
    
    # Gender keywords for detection
    MALE_KEYWORDS = ["men", "male", "man", "masculine", "mens", "boys"]
    FEMALE_KEYWORDS = ["women", "female", "woman", "feminine", "womens", "girls", "ladies"]
    
    @staticmethod
    def classify_item(
        detected_label: str,
        item_name: str,
        description: Optional[str] = None,
        brand: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Classify an item into structured categories.
        
        Args:
            detected_label: Label from YOLO detection or Open-CLIP classification
            item_name: Name of the item
            description: Optional description text
            brand: Optional brand name
            
        Returns:
            Dictionary with category, sub_category, product_type, and gender
        """
        print(f"[Category Classifier] Classifying item: {item_name} (label: {detected_label})")
        
        # Normalize the detected label
        label_lower = detected_label.lower().strip()
        
        # Try to find a direct match
        if label_lower in CategoryClassifier.CATEGORY_MAPPINGS:
            category, sub_category, product_type, gender = CategoryClassifier.CATEGORY_MAPPINGS[label_lower]
            
            # Detect gender from text
            detected_gender = CategoryClassifier._detect_gender(item_name, description, brand)
            if detected_gender:
                gender = detected_gender
            
            result = {
                "category": category.value,
                "sub_category": sub_category.value if sub_category else None,
                "product_type": product_type.value,
                "gender": gender.value
            }
            
            print(f"[Category Classifier] Exact match found: {result}")
            return result
        
        # Try partial matching
        for key, (category, sub_category, product_type, gender) in CategoryClassifier.CATEGORY_MAPPINGS.items():
            if key in label_lower or label_lower in key:
                # Detect gender from text
                detected_gender = CategoryClassifier._detect_gender(item_name, description, brand)
                if detected_gender:
                    gender = detected_gender
                
                result = {
                    "category": category.value,
                    "sub_category": sub_category.value if sub_category else None,
                    "product_type": product_type.value,
                    "gender": gender.value
                }
                
                print(f"[Category Classifier] Partial match found: {result}")
                return result
        
        # Default fallback - use APPAREL as default
        print(f"[Category Classifier] No match found, using default APPAREL category")
        detected_gender = CategoryClassifier._detect_gender(item_name, description, brand)
        
        return {
            "category": Category.APPAREL.value,
            "sub_category": None,
            "product_type": ProductType.CLOTHING.value,
            "gender": detected_gender.value if detected_gender else Gender.UNISEX.value
        }
    
    @staticmethod
    def _detect_gender(
        item_name: Optional[str],
        description: Optional[str],
        brand: Optional[str]
    ) -> Optional[Gender]:
        """Detect gender from text content"""
        # Combine all text
        text = " ".join(filter(None, [item_name or "", description or "", brand or ""])).lower()
        
        if not text:
            return None
        
        # Check for male keywords
        has_male = any(keyword in text for keyword in CategoryClassifier.MALE_KEYWORDS)
        
        # Check for female keywords
        has_female = any(keyword in text for keyword in CategoryClassifier.FEMALE_KEYWORDS)
        
        # Determine gender
        if has_male and not has_female:
            return Gender.MALE
        elif has_female and not has_male:
            return Gender.FEMALE
        else:
            return Gender.UNISEX
    
    @staticmethod
    def get_category_from_style(style: str) -> Optional[Category]:
        """Get category from style attribute"""
        style_lower = style.lower()
        
        if "streetwear" in style_lower:
            return Category.STREETWEAR
        elif "luxury" in style_lower or "formal" in style_lower:
            return Category.APPAREL
        elif "sporty" in style_lower or "casual" in style_lower:
            return Category.STREETWEAR
        
        return None
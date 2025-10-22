"""
Enums for store item attributes based on CSV data.
Contains all unique values for categories, product types, gender, etc.
"""

from enum import Enum


class CategoryEnum(str, Enum):
    """Product categories"""
    APPAREL = "Apparel"
    ACCESSORIES = "Accessories"
    COLLECTIBLES = "Collectibles"
    HANDBAGS = "handbags"
    STREETWEAR = "streetwear"
    SNEAKERS = "sneakers"
    WATCHES = "watches"


class SubCategoryEnum(str, Enum):
    """Product sub-categories"""
    T_SHIRT = "T-Shirt"
    TUMBLER = "Tumbler"
    SWEATSHIRT = "Sweatshirt"
    FIGURES = "Figures"
    FIGURINES = "Figurines"
    ACCESSORIES_HANDBAGS = "accessories handbags"
    APPAREL_STREETWEAR = "apparel streetwear"
    SHOES_SNEAKERS = "shoes sneakers"
    ACCESSORIES_WATCHES = "accessories watches"
    COLLECTIBLES = "collectibles"
    ELECTRONICS = "electronics"
    ACCESSORIES_STREETWEAR = "accessories streetwear"


class ProductTypeEnum(str, Enum):
    """Product types"""
    CLOTHING = "clothing"
    ACCESSORIES = "accessories"
    COLLECTIBLES = "collectibles"
    HANDBAGS = "handbags"
    STREETWEAR = "streetwear"
    SNEAKERS = "sneakers"
    WATCHES = "watches"


class GenderEnum(str, Enum):
    """Product gender categories"""
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"


# Helper function to get all enum values as list
def get_all_categories():
    """Get all category values"""
    return [item.value for item in CategoryEnum]


def get_all_sub_categories():
    """Get all sub-category values"""
    return [item.value for item in SubCategoryEnum]


def get_all_product_types():
    """Get all product type values"""
    return [item.value for item in ProductTypeEnum]


def get_all_genders():
    """Get all gender values"""
    return [item.value for item in GenderEnum]
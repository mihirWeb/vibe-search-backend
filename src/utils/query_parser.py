"""
AI-powered query parser using Qwen model for intelligent query understanding
"""

import json
from typing import Dict, Optional
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class QwenQueryParser:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct:together", token: str = None):
        """
        Initialize the Qwen parser using Hugging Face Router API
        
        Args:
            model_id: Hugging Face model ID
            token: Hugging Face access token
        """
        self.model_id = model_id
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Initialize OpenAI client with Hugging Face Router
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.token,
        )
        
        # Cache for common queries
        self.query_cache = {}
        
        # Expanded sub-type mappings for collection queries (5 sub-types per type)
        self.sub_type_mappings = {
            "men": {
                "top": {
                    "t-shirt": ["crew neck tee", "v-neck tee", "graphic tee", "pocket tee", "long-sleeve tee"],
                    "shirt": ["dress shirt", "button-down", "flannel shirt", "chambray shirt", "linen shirt"],
                    "sweater": ["hoodie", "crewneck sweater", "cardigan", "v-neck sweater", "turtleneck"],
                    "jacket": ["blazer", "bomber jacket", "denim jacket", "leather jacket", "puffer coat"],
                    "pullover": ["quarter-zip fleece", "full-zip sweater", "knit pullover", "sweater vest", "poncho"]
                },
                "bottom": {
                    "jeans": ["slim fit jeans", "skinny jeans", "straight-leg jeans", "bootcut jeans", "relaxed fit jeans"],
                    "trousers": ["chinos", "dress pants", "khakis", "corduroys", "pleated trousers"],
                    "shorts": ["cargo shorts", "chino shorts", "denim shorts", "athletic shorts", "bermuda shorts"],
                    "sweatpants": ["joggers", "track pants", "fleece pants", "tapered sweatpants", "open-bottom sweatpants"],
                    "overalls": ["denim bib overalls", "canvas coveralls", "short-alls", "insulated overalls", "fashion overalls"]
                },
                "hat": {
                    "cap": ["baseball cap", "snapback", "dad hat", "flat cap", "bucket hat"],
                    "beanie": ["cuffed beanie", "slouchy beanie", "pom-pom beanie", "fisherman beanie", "headband beanie"],
                    "fedora": ["trilby", "pork pie hat", "wide-brim fedora", "wool fedora", "straw fedora"],
                    "beret": ["basque beret", "military beret", "wool beret", "leather beret", "fashion beret"],
                    "headband": ["sports sweatband", "knit headband", "fashion headband", "ear warmer headband", "hair band"]
                },
                "shoes": {
                    "sneakers": ["running shoes", "high-tops", "canvas sneakers", "skate shoes", "retro sneakers"],
                    "dress shoes": ["oxfords", "loafers", "derby shoes", "brogues", "monk straps"],
                    "boots": ["Chelsea boots", "work boots", "hiking boots", "chukka boots", "combat boots"],
                    "sandals": ["flip-flops", "slides", "sport sandals", "leather sandals", "fisherman sandals"],
                    "slippers": ["moccasin slippers", "bootie slippers", "slide slippers", "open-heel slippers", "house shoes"]
                },
                "watch": {
                    "analog watch": ["dress watch", "field watch", "pilot watch", "minimalist watch", "automatic watch"],
                    "digital watch": ["LED watch", "sports digital watch", "calculator watch", "digital chronograph", "stopwatch"],
                    "smartwatch": ["Apple Watch", "Samsung Galaxy Watch", "Garmin", "Fitbit Sense", "Amazfit"],
                    "chronograph": ["tachymeter watch", "racing watch", "aviation chronograph", "diving chronograph", "sport chronograph"],
                    "fashion watch": ["designer watch", "minimalist fashion watch", "leather strap watch", "mesh band watch", "statement watch"]
                },
                "bag": {
                    "backpack": ["rucksack", "laptop backpack", "daypack", "leather backpack", "canvas backpack"],
                    "messenger bag": ["crossbody bag", "satchel", "laptop bag", "canvas messenger", "leather satchel"],
                    "briefcase": ["hard-shell briefcase", "soft briefcase", "attache case", "portfolio bag", "document case"],
                    "duffel bag": ["gym bag", "weekender bag", "travel duffel", "roll-top duffel", "wheeled duffel"],
                    "tote bag": ["canvas tote", "leather tote", "work tote", "zip-top tote", "reusable shopping bag"]
                }
            },
            "women": {
                "top": {
                    "t-shirt": ["v-neck tee", "graphic tee", "boyfriend tee", "crop top", "longline tee"],
                    "blouse": ["silk blouse", "chiffon blouse", "wrap top", "peplum top", "tunic top"],
                    "sweater": ["cardigan", "pullover sweater", "turtleneck", "cowl neck sweater", "crop sweater"],
                    "crop top": ["halter neck", "tie-front top", "off-the-shoulder crop", "long-sleeve crop", "sports bra top"],
                    "camisole": ["silk cami", "lace cami", "spaghetti strap top", "built-in-bra cami", "layering tank"]
                },
                "bottom": {
                    "jeans": ["skinny jeans", "high-waisted jeans", "straight-leg jeans", "bootcut jeans", "flare jeans"],
                    "trousers": ["wide-leg pants", "palazzo pants", "chinos", "dress pants", "culottes"],
                    "skirt": ["mini skirt", "midi skirt", "maxi skirt", "pencil skirt", "A-line skirt"],
                    "shorts": ["denim shorts", "high-waisted shorts", "paperbag shorts", "biker shorts", "skort"],
                    "leggings": ["full-length leggings", "capri leggings", "faux leather leggings", "printed leggings", "yoga pants"]
                },
                "hat": {
                    "cap": ["baseball cap", "visor", "bucket hat", "dad hat", "trucker hat"],
                    "beanie": ["slouchy beanie", "cuffed beanie", "pom-pom beanie", "fashion beanie", "headband beanie"],
                    "sun hat": ["wide-brim hat", "floppy hat", "straw hat", "fedora", "panama hat"],
                    "fascinator": ["hair fascinator", "cocktail hat", "feather headpiece", "bridal fascinator", "sinamay hat"],
                    "headband": ["knotted headband", "padded headband", "jeweled headband", "elastic headband", "sport headband"]
                },
                "shoes": {
                    "sneakers": ["platform sneakers", "slip-on sneakers", "fashion sneakers", "athletic shoes", "high-tops"],
                    "heels": ["stilettos", "pumps", "block heels", "kitten heels", "wedges"],
                    "flats": ["ballet flats", "pointed-toe flats", "loafers", "mules", "espadrilles"],
                    "boots": ["ankle boots", "knee-high boots", "over-the-knee boots", "heeled boots", "combat boots"],
                    "sandals": ["strappy sandals", "gladiator sandals", "slide sandals", "wedge sandals", "flat sandals"]
                },
                "watch": {
                    "analog watch": ["dress watch", "bracelet watch", "minimalist watch", "rose gold watch", "two-tone watch"],
                    "digital watch": ["sports watch", "LED watch", "digital fitness watch", "touchscreen watch", "activity tracker"],
                    "smartwatch": ["Apple Watch", "Samsung Galaxy Watch", "Garmin Vivomove", "Fitbit Versa", "Fossil Gen 6"],
                    "bracelet watch": ["bangle watch", "chain link watch", "cuff watch", "charm watch", "wrap watch"],
                    "fashion watch": ["designer watch", "crystal watch", "beaded watch", "interchangeable strap watch", "colorful watch"]
                },
                "bag": {
                    "handbag": ["shoulder bag", "hobo bag", "satchel", "top-handle bag", "structured bag"],
                    "tote bag": ["work tote", "canvas tote", "leather tote", "zippered tote", "beach tote"],
                    "clutch": ["evening clutch", "envelope clutch", "wristlet", "box clutch", "minaudière"],
                    "backpack": ["leather backpack", "drawstring backpack", "mini backpack", "rucksack", "laptop backpack"],
                    "crossbody bag": ["messenger bag", "belt bag", "chest pack", "phone purse", "sling bag"]
                }
            }
        }
        
        print(f"[Query Parser] Initialized with model: {model_id}")
    
    def parse_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Parse natural language query using Hugging Face Router API
        
        Args:
            query: Natural language query from user
            context: Optional context from previous interactions
            
        Returns:
            Structured search request with original and refined query
        """
        # Check cache first
        cache_key = f"{query}_{json.dumps(context) if context else ''}"
        if cache_key in self.query_cache:
            print("[Query Parser] Using cached result")
            return self.query_cache[cache_key]
        
        print(f"[Query Parser] Parsing query: {query}")
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, context)
        
        try:
            # Generate response using OpenAI-compatible API
            response = self._generate_response(system_prompt, user_prompt)
            
            # Parse response
            parsed = self._parse_response(response)
            
            # Validate and clean up
            parsed = self._validate_and_cleanup(parsed, query)
            
            # Cache result
            self.query_cache[cache_key] = parsed
            
            print(f"[Query Parser] Parsed successfully: {parsed}")
            return parsed
            
        except Exception as e:
            logger.error(f"[Query Parser] Qwen API parsing failed: {e}")
            return self._fallback_parsing(query)

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the model with collection query support"""
        return """You are a search query parser for a fashion e-commerce store. Your task is to convert natural language queries into structured search requests.

Instructions:
1. Analyze the user's query carefully
2. Extract the ORIGINAL query as-is
3. Create a REFINED query that removes excluded/negated terms and keeps only what the user WANTS
4. Extract filters including exclusions (brands, colors, etc.)
5. Identify the ITEM TYPE from: top, bottom, hat, shoes, watch, bag (can be multiple)
6. For gender filters: If "men" or "male" is detected, automatically exclude "women" gender. If "women" or "female" is detected, automatically exclude "men" gender
7. Determine if this is a COLLECTION query:
   - If after item type identification, the number of UNIQUE item types > 1, set is_collection_query: true
   - Examples: 
       "shirt and shoes" → type: ["top", "shoes"], is_collection_query: true
       "jeans and t-shirt" → type: ["bottom", "top"], is_collection_query: true
       "sneakers" → type: ["shoes"], is_collection_query: false
   - If user explicitly asks for a collection/outfit (e.g., "suggest me a collection", "complete my collection")
   - If user asks for items to match something they already have (e.g., "clothes for my Nike shoes")
   - If user asks for items for a specific occasion (e.g., "outfit for a party")
8. If collection query, identify any existing items the user has (e.g., "for my Nike shoes")
9. For collection queries, include ALL relevant item types in the "type" array
10. Respond with ONLY a valid JSON object
11. Be precise and accurate in your parsing

Respond with a JSON object in this exact format:
{
    "original_query": "the exact original query text",
    "refined_query": "cleaned query with only positive intent, no excluded words",
    "filters": {
        "category": ["array of categories or empty"],
        "brands": ["array of brands to INCLUDE or empty"],
        "exclude_brands": ["array of brands to EXCLUDE or empty"],
        "colors": ["array of colors to INCLUDE or empty"],
        "exclude_colors": ["array of colors to EXCLUDE or empty"],
        "price_range": [min_price, max_price] or null,
        "gender": "gender filter or null (use 'Men' or 'Women')",
        "exclude_gender": ["array of genders to EXCLUDE or empty (use 'Men' or 'Women')"],
        "type": ["array of item types: 'top', 'bottom', 'hat', 'shoes', 'watch', 'bag' or empty"]
    },
    "top_k": 10,
    "is_collection_query": true/false,
    "existing_items": {
        "type": ["array of item types the user already has"],
        "brands": ["array of brands the user already has"]
    },
    "explanation": "brief explanation of how you interpreted the query"
}

Item Type Detection Rules:
- "top": t-shirt, shirt, blouse, hoodie, sweater, jacket, blazer, coat, cardigan, etc.
- "bottom": jeans, pants, trousers, shorts, skirt, leggings, etc.
- "hat": cap, beanie, fedora, hat, etc.
- "shoes": sneakers, boots, sandals, heels, flats, etc.
- "watch": watch, smartwatch, timepiece, etc.
- "bag": bag, backpack, purse, tote, clutch, handbag, etc.

Collection Query Detection Rules:
- Multiple item types mentioned: "shirt and pants" -> is_collection_query: true, type: ["top", "bottom"]
- Explicit collection request: "suggest me a collection" -> is_collection_query: true, type: ["top", "bottom", "hat", "shoes", "watch", "bag"]
- Matching existing items: "clothes for my Nike shoes" -> is_collection_query: true, type: ["top", "bottom", "hat", "bag", "watch"], existing_items: {type: ["shoes"], brands: ["Nike"]}
- Occasion-based outfit: "outfit for a party" -> is_collection_query: true, type: ["top", "bottom", "hat", "shoes", "watch", "bag"]
- Single item type: "jeans for my Nike shoes" -> is_collection_query: false, type: ["bottom"], existing_items: {type: ["shoes"], brands: ["Nike"]}

Examples:
- Query: "red sneakers but not nike" -> refined_query: "red sneakers", type: ["shoes"], exclude_brands: ["Nike"], is_collection_query: false
- Query: "suggest me a Nike collection" -> refined_query: "Nike collection", brands: ["Nike"], is_collection_query: true, type: ["top", "bottom", "hat", "shoes", "watch", "bag"]
- Query: "jeans for my Nike shoes" -> refined_query: "jeans", type: ["bottom"], existing_items: {type: ["shoes"], brands: ["Nike"]}, is_collection_query: false
- Query: "shirt and pant for my sneakers" -> refined_query: "shirt pant", type: ["top", "bottom"], existing_items: {type: ["shoes"]}, is_collection_query: true
- Query: "men sneakers not black" -> refined_query: "sneakers", type: ["shoes"], gender: "Men", exclude_gender: ["Women"], exclude_colors: ["black"], is_collection_query: false
- Query: "complete my collection with a watch" -> refined_query: "watch", type: ["watch"], is_collection_query: true
- Query: "suggest me the best outfits for a party" -> refined_query: "party outfits", is_collection_query: true, type: ["top", "bottom", "hat", "shoes", "watch", "bag"]
- Query: "suggest me the best clothes for my Nike shoes" -> refined_query: "clothes", is_collection_query: true, type: ["top", "bottom", "hat", "bag", "watch"], existing_items: {type: ["shoes"], brands: ["Nike"]}
- Query: "suggest me a collection with no blue color" -> refined_query: "collection", is_collection_query: true, type: ["top", "bottom", "hat", "shoes", "watch", "bag"], exclude_colors: ["blue"]
- Query: "complete outfit under 5000" -> refined_query: "outfit", is_collection_query: true, type: ["top", "bottom", "hat", "shoes", "watch", "bag"], price_range: [0, 5000]"""
    
    def _create_user_prompt(self, query: str, context: Optional[Dict]) -> str:
        """Create the user prompt with the query"""
        context_str = json.dumps(context) if context else "None"
        return f"""User Query: "{query}"

Context: {context_str}

Parse this query and respond with the JSON structure."""
    
    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Hugging Face Router API"""
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return completion.choices[0].message.content
    
    def _parse_response(self, response: str) -> Dict:
        """Parse the model's response as JSON"""
        try:
            # Find JSON in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except json.JSONDecodeError as e:
            logger.error(f"[Query Parser] JSON parsing failed: {e}")
            logger.error(f"[Query Parser] Response was: {response}")
            raise ValueError("Invalid JSON response")

    def _validate_and_cleanup(self, parsed: Dict, original_query: str) -> Dict:
        """Validate and clean up the parsed response"""
        # Ensure original_query exists
        if "original_query" not in parsed:
            parsed["original_query"] = original_query
        
        # Ensure refined_query exists
        if "refined_query" not in parsed:
            parsed["refined_query"] = original_query
        
        if "filters" not in parsed:
            parsed["filters"] = {}
        
        if "top_k" not in parsed:
            parsed["top_k"] = 10
        
        # Add collection query fields
        if "is_collection_query" not in parsed:
            parsed["is_collection_query"] = False
        
        if "existing_items" not in parsed:
            parsed["existing_items"] = {"type": [], "brands": []}
        
        # Clean up empty filter arrays and None values
        filters = parsed["filters"]
        cleaned_filters = {}
        
        for key, value in filters.items():
            # Skip None values
            if value is None:
                continue
            
            # Skip empty lists
            if isinstance(value, list):
                # Filter out None values from lists
                cleaned_list = [item for item in value if item is not None and item != ""]
                if len(cleaned_list) > 0:
                    cleaned_filters[key] = cleaned_list
            # Keep non-empty strings
            elif isinstance(value, str) and value.strip():
                cleaned_filters[key] = value.strip()
            # Keep valid price ranges
            elif key == "price_range" and isinstance(value, list) and len(value) == 2:
                if all(v is not None and isinstance(v, (int, float)) for v in value):
                    cleaned_filters[key] = value
        
        # Auto-add gender exclusions if gender filter is present
        if "gender" in cleaned_filters and "exclude_gender" not in cleaned_filters:
            gender = cleaned_filters["gender"].lower()
            if "men" in gender or "male" in gender:
                cleaned_filters["exclude_gender"] = ["Women"]
            elif "women" in gender or "female" in gender:
                cleaned_filters["exclude_gender"] = ["Men"]
        
        parsed["filters"] = cleaned_filters
        return parsed
    
    def _fallback_parsing(self, query: str) -> Dict:
        """Simple fallback parsing if the model fails"""
        logger.warning("[Query Parser] Using fallback parsing")
        return {
            "original_query": query,
            "refined_query": query,
            "filters": {},
            "top_k": 10,
            "is_collection_query": False,
            "existing_items": {"type": [], "brands": []},
            "explanation": "Fallback parsing - using original query"
        }
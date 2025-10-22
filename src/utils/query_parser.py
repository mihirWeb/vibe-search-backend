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
        """Create the system prompt for the model"""
        return """You are a search query parser for a fashion e-commerce store. Your task is to convert natural language queries into structured search requests.

Instructions:
1. Analyze the user's query carefully
2. Extract the ORIGINAL query as-is
3. Create a REFINED query that removes excluded/negated terms and keeps only what the user WANTS
4. Extract filters including exclusions (brands, colors, etc.)
5. Respond with ONLY a valid JSON object
6. Be precise and accurate in your parsing

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
        "gender": "gender filter or null"
    },
    "top_k": 10,
    "explanation": "brief explanation of how you interpreted the query"
}

Examples:
- Query: "red sneakers but not nike" -> refined_query: "red sneakers", exclude_brands: ["Nike"]
- Query: "black shirt without stripes" -> refined_query: "black shirt", exclude_colors: []
- Query: "shoes under $100 not adidas" -> refined_query: "shoes", price_range: [0, 100], exclude_brands: ["Adidas"]"""
    
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
            "explanation": "Fallback parsing - using original query"
        }
"""
Configuration management for the RAG application.
Loads settings from .env file and provides centralized configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Centralized configuration for the application."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    STORAGE_DIR = BASE_DIR / "storage"
    UPLOADS_DIR = STORAGE_DIR / "uploads"
    VECTORS_DIR = STORAGE_DIR / "vectors"
    METADATA_FILE = STORAGE_DIR / "metadata.json"
    
    # Ensure directories exist
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load environment variables
    load_dotenv(BASE_DIR / ".env")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Model Configuration (Cost Optimization)
    SMALL_MODEL = "gpt-4o-mini"  # For retrieval, verification
    LARGE_MODEL = "gpt-4o"       # For answer generation
    EMBEDDING_MODEL = "text-embedding-3-small"
    VISION_MODEL = "gpt-4o-mini"
    
    # RAG Configuration
    CHUNK_SIZE = 400  # tokens
    CHUNK_OVERLAP = 50  # tokens
    TOP_K_RETRIEVAL = 5  # max chunks to retrieve
    
    # Memory Configuration
    CONVERSATION_WINDOW = 3  # last N turns
    MAX_CONTEXT_TOKENS = 8000  # safety limit
    
    # Supported file types
    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "text",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image"
    }
    
    # Server Configuration
    HOST = "0.0.0.0"
    PORT = 7860  # Hugging Face Spaces default
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return True
    
    @classmethod
    def get_openai_client(cls):
        """Get configured OpenAI client."""
        from openai import OpenAI
        return OpenAI(api_key=cls.OPENAI_API_KEY)

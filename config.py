import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the PDF Chat application."""
    
    def __init__(self):
        # Google Gemini API
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Gemini Model Configuration
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        
        # Qdrant Cloud Configuration (required for Streamlit Cloud)
        self.QDRANT_HOST = os.getenv("QDRANT_HOST")
        if not self.QDRANT_HOST:
            raise ValueError("QDRANT_HOST is required for Streamlit Cloud deployment")
        
        self.QDRANT_PORT = int(os.getenv("QDRANT_PORT", "443"))
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        if not self.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY is required for Streamlit Cloud deployment")
        
        self.QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "pdf_documents")
        
        # Embedding model
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Other settings
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
        self.OCR_CONFIG = os.getenv("OCR_CONFIG", "--psm 6")
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024
        self.ALLOWED_EXTENSIONS = {".pdf"}
    
    @property
    def is_qdrant_cloud(self) -> bool:
        """Check if using Qdrant Cloud."""
        return self.QDRANT_HOST.startswith('https://')
    
    def get_qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        if self.is_qdrant_cloud:
            return self.QDRANT_HOST
        else:
            return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
    
    def validate(self) -> bool:
        """Validate configuration."""
        errors = []
        
        if not self.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required")
        
        if not self.QDRANT_HOST:
            errors.append("QDRANT_HOST is required")
        
        if not self.QDRANT_API_KEY:
            errors.append("QDRANT_API_KEY is required")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True 
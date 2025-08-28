import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the PDF Chat application."""
    
    # Google Gemini API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Qdrant Cloud Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # New: API key for cloud
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "pdf_documents")
    
    # Check if using Qdrant Cloud
    @property
    def is_qdrant_cloud(self):
        return self.QDRANT_HOST.startswith('https://')
    
    # Embedding model (keep sentence-transformers for now)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Other settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    OCR_LANGUAGE = "eng"
    OCR_CONFIG = "--psm 6"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.pdf'} 
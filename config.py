import os
import streamlit as st

# Only load dotenv if it's available (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available in Streamlit Cloud
    pass

def get_config():
    """Get configuration values - this function is called only when needed."""
    
    def _get_config(key: str, default: str = None) -> str:
        """Get configuration value from Streamlit secrets or environment variables."""
        try:
            # First try to get from Streamlit secrets (for Streamlit Cloud)
            if hasattr(st, 'secrets') and st.secrets and key in st.secrets:
                print(f"Found {key} in st.secrets: {st.secrets[key][:10] if st.secrets[key] else 'None'}...")
                return st.secrets[key]
            
            # Fall back to environment variables
            env_value = os.getenv(key, default)
            print(f"Using environment variable for {key}: {env_value[:10] if env_value else 'None'}...")
            return env_value
        except Exception as e:
            print(f"Error getting config for {key}: {str(e)}")
            # If st.secrets is not available, fall back to environment variables
            return os.getenv(key, default)
    
    # Google Gemini API
    GOOGLE_API_KEY = _get_config("GOOGLE_API_KEY")
    
    # Gemini Model Configuration
    GEMINI_MODEL = _get_config("GEMINI_MODEL", "gemini-2.0-flash")
    
    # Qdrant Cloud Configuration (required for Streamlit Cloud)
    QDRANT_HOST = _get_config("QDRANT_HOST")
    if not QDRANT_HOST:
        # Add debugging information
        print(f"Debug: QDRANT_HOST not found. Available secrets: {list(st.secrets.keys()) if hasattr(st, 'secrets') and st.secrets else 'No secrets'}")
        print(f"Debug: Environment variables: QDRANT_HOST={os.getenv('QDRANT_HOST')}")
        raise ValueError("QDRANT_HOST is required for Streamlit Cloud deployment")
    
    QDRANT_PORT = int(_get_config("QDRANT_PORT", "443"))
    QDRANT_API_KEY = _get_config("QDRANT_API_KEY")
    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is required for Streamlit Cloud deployment")
    
    QDRANT_COLLECTION_NAME = _get_config("QDRANT_COLLECTION_NAME", "pdf_documents")
    
    # Embedding model
    EMBEDDING_MODEL = _get_config("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Other settings
    CHUNK_SIZE = int(_get_config("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(_get_config("CHUNK_OVERLAP", "200"))
    OCR_LANGUAGE = _get_config("OCR_LANGUAGE", "eng")
    OCR_CONFIG = _get_config("OCR_CONFIG", "--psm 6")
    MAX_TOKENS = int(_get_config("MAX_TOKENS", "2048"))
    TEMPERATURE = float(_get_config("TEMPERATURE", "0.7"))
    MAX_FILE_SIZE = int(_get_config("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024
    ALLOWED_EXTENSIONS = {".pdf"}
    
    # Return a simple object with all the config values
    class Config:
        pass
    
    config = Config()
    config.GOOGLE_API_KEY = GOOGLE_API_KEY
    config.GEMINI_MODEL = GEMINI_MODEL
    config.QDRANT_HOST = QDRANT_HOST
    config.QDRANT_PORT = QDRANT_PORT
    config.QDRANT_API_KEY = QDRANT_API_KEY
    config.QDRANT_COLLECTION_NAME = QDRANT_COLLECTION_NAME
    config.EMBEDDING_MODEL = EMBEDDING_MODEL
    config.CHUNK_SIZE = CHUNK_SIZE
    config.CHUNK_OVERLAP = CHUNK_OVERLAP
    config.OCR_LANGUAGE = OCR_LANGUAGE
    config.OCR_CONFIG = OCR_CONFIG
    config.MAX_TOKENS = MAX_TOKENS
    config.TEMPERATURE = TEMPERATURE
    config.MAX_FILE_SIZE = MAX_FILE_SIZE
    config.ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
    
    # Add properties
    @property
    def is_qdrant_cloud(self):
        """Check if using Qdrant Cloud."""
        return self.QDRANT_HOST.startswith('https://')
    
    @property
    def get_qdrant_url(self):
        """Get Qdrant connection URL."""
        if self.is_qdrant_cloud:
            return self.QDRANT_HOST
        else:
            return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
    
    config.is_qdrant_cloud = property(is_qdrant_cloud)
    config.get_qdrant_url = property(get_qdrant_url)
    
    return config

class Config:
    """Configuration class for the PDF Chat application."""
    
    def __init__(self):
        # Get config from the function
        config = get_config()
        # Copy all attributes
        for attr in dir(config):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(config, attr))
    
    def _get_config(self, key: str, default: str = None) -> str:
        """Get configuration value from Streamlit secrets or environment variables."""
        try:
            # First try to get from Streamlit secrets (for Streamlit Cloud)
            if hasattr(st, 'secrets') and st.secrets and key in st.secrets:
                print(f"Found {key} in st.secrets: {st.secrets[key][:10] if st.secrets[key] else 'None'}...")
                return st.secrets[key]
            
            # Fall back to environment variables
            env_value = os.getenv(key, default)
            print(f"Using environment variable for {key}: {env_value[:10] if env_value else 'None'}...")
            return env_value
        except Exception as e:
            print(f"Error getting config for {key}: {str(e)}")
            # If st.secrets is not available, fall back to environment variables
            return os.getenv(key, default)
    
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
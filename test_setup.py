#!/usr/bin/env python3
"""
Test script to verify the Chat with PDF RAG system setup.
Run this script to check if all components are properly configured.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        import streamlit
        logger.info("✅ Streamlit imported successfully")
    except ImportError as e:
        logger.error(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import qdrant_client
        logger.info("✅ Qdrant client imported successfully")
    except ImportError as e:
        logger.error(f"❌ Qdrant client import failed: {e}")
        return False
    
    try:
        import langchain
        logger.info("✅ LangChain imported successfully")
    except ImportError as e:
        logger.error(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        logger.info("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        logger.error(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import pytesseract
        logger.info("✅ PyTesseract imported successfully")
    except ImportError as e:
        logger.error(f"❌ PyTesseract import failed: {e}")
        return False
    
    try:
        import fitz  # PyMuPDF
        logger.info("✅ PyMuPDF imported successfully")
    except ImportError as e:
        logger.error(f"❌ PyMuPDF import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from config import Config
        config = Config()
        
        if config.GOOGLE_API_KEY:
            logger.info("✅ Google API key found")
        else:
            logger.warning("⚠️ Google API key not found - set GOOGLE_API_KEY in .env file")
        
        logger.info(f"✅ Configuration loaded: Qdrant {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_qdrant_connection():
    """Test Qdrant connection."""
    logger.info("Testing Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host="localhost", port=6333)
        
        # Try to get collections
        collections = client.get_collections()
        logger.info(f"✅ Qdrant connection successful - {len(collections.collections)} collections found")
        return True
        
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        logger.warning("⚠️ Make sure Qdrant is running on localhost:6333")
        logger.info("💡 Start Qdrant with: docker-compose up -d")
        return False

def test_ocr_setup():
    """Test OCR setup."""
    logger.info("Testing OCR setup...")
    
    try:
        import pytesseract
        
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        logger.info(f"✅ Tesseract OCR found: version {version}")
        return True
        
    except Exception as e:
        logger.error(f"❌ OCR setup failed: {e}")
        logger.warning("⚠️ Tesseract OCR not properly configured")
        logger.info("💡 Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("💡 Linux/Mac: Use package manager (apt, brew, yum)")
        return False

def test_custom_modules():
    """Test custom module imports."""
    logger.info("Testing custom modules...")
    
    try:
        from pdf_processor import PDFProcessor
        logger.info("✅ PDF Processor imported successfully")
    except Exception as e:
        logger.error(f"❌ PDF Processor import failed: {e}")
        return False
    
    try:
        from vector_store import VectorStore
        logger.info("✅ Vector Store imported successfully")
    except Exception as e:
        logger.error(f"❌ Vector Store import failed: {e}")
        return False
    
    try:
        from rag_engine import RAGEngine
        logger.info("✅ RAG Engine imported successfully")
    except Exception as e:
        logger.error(f"❌ RAG Engine import failed: {e}")
        return False
    
    try:
        from document_manager import DocumentManager
        logger.info("✅ Document Manager imported successfully")
    except Exception as e:
        logger.error(f"❌ Document Manager import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if all required files exist."""
    logger.info("Testing file structure...")
    
    required_files = [
        "app.py",
        "config.py",
        "pdf_processor.py",
        "vector_store.py",
        "rag_engine.py",
        "document_manager.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        logger.info("✅ All required files found")
        return True

def main():
    """Run all tests."""
    logger.info("🚀 Starting Chat with PDF RAG System setup test...")
    logger.info("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Custom Modules", test_custom_modules),
        ("Qdrant Connection", test_qdrant_connection),
        ("OCR Setup", test_ocr_setup),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your setup is ready.")
        logger.info("💡 Run the application with: streamlit run app.py")
    else:
        logger.warning("⚠️ Some tests failed. Please fix the issues above.")
        logger.info("💡 Check the README.md for troubleshooting tips.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
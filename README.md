# 📚 Chat with PDF - RAG System

A powerful AI-powered PDF chat application that uses Retrieval-Augmented Generation (RAG) to provide intelligent conversations with your documents. Built with LangChain, Google Gemini, Qdrant, and modern AI technologies.

## ✨ Features

- **🔍 Smart PDF Processing**: Automatic text extraction with OCR fallback for image-based PDFs
- **🚀 Fast Embeddings**: Efficient vector embeddings using Sentence Transformers
- **💬 AI Chat Interface**: Natural language conversations with your documents
- **📊 Multiple Interaction Modes**: RAG queries, document summaries, and analysis
- **🔄 Document Management**: Upload, reprocess, and manage multiple PDFs
- **📱 Beautiful UI**: Modern Streamlit interface with responsive design
- **⚡ Performance Optimized**: Handles large PDF files efficiently
- **🔒 Source Attribution**: See exactly which parts of your document were used

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Document Manager│    │   PDF Processor │
│                 │◄──►│                 │◄──►│                 │
│  - Chat Interface│    │  - File Upload   │    │  - Text Extract │
│  - Dashboard    │    │  - Storage       │    │  - OCR Support  │
│  - Document Mgmt│    │  - Metadata      │    │  - Chunking     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RAG Engine    │    │  Vector Store    │    │  Google Gemini  │
│                 │◄──►│                 │    │                 │
│  - Query       │    │  - Qdrant DB     │    │  - LLM Model    │
│  - Context     │    │  - Embeddings    │    │  - Generation   │
│  - Generation  │    │  - Search        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Qdrant vector database (local or cloud)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd chat-with-document
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=pdf_documents
```

### 4. Start Qdrant (Local)

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or download from https://qdrant.tech/documentation/quick-start/
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📋 Installation Details

### System Requirements

- **Windows**: Tesseract OCR installation required
- **Linux/Mac**: Tesseract available via package managers
- **Memory**: Minimum 4GB RAM, 8GB+ recommended for large PDFs
- **Storage**: Depends on PDF collection size

### OCR Setup (Windows)

1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Add to PATH or update the path in `pdf_processor.py`

### OCR Setup (Linux/Mac)

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# CentOS/RHEL
sudo yum install tesseract
```

## 🎯 Usage Guide

### 1. Upload Documents

1. Click "Upload Document" in the sidebar
2. Select a PDF file (max 50MB)
3. Click "Process Document"
4. Wait for processing to complete

### 2. Start Chatting

1. Select a document from the sidebar
2. Choose your interaction mode:
   - **RAG Query**: Ask specific questions
   - **Document Summary**: Get comprehensive overview
   - **Document Analysis**: Deep analysis on topics
3. Type your question and click "Send"

### 3. Explore Features

- **Dashboard**: View system statistics and performance metrics
- **Document Management**: Reprocess, delete, or manage documents
- **Chat History**: Review your conversation history
- **Source Attribution**: See which document parts were used

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `QDRANT_HOST` | Qdrant server host | localhost |
| `QDRANT_PORT` | Qdrant server port | 6333 |
| `QDRANT_COLLECTION_NAME` | Vector collection name | pdf_documents |

### Application Settings

Edit `config.py` to customize:

- Chunk size and overlap
- OCR language and settings
- Model parameters
- File size limits

## 📊 Performance Features

### Fast Embeddings

- Uses Sentence Transformers for efficient embedding generation
- Batch processing for multiple chunks
- Optimized for large PDF files

### Smart Chunking

- Sentence-aware text splitting
- Configurable chunk sizes
- Overlap preservation for context continuity

### OCR Optimization

- Automatic fallback from text extraction to OCR
- Image preprocessing for better accuracy
- Configurable OCR parameters

## 🛠️ Development

### Project Structure

```
chat-with-document/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration management
├── pdf_processor.py      # PDF processing and OCR
├── vector_store.py       # Qdrant vector operations
├── rag_engine.py         # RAG logic and LLM integration
├── document_manager.py   # Document lifecycle management
├── requirements.txt      # Python dependencies
├── env_example.txt       # Environment variables template
├── README.md            # This file
└── uploads/             # Document storage directory
```

### Adding New Features

1. **New Processing Methods**: Extend `PDFProcessor` class
2. **Additional Vector Operations**: Add methods to `VectorStore` class
3. **New Chat Modes**: Extend `RAGEngine` class
4. **UI Components**: Modify `app.py` and add CSS styles

### Testing

```bash
# Run basic tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=. tests/
```

## 🔍 Troubleshooting

### Common Issues

1. **OCR Not Working**
   - Verify Tesseract installation
   - Check PATH configuration
   - Update OCR path in `pdf_processor.py`

2. **Qdrant Connection Error**
   - Ensure Qdrant is running
   - Check host/port configuration
   - Verify network connectivity

3. **Google API Errors**
   - Validate API key in `.env` file
   - Check API quota and billing
   - Verify internet connectivity

4. **Large PDF Processing**
   - Increase chunk size in config
   - Monitor memory usage
   - Consider splitting very large documents

### Performance Tips

- Use SSD storage for better I/O performance
- Increase chunk size for faster processing
- Monitor memory usage with large documents
- Use appropriate OCR settings for your document types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: RAG framework and LLM integration
- **Google Gemini**: Advanced language model capabilities
- **Qdrant**: High-performance vector database
- **Sentence Transformers**: Efficient text embeddings
- **Streamlit**: Beautiful web application framework
- **Tesseract**: OCR engine for image-based PDFs

## 📞 Support

- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check the code comments and docstrings

---

**Happy Document Chatting! 🚀📚** 
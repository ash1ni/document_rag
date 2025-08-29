import streamlit as st
import os
import tempfile
import logging
from datetime import datetime
import json
from typing import Dict, Any, List
import traceback
import time # Added for manual refresh

# Remove these module-level imports to avoid circular dependency issues
# from document_manager import DocumentManager
# from rag_engine import RAGEngine
# from config import Config

# At the top of the file, after imports
# Global document manager instance - initialize only when needed
if 'global_document_manager' not in st.session_state:
    st.session_state.global_document_manager = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="UDISE Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #28a745;
    }
    .source-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #dee2e6;
    }
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_document_id' not in st.session_state:
    st.session_state.current_document_id = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
# Add this: Store documents in session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}

def get_document_manager():
    """Get or create the document manager instance with lazy import."""
    if st.session_state.global_document_manager is None:
        try:
            # Import here to avoid circular dependency issues
            from document_manager import DocumentManager
            st.session_state.global_document_manager = DocumentManager()
        except Exception as e:
            st.error(f"Failed to initialize document manager: {str(e)}")
            st.stop()
    return st.session_state.global_document_manager

def get_rag_engine():
    """Get or create the RAG engine instance with lazy import."""
    if 'rag_engine' not in st.session_state:
        try:
            # Import here to avoid circular dependency issues
            from rag_engine import RAGEngine
            st.session_state.rag_engine = RAGEngine()
        except Exception as e:
            st.error(f"Failed to initialize RAG engine: {str(e)}")
            st.stop()
    return st.session_state.rag_engine

def initialize_components():
    """Initialize the main components of the application."""
    try:
        # Check if Google API key is set
        from config import Config
        if not Config().GOOGLE_API_KEY:
            st.error("⚠️ Google API key not found! Please set GOOGLE_API_KEY in your .env file.")
            st.stop()
        
        # Use global document manager
        document_manager = get_document_manager()
        
        rag_engine = get_rag_engine()
        
        return document_manager, rag_engine
        
    except Exception as e:
        st.error(f"❌ Error initializing application: {str(e)}")
        st.error("Please check your configuration and try again.")
        st.stop()

def display_header():
    """Display the main header of the application."""
    st.markdown("""
    <div class="main-header">
        <h1>UDISE Document Chat</h1>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar(document_manager):
    """Display the sidebar with document management options."""
    st.sidebar.markdown("## 📁 Document Management")
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Documents"):
        st.rerun()
    
    # Document upload section
    with st.sidebar.expander("📤 Upload Document", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file to start chatting with it"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Process document
                        result = document_manager.upload_document(tmp_file_path, uploaded_file.name)
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        if result['status'] == 'success':
                            st.success(f"✅ {result['message']}")
                            st.session_state.current_document_id = result['document_id']
                            st.session_state.documents_loaded = True
                            
                            # Store documents in session state
                            st.session_state.documents = document_manager.documents
                            
                            # Add this debug info
                            st.info(f"Debug: Document ID: {result['document_id']}")
                            st.info(f"Debug: Total chunks: {result['total_chunks']}")
                            st.info(f"Debug: Documents in session: {len(st.session_state.documents)}")
                            
                            # Force a rerun
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        logger.error(f"Error processing uploaded file: {str(e)}")
    
    # Document list section
    with st.sidebar.expander("📋 Document Library", expanded=True):
        # Debug: Show document manager state
        st.write(f"**Debug: Document Manager ID:** {id(document_manager)}")
        st.write(f"**Debug: Documents in session state:** {len(st.session_state.documents)}")
        
        documents = document_manager.list_documents()
        
        # Add debug info
        st.write(f"**Total Documents:** {len(documents)}")
        st.write(f"**Debug: Raw documents:** {documents}")
        
        if not documents:
            st.info("No documents uploaded yet. Upload a PDF to get started!")
        else:
            for doc in documents:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{doc['filename']}**")
                        st.caption(f"Pages: {doc['total_pages']} | Chunks: {doc['total_chunks']}")
                    
                    with col2:
                        if st.button("📖", key=f"select_{doc['document_id']}", help="Select this document"):
                            st.session_state.current_document_id = doc['document_id']
                            st.session_state.documents_loaded = True
                            st.rerun()
                    
                    # Document actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄", key=f"reprocess_{doc['document_id']}", help="Reprocess document"):
                            with st.spinner("Reprocessing..."):
                                result = document_manager.reprocess_document(doc['document_id'])
                                if result['status'] == 'success':
                                    st.success("✅ Reprocessed!")
                                    # Update session state
                                    st.session_state.documents = document_manager.documents
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{doc['document_id']}", help="Delete document"):
                            if st.checkbox(f"Confirm deletion of {doc['filename']}", key=f"confirm_{doc['document_id']}"):
                                result = document_manager.delete_document(doc['document_id'])
                                if result['status'] == 'success':
                                    st.success("✅ Deleted!")
                                    # Update session state
                                    st.session_state.documents = document_manager.documents
                                    if st.session_state.current_document_id == doc['document_id']:
                                        st.session_state.current_document_id = None
                                        st.session_state.documents_loaded = False
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
                    
                    st.divider()
    
    # System information
    with st.sidebar.expander("⚙️ System Info", expanded=False):
        try:
            rag_engine = get_rag_engine()
            system_info = rag_engine.get_system_info()
            
            if 'error' not in system_info:
                st.write(f"**Model:** {system_info['model']}")
                st.write(f"**Embeddings:** {system_info['embedding_model']}")
                st.write(f"**Vector Store:** {system_info['vector_store']}")
                
                if 'vector_stats' in system_info and system_info['vector_stats']:
                    stats = system_info['vector_stats']
                    st.write(f"**Total Vectors:** {stats.get('total_points', 'N/A')}")
                    st.write(f"**Vector Size:** {stats.get('vector_size', 'N/A')}")
            else:
                st.error("Could not retrieve system info")
                
        except Exception as e:
            st.error(f"Error getting system info: {str(e)}")

def display_chat_interface(rag_engine):
    """Display the main chat interface."""
    st.markdown("## 💬 Chat Interface")
    
    # Document selection info
    if st.session_state.current_document_id:
        # Use the session state document manager instead of creating a new one
        document_manager = get_document_manager()
        doc_info = document_manager.get_document_info(st.session_state.current_document_id)
        
        if doc_info:
            st.success(f"📖 Currently chatting with: **{doc_info['filename']}**")
            st.caption(f"Pages: {doc_info['total_pages']} | Chunks: {doc_info['total_chunks']} | Method: {doc_info['processing_method']}")
        else:
            st.error("Document not found!")
            st.session_state.current_document_id = None
            st.session_state.documents_loaded = False
            return
    else:
        st.info("👆 Please select a document from the sidebar to start chatting!")
        return
    
    # Chat input
    user_question = st.text_input(
        "Ask a question about your document:",
        placeholder="e.g., What are the main topics discussed?",
        key="user_question"
    )
    
    # Chat options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_mode = st.selectbox(
            "Search Mode",
            ["RAG Query", "Document Summary", "Document Analysis"],
            help="Choose how to interact with your document"
        )
    
    with col2:
        top_k = st.slider(
            "Context Chunks",
            min_value=5,  # Increased minimum
            max_value=25,  # Increased maximum
            value=15,      # Increased default
            help="Number of document chunks to use for context (higher = more comprehensive but slower)"
        )
    
    with col3:
        if st.button("🚀 Send", type="primary", disabled=not user_question):
            if user_question:
                process_user_question(user_question, search_mode, top_k, rag_engine)
    
    # Display chat history
    display_chat_history()

def process_user_question(question: str, search_mode: str, top_k: int, rag_engine):
    """Process a user question and generate a response."""
    try:
        # Add user message to chat history
        user_message = {
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat(),
            'search_mode': search_mode
        }
        st.session_state.chat_history.append(user_message)
        
        # Generate response based on search mode
        with st.spinner("🤔 Thinking..."):
            if search_mode == "RAG Query":
                response = rag_engine.query(
                    question=question,
                    document_id=st.session_state.current_document_id,
                    top_k=top_k
                )
                
                assistant_message = {
                    'role': 'assistant',
                    'content': response['answer'],
                    'timestamp': datetime.now().isoformat(),
                    'sources': response.get('sources', []),
                    'confidence': response.get('confidence', 0.0),
                    'chunks_retrieved': response.get('chunks_retrieved', 0)
                }
                
            elif search_mode == "Document Summary":
                response = rag_engine.summarize_document(st.session_state.current_document_id)
                
                assistant_message = {
                    'role': 'assistant',
                    'content': response['summary'],
                    'timestamp': datetime.now().isoformat(),
                    'summary': True,
                    'chunks_processed': response.get('chunks_processed', 0)
                }
                
            elif search_mode == "Document Analysis":
                response = rag_engine.analyze_document(
                    st.session_state.current_document_id,
                    question
                )
                
                assistant_message = {
                    'role': 'assistant',
                    'content': response['analysis'],
                    'timestamp': datetime.now().isoformat(),
                    'analysis': True,
                    'chunks_processed': response.get('chunks_processed', 0)
                }
        
        # Add assistant message to chat history
        st.session_state.chat_history.append(assistant_message)
        
        # Don't try to clear the input - let Streamlit handle it
        # The input will be cleared automatically on the next rerun
        
        # Rerun to display the new message
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")
        logger.error(f"Error processing user question: {str(e)}")
        logger.error(traceback.format_exc())

def display_chat_history():
    """Display the chat history."""
    if not st.session_state.chat_history:
        st.info("💡 Start a conversation by asking a question about your document!")
        return
    
    st.markdown("### Chat History")
    
    # Reverse the order to show recent messages first
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if message['role'] == 'user':
            # User message with better styling
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6; 
                padding: 1rem; 
                border-radius: 10px; 
                margin: 1rem 0; 
                border-left: 4px solid #667eea;
                color: #333;
            ">
                <strong style="color: #667eea;">👤 You:</strong><br>
                <span style="color: #333;">{message['content']}</span>
                <br><small style="color: #666;"><em>Mode: {message.get('search_mode', 'Unknown')}</em></small>
            </div>
            """, unsafe_allow_html=True)
            
        elif message['role'] == 'assistant':
            # Assistant message with better styling
            st.markdown(f"""
            <div style="
                background-color: #e8f4fd; 
                padding: 1rem; 
                border-radius: 10px; 
                margin: 1rem 0; 
                border-left: 4px solid #28a745;
                color: #333;
            ">
                <strong style="color: #28a745;">🤖 Assistant:</strong><br>
                <span style="color: #333;">{message['content']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Display additional information
            if 'sources' in message and message['sources']:
                with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                    for j, source in enumerate(message['sources']):
                        st.markdown(f"""
                        <div style="
                            background-color: #f8f9fa; 
                            padding: 0.5rem; 
                            border-radius: 5px; 
                            margin: 0.5rem 0; 
                            border-left: 3px solid #dee2e6;
                            color: #333;
                        ">
                            <strong style="color: #495057;">Page {source['page_number']}</strong> (via {source['method']})<br>
                            <small style="color: #6c757d;">Similarity: {source['similarity_score']:.3f}</small><br>
                            <em style="color: #495057;">{source['text_preview']}</em>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display confidence and metadata
            if 'confidence' in message:
                st.caption(f"Confidence: {message['confidence']:.2%}")
            
            if 'chunks_retrieved' in message:
                st.caption(f"Context chunks used: {message['chunks_retrieved']}")
            
            if 'chunks_processed' in message:
                st.caption(f"Document chunks processed: {message['chunks_processed']}")
            
            # Display timestamp
            timestamp = datetime.fromisoformat(message['timestamp']).strftime("%H:%M:%S")
            st.caption(f"🕐 {timestamp}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def display_dashboard(document_manager):
    """Display the dashboard with statistics and system information."""
    st.markdown("## 📊 Dashboard")
    
    try:
        # Get document statistics
        doc_stats = document_manager.get_document_stats()
        
        if 'error' not in doc_stats:
            # Create columns for stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <h3>📚 Documents</h3>
                    <h2>{doc_stats['total_documents']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <h3>📄 Pages</h3>
                    <h2>{doc_stats['total_pages']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stats-card">
                    <h3>🧩 Chunks</h3>
                    <h2>{doc_stats['total_chunks']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stats-card">
                    <h3>💾 Size</h3>
                    <h2>{doc_stats['total_size_mb']:.1f} MB</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Performance Metrics")
                st.write(f"**Average chunks per document:** {doc_stats['average_chunks_per_document']:.1f}")
                st.write(f"**Average pages per document:** {doc_stats['average_pages_per_document']:.1f}")
            
            with col2:
                st.markdown("### 🗄️ Vector Store")
                if 'vector_store_stats' in doc_stats and doc_stats['vector_store_stats']:
                    stats = doc_stats['vector_store_stats']
                    st.write(f"**Collection:** {stats.get('collection_name', 'N/A')}")
                    st.write(f"**Total vectors:** {stats.get('total_points', 'N/A')}")
                    st.write(f"**Vector size:** {stats.get('vector_size', 'N/A')}")
                    st.write(f"**Distance metric:** {stats.get('distance', 'N/A')}")
                else:
                    st.write("Vector store information not available")
        
        else:
            st.error(f"Error retrieving statistics: {doc_stats['error']}")
            
    except Exception as e:
        st.error(f"Error displaying dashboard: {str(e)}")
        logger.error(f"Error displaying dashboard: {str(e)}")

def main():
    """Main application function."""
    try:
        # Initialize components
        document_manager, rag_engine = initialize_components()
        
        # Display header
        display_header()
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Dashboard", "📁 Documents"])
        
        with tab1:
            display_chat_interface(rag_engine)
        
        with tab2:
            display_dashboard(document_manager)
        
        with tab3:
            st.markdown("## 📁 Document Management")
            st.info("Use the sidebar to manage your documents!")
            
            # Display current document info
            if st.session_state.current_document_id:
                doc_info = document_manager.get_document_info(st.session_state.current_document_id)
                if doc_info:
                    st.success(f"**Current Document:** {doc_info['filename']}")
                    
                    # Document actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Reprocess Document", type="secondary"):
                            with st.spinner("Reprocessing..."):
                                result = document_manager.reprocess_document(st.session_state.current_document_id)
                                if result['status'] == 'success':
                                    st.success("✅ Document reprocessed successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
                    
                    with col2:
                        if st.button("🗑️ Delete Document", type="secondary"):
                            if st.checkbox("Confirm deletion"):
                                result = document_manager.delete_document(st.session_state.current_document_id)
                                if result['status'] == 'success':
                                    st.success("✅ Document deleted successfully!")
                                    st.session_state.current_document_id = None
                                    st.session_state.documents_loaded = False
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
        
        # Display sidebar
        display_sidebar(document_manager)
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
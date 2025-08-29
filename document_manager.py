import os
import uuid
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
import streamlit as st
from pdf_processor import PDFProcessor
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manages document lifecycle including upload, storage, and metadata."""
    
    def __init__(self):
        # Lazy load config to avoid import issues
        from config import get_config
        self.config = get_config()
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
        # Initialize documents from session state if available
        if 'documents' not in st.session_state:
            st.session_state.documents = {}
        
        # Use session state for document storage
        self.documents = st.session_state.documents
    
    def upload_document(self, file_path: str, original_filename: str) -> dict:
        """Upload and process a document."""
        try:
            # Validate file
            if not self._validate_file(file_path, original_filename):
                return {'status': 'error', 'message': 'Invalid file'}
            
            # Generate document ID
            document_id = self._generate_document_id(file_path)
            
            # Check if document already exists
            if document_id in self.documents:
                return {'status': 'error', 'message': 'Document already exists'}
            
            # Process PDF
            logger.info(f"Processing PDF: {original_filename}")
            chunks = self.pdf_processor.process_pdf(file_path)
            
            if not chunks:
                return {'status': 'error', 'message': 'Failed to extract text from PDF'}
            
            # Store document file
            stored_path = self._store_document_file(file_path, document_id)
            
            # Add to vector store
            if not self.vector_store.add_documents(chunks, document_id):
                # Clean up if vector store fails
                self._cleanup_document(document_id)
                return {'status': 'error', 'message': 'Failed to store document in vector database'}
            
            # Create document metadata
            document_info = {
                'document_id': document_id,
                'filename': original_filename,
                'stored_path': str(stored_path),
                'upload_date': datetime.now().isoformat(),
                'file_size': os.path.getsize(file_path),
                'total_chunks': len(chunks),
                'total_pages': max(chunk['page_number'] for chunk in chunks) + 1,
                'processing_method': chunks[0].get('processing_method', 'unknown'),
                'status': 'processed'
            }
            
            # Store in session state (this persists across reruns)
            self.documents[document_id] = document_info
            st.session_state.documents = self.documents
            
            logger.info(f"Successfully processed document {original_filename} with {len(chunks)} chunks")
            
            return {
                'status': 'success',
                'message': f'Document uploaded successfully! {len(chunks)} chunks created.',
                'document_id': document_id,
                'total_chunks': len(chunks),
                'total_pages': document_info['total_pages']
            }
            
        except Exception as e:
            logger.error(f"Error uploading document {original_filename}: {str(e)}")
            return {'status': 'error', 'message': f'Upload failed: {str(e)}'}
    
    def list_documents(self) -> list:
        """List all uploaded documents."""
        try:
            # Use session state documents
            documents = list(st.session_state.documents.values())
            logger.info(f"list_documents called. Total documents in session state: {len(documents)}")
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def get_document_info(self, document_id: str) -> dict:
        """Get information about a specific document."""
        try:
            return st.session_state.documents.get(document_id, {})
        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}")
            return {}
    
    def delete_document(self, document_id: str) -> dict:
        """Delete a document and its associated data."""
        try:
            if document_id not in st.session_state.documents:
                return {'status': 'error', 'message': 'Document not found'}
            
            # Delete from vector store
            if not self.vector_store.delete_document(document_id):
                logger.warning(f"Failed to delete document {document_id} from vector store")
            
            # Delete file
            document_info = st.session_state.documents[document_id]
            stored_path = Path(document_info['stored_path'])
            if stored_path.exists():
                stored_path.unlink()
            
            # Remove from session state
            del st.session_state.documents[document_id]
            
            # Clean up empty directory
            doc_dir = stored_path.parent
            if doc_dir.exists() and not any(doc_dir.iterdir()):
                doc_dir.rmdir()
            
            logger.info(f"Successfully deleted document {document_id}")
            return {'status': 'success', 'message': 'Document deleted successfully'}
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return {'status': 'error', 'message': f'Delete failed: {str(e)}'}
    
    def reprocess_document(self, document_id: str) -> dict:
        """Reprocess a document."""
        try:
            if document_id not in st.session_state.documents:
                return {'status': 'error', 'message': 'Document not found'}
            
            document_info = st.session_state.documents[document_id]
            stored_path = document_info['stored_path']
            
            # Reprocess PDF
            chunks = self.pdf_processor.process_pdf(stored_path)
            
            if not chunks:
                return {'status': 'error', 'message': 'Failed to reprocess PDF'}
            
            # Update vector store
            if not self.vector_store.add_documents(chunks, document_id):
                return {'status': 'error', 'message': 'Failed to update vector store'}
            
            # Update metadata
            document_info['total_chunks'] = len(chunks)
            document_info['total_pages'] = max(chunk['page_number'] for chunk in chunks) + 1
            document_info['processing_method'] = chunks[0].get('processing_method', 'unknown')
            document_info['last_processed'] = datetime.now().isoformat()
            
            # Update session state
            st.session_state.documents[document_id] = document_info
            
            logger.info(f"Successfully reprocessed document {document_id}")
            return {'status': 'success', 'message': 'Document reprocessed successfully'}
            
        except Exception as e:
            logger.error(f"Error reprocessing document {document_id}: {str(e)}")
            return {'status': 'error', 'message': f'Reprocessing failed: {str(e)}'}
    
    def get_document_stats(self) -> dict:
        """Get statistics about all documents."""
        try:
            documents = list(st.session_state.documents.values())
            
            if not documents:
                return {
                    'total_documents': 0,
                    'total_pages': 0,
                    'total_chunks': 0,
                    'total_size_mb': 0.0,
                    'average_chunks_per_document': 0.0,
                    'average_pages_per_document': 0.0
                }
            
            total_documents = len(documents)
            total_pages = sum(doc.get('total_pages', 0) for doc in documents)
            total_chunks = sum(doc.get('total_chunks', 0) for doc in documents)
            total_size_mb = sum(doc.get('file_size', 0) for doc in documents) / (1024 * 1024)
            
            return {
                'total_documents': total_documents,
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'total_size_mb': total_size_mb,
                'average_chunks_per_document': total_chunks / total_documents,
                'average_pages_per_document': total_pages / total_documents,
                'vector_store_stats': self.vector_store.get_collection_stats()
            }
            
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {'error': str(e)}
    
    def _validate_file(self, file_path: str, filename: str) -> bool:
        """Validate uploaded file."""
        try:
            # Check file extension
            if not filename.lower().endswith('.pdf'):
                return False
            
            # Check file size
            if os.path.getsize(file_path) > self.config.MAX_FILE_SIZE:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {filename}: {str(e)}")
            return False
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Error generating document ID: {str(e)}")
            return str(uuid.uuid4())[:16]
    
    def _store_document_file(self, file_path: str, document_id: str) -> Path:
        """Store the document file in the uploads directory."""
        try:
            doc_dir = self.upload_dir / document_id
            doc_dir.mkdir(exist_ok=True)
            
            stored_path = doc_dir / f"{document_id}.pdf"
            shutil.copy2(file_path, stored_path)
            
            return stored_path
            
        except Exception as e:
            logger.error(f"Error storing document file: {str(e)}")
            raise
    
    def _cleanup_document(self, document_id: str):
        """Clean up document files if processing fails."""
        try:
            doc_dir = self.upload_dir / document_id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
        except Exception as e:
            logger.error(f"Error cleaning up document {document_id}: {str(e)}") 
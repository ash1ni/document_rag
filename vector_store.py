import os
import uuid
from typing import List, Dict, Any, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector storage and retrieval using Qdrant and sentence-transformers."""
    
    def __init__(self):
        # Lazy load config to avoid import issues
        from config import Config
        self.config = Config()
        
        # Initialize Qdrant client with cloud support
        if self.config.is_qdrant_cloud:
            # Qdrant Cloud connection
            self.client = QdrantClient(
                url=self.config.QDRANT_HOST,
                api_key=self.config.QDRANT_API_KEY,
                timeout=60.0  # Increased timeout for cloud
            )
            logger.info(f"Connected to Qdrant Cloud: {self.config.QDRANT_HOST}")
        else:
            # Local Qdrant connection
            self.client = QdrantClient(
                host=self.config.QDRANT_HOST, 
                port=self.config.QDRANT_PORT
            )
            logger.info(f"Connected to local Qdrant: {self.config.QDRANT_HOST}:{self.config.QDRANT_PORT}")
        
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.collection_name = self.config.QDRANT_COLLECTION_NAME
        self._init_collection()

    def _init_collection(self):
        """Initialize the Qdrant collection with proper indexes."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_model.get_sentence_embedding_dimension(),
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
                
                # Create indexes for filtering
                self._create_indexes()
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                # Ensure indexes exist
                self._create_indexes()
                
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    def _create_indexes(self):
        """Create necessary indexes for filtering using the correct API."""
        try:
            # Create index for document_id (required for filtering)
            try:
                # Use the correct method for creating indexes
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema="keyword"
                )
                logger.info("Created index for document_id field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info("Index for document_id already exists")
                else:
                    logger.warning(f"Could not create document_id index: {e}")
            
            # Create index for page_number (optional, for better performance)
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="page_number",
                    field_schema="integer"
                )
                logger.info("Created index for page_number field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info("Index for page_number already exists")
                else:
                    logger.warning(f"Could not create page_number index: {e}")
                    
        except Exception as e:
            logger.warning(f"Error creating indexes: {str(e)}")

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            # Get the embedding model (will initialize if needed)
            model = self.embedding_model
            
            # Use sentence-transformers - fully synchronous
            embeddings = model.encode(texts, convert_to_tensor=False)
            
            # Convert to list of lists if needed
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]], document_id: str) -> bool:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            logger.info(f"Adding {len(documents)} chunks to vector store...")
            
            # Generate embeddings
            texts = [doc['text'] for doc in documents]
            embeddings = self._generate_embeddings(texts)
            
            # Prepare points for Qdrant
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique UUID for each point
                    vector=embedding,
                    payload={
                        'document_id': document_id,
                        'chunk_id': f"{document_id}_{i}",
                        'text': doc['text'],
                        'page_number': doc['page_number'],
                        'chunk_index': i,
                        'filename': doc.get('filename', ''),
                        'processing_method': doc.get('processing_method', ''),
                        'timestamp': doc.get('timestamp', '')
                    }
                )
                points.append(point)
            
            # Add points in batches to avoid memory issues
            batch_size = 100
            total_batches = (len(points) + batch_size - 1) // batch_size
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                
                logger.info(f"Processed batch {batch_num}/{total_batches}")
            
            logger.info(f"Added {len(points)} chunks to vector store for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def search_similar(self, query: str, top_k: int = 5, document_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            
            # Build filter if document_id is specified
            search_filter = None
            if document_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_filter)
                        )
                    ]
                )
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'text': result.payload['text'],
                    'page_number': result.payload['page_number'],
                    'chunk_index': result.payload['chunk_index'],
                    'document_id': result.payload['document_id'],
                    'filename': result.payload.get('filename', ''),
                    'processing_method': result.payload.get('processing_method', ''),
                    'similarity_score': result.score,
                    'method': 'vector_search'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            collection_stats = self.client.get_collection(self.collection_name)
            
            return {
                'collection_name': self.collection_name,
                'total_points': collection_stats.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            # Delete points with matching document_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted all chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document chunks: {str(e)}")
            return False 
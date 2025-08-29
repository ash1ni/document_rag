import os
import logging
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from vector_store import VectorStore
import time
import random

logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG engine using Google Gemini for generation and vector store for retrieval."""
    
    def __init__(self):
        # Lazy load config to avoid import issues
        from config import get_config
        self.config = get_config()
        self.vector_store = VectorStore()
        
        # Initialize Gemini with correct parameter names
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.GEMINI_MODEL,
            google_api_key=self.config.GOOGLE_API_KEY,
            temperature=self.config.TEMPERATURE,
            max_output_tokens=self.config.MAX_TOKENS,
            convert_system_message_to_human=False,  # Fixed parameter name
            timeout=120,  # Fixed parameter name
            max_retries=3
        )
        
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates."""
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context from documents. 
            Always provide accurate, helpful responses based on the context given. If the context doesn't contain enough information 
            to answer the question completely, say so and provide what information you can from the context.
            
            Context: {context}
            
            Question: {question}
            
            Answer the question based on the context above. Be concise but thorough."""),
            ("human", "{question}")
        ])
        
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that creates concise summaries of documents. 
            Based on the provided document chunks, create a comprehensive summary that covers the main topics, 
            key points, and important information.
            
            Document chunks: {context}
            
            Create a well-structured summary that captures the essence of the document."""),
            ("human", "Please summarize this document.")
        ])
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that analyzes documents based on specific requests. 
            Analyze the provided document chunks according to the user's request and provide detailed insights.
            
            Document chunks: {context}
            
            Analysis request: {analysis_request}
            
            Provide a thorough analysis based on the document content. Keep page numbers and references where relevant, just do not show page numbers in the final answer unless specifically asked for it.
            In your answer, give me a well-structured summary with these elements:

            Key figures (overall + category-wise) stated clearly and upfront.

            Alignment with national benchmarks or policies (like RTE/NEP).

            State-level variations or insights if available (high/low outliers).

            A 2–3 line concluding summary that synthesizes the insights (not just repeating numbers).
            Keep the tone concise, formal, and dashboard-ready"""),
            ("human", "Please analyze this document according to the request: {analysis_request}")
        ])
    
    def _call_llm_with_retry(self, messages: List, max_retries: int = 3) -> Dict[str, Any]:
        """Call the LLM with retry logic for better error handling."""
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                return {
                    'success': True,
                    'content': response.content,
                    'attempt': attempt + 1
                }
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"LLM call attempt {attempt + 1} failed: {error_msg}")
                
                # Check if it's a retryable error
                if any(keyword in error_msg.lower() for keyword in [
                    'internal server error', '500', 'rate limit', 'quota exceeded', 'timeout'
                ]):
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                
                # Non-retryable error or max retries reached
                return {
                    'success': False,
                    'error': error_msg,
                    'attempt': attempt + 1
                }
        
        return {
            'success': False,
            'error': 'Max retries exceeded',
            'attempt': max_retries
        }
    
    def query(self, question: str, document_id: Optional[str] = None, top_k: int = 10) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        try:
            # Retrieve more context for better responses
            similar_docs = self.vector_store.search_similar(
                question, 
                top_k=max(top_k, 15),  # Ensure minimum of 15 chunks for context
                document_filter=document_id
            )
            
            if not similar_docs:
                return {
                    'answer': 'I could not find any relevant information in the documents to answer your question.',
                    'sources': [],
                    'confidence': 0.0,
                    'chunks_retrieved': 0
                }
            
            # Prepare context with better formatting
            context_parts = []
            for doc in similar_docs:
                context_parts.append(f"Page {doc['page_number']}: {doc['text']}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Enhanced prompt for better responses
            enhanced_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant that answers questions based on the provided context from documents. 
                
                IMPORTANT INSTRUCTIONS:
                1. Always provide accurate, helpful responses based on the context given
                2. If the context doesn't contain enough information to answer the question completely, say so and provide what information you can from the context
                3. Use specific page numbers and references when possible
                4. Be thorough but concise
                5. If the question is unclear, ask for clarification
                6. Structure your response logically with clear sections if appropriate
                
                Context from document:
                {context}
                
                Question: {question}
                
                Answer the question based on the context above. Be comprehensive and accurate."""),
                ("human", "{question}")
            ])
            
            # Generate answer with enhanced prompt
            messages = enhanced_prompt.format_messages(
                context=context,
                question=question
            )
            
            llm_response = self._call_llm_with_retry(messages)
            
            if not llm_response['success']:
                return {
                    'answer': f"I'm experiencing technical difficulties. Please try again later. Error: {llm_response['error']}",
                    'sources': [],
                    'confidence': 0.0,
                    'chunks_retrieved': len(similar_docs),
                    'error': llm_response['error']
                }
            
            # Prepare sources with better information
            sources = []
            for doc in similar_docs:
                sources.append({
                    'text_preview': doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'],
                    'page_number': doc['page_number'],
                    'similarity_score': doc['similarity_score'],
                    'method': doc['method']
                })
            
            return {
                'answer': llm_response['content'],
                'sources': sources,
                'confidence': min(0.9, max(0.1, similar_docs[0]['similarity_score'])),
                'chunks_retrieved': len(similar_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'chunks_retrieved': 0,
                'error': str(e)
            }
    
    def summarize_document(self, document_id: str) -> Dict[str, Any]:
        """Generate a summary of a document."""
        try:
            # Get more chunks for comprehensive summary
            similar_docs = self.vector_store.search_similar(
                "summarize this document comprehensively", 
                top_k=100,  # Increased from 50 to 100 for better coverage
                document_filter=document_id
            )
            
            if not similar_docs:
                return {
                    'summary': 'No content found to summarize.',
                    'chunks_processed': 0
                }
            
            # Prepare context with better organization
            context_parts = []
            for doc in similar_docs:
                context_parts.append(f"Page {doc['page_number']}: {doc['text']}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Enhanced summary prompt
            enhanced_summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an AI assistant that creates comprehensive summaries of documents. 
                
                Based on the provided document chunks, create a well-structured summary that covers:
                1. Main topics and themes
                2. Key findings and conclusions
                3. Important data and statistics
                4. Structure and organization of the document
                5. Any notable insights or recommendations
                
                Document chunks: {context}
                
                Create a comprehensive, well-organized summary that captures the essence and key information of the document."""),
                ("human", "Please provide a comprehensive summary of this document.")
            ])
            
            # Generate summary with enhanced prompt
            messages = enhanced_summary_prompt.format_messages(context=context)
            
            llm_response = self._call_llm_with_retry(messages)
            
            if not llm_response['success']:
                return {
                    'summary': f"Unable to generate summary due to technical difficulties. Please try again later. Error: {llm_response['error']}",
                    'chunks_processed': len(similar_docs),
                    'error': llm_response['error']
                }
            
            return {
                'summary': llm_response['content'],
                'chunks_processed': len(similar_docs)
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'chunks_processed': 0,
                'error': str(e)
            }
    
    def analyze_document(self, document_id: str, analysis_request: str) -> Dict[str, Any]:
        """Analyze a document based on a specific request."""
        try:
            # Get relevant chunks for analysis
            similar_docs = self.vector_store.search_similar(
                analysis_request, 
                top_k=30,  # Get more chunks for analysis
                document_filter=document_id
            )
            
            if not similar_docs:
                return {
                    'analysis': 'No relevant content found for analysis.',
                    'chunks_processed': 0
                }
            
            # Prepare context
            context = "\n\n".join([f"Page {doc['page_number']}: {doc['text']}" for doc in similar_docs])
            
            # Generate analysis
            messages = self.analysis_prompt.format_messages(
                context=context,
                analysis_request=analysis_request
            )
            
            llm_response = self._call_llm_with_retry(messages)
            
            if not llm_response['success']:
                return {
                    'analysis': f"Unable to perform analysis due to technical difficulties. Please try again later. Error: {llm_response['error']}",
                    'chunks_processed': len(similar_docs),
                    'error': llm_response['error']
                }
            
            return {
                'analysis': llm_response['content'],
                'chunks_processed': len(similar_docs)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {
                'analysis': f"Error performing analysis: {str(e)}",
                'chunks_processed': 0,
                'error': str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system."""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                'model': self.config.GEMINI_MODEL,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'vector_store': 'qdrant',
                'vector_stats': vector_stats,
                'max_tokens': self.config.MAX_TOKENS,
                'temperature': self.config.TEMPERATURE
            }
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {'error': str(e)} 
import os
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        from pymupdf import fitz  # Alternative import
    except ImportError:
        import fitz  # Try direct import as fallback

import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict, Any
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing including text extraction, OCR, and chunking."""
    
    def __init__(self):
        # Lazy load config to avoid import issues
        from config import get_config
        self.config = get_config()
        # Set tesseract path for Windows if needed
        if os.name == 'nt':
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with OCR fallback for image-based content.
        Returns a list of dictionaries with page number, text, and metadata.
        """
        try:
            # Try text extraction first
            text_pages = self._extract_text_pages(pdf_path)
            
            # Check if text extraction was successful
            if self._has_meaningful_text(text_pages):
                logger.info(f"Text extraction successful for {pdf_path}")
                return text_pages
            else:
                logger.info(f"Text extraction failed, falling back to OCR for {pdf_path}")
                return self._extract_text_with_ocr(pdf_path)
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            # Fallback to OCR
            return self._extract_text_with_ocr(pdf_path)
    
    def _extract_text_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using PyMuPDF (faster for text-based PDFs)."""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                pages.append({
                    'page_number': page_num + 1,
                    'text': text.strip(),
                    'method': 'text_extraction',
                    'bbox': page.rect
                })
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")
            
        return pages
    
    def _extract_text_with_ocr(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using OCR for image-based PDFs."""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Convert page to image
                mat = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_data = mat.tobytes("png")
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Preprocess image for better OCR
                img_processed = self._preprocess_image_for_ocr(img)
                
                # Perform OCR
                text = pytesseract.image_to_string(
                    img_processed, 
                    lang=self.config.OCR_LANGUAGE,
                    config=self.config.OCR_CONFIG
                )
                
                pages.append({
                    'page_number': page_num + 1,
                    'text': text.strip(),
                    'method': 'ocr',
                    'bbox': page.rect
                })
                
            doc.close()
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            
        return pages
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL Image
        return Image.fromarray(binary)
    
    def _has_meaningful_text(self, pages: List[Dict[str, Any]]) -> bool:
        """Check if extracted text contains meaningful content."""
        total_text = " ".join([page['text'] for page in pages])
        # Check if we have substantial text (more than just whitespace and special characters)
        meaningful_chars = sum(1 for char in total_text if char.isalnum())
        return meaningful_chars > 100  # At least 100 alphanumeric characters
    
    def chunk_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split text into chunks for better embedding and retrieval.
        Uses sliding window approach for better context preservation.
        """
        chunks = []
        
        for page in pages:
            text = page['text']
            if not text.strip():
                continue
                
            # Split text into sentences first
            sentences = self._split_into_sentences(text)
            
            current_chunk = ""
            chunk_start = 0
            
            for i, sentence in enumerate(sentences):
                # If adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > self.config.CHUNK_SIZE:
                    if current_chunk.strip():
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page_number': page['page_number'],
                            'chunk_id': f"page_{page['page_number']}_chunk_{len(chunks)}",
                            'method': page['method'],
                            'start_sentence': chunk_start,
                            'end_sentence': i - 1
                        })
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, i - 2)  # Include last 2 sentences for overlap
                    current_chunk = " ".join(sentences[overlap_start:i])
                    chunk_start = overlap_start
                
                current_chunk += " " + sentence
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'page_number': page['page_number'],
                    'chunk_id': f"page_{page['page_number']}_chunk_{len(chunks)}",
                    'method': page['method'],
                    'start_sentence': chunk_start,
                    'end_sentence': len(sentences) - 1
                })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple delimiters."""
        import re
        
        # Split by multiple sentence delimiters
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Only keep substantial sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Main method to process a PDF file.
        Returns chunked text ready for embedding.
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        pages = self.extract_text_from_pdf(pdf_path)
        
        if not pages:
            raise ValueError(f"No text could be extracted from {pdf_path}")
        
        # Chunk the text
        chunks = self.chunk_text(pages)
        
        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        
        return chunks 
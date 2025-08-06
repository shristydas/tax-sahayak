import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from config.config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = config.embedding.chunk_size
        self.chunk_overlap = config.embedding.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_content += page.get_text()
                text_content += "\n\n"  # Add page separator
            
            doc.close()
            logger.info(f"Extracted {len(text_content)} characters from PDF")
            return text_content
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove page numbers (common patterns)
            text = re.sub(r'Page \d+', '', text)
            text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
            
            # Remove footer/header patterns
            text = re.sub(r'Income Tax Act, 1961', '', text)
            text = re.sub(r'Chapter [IVX]+', '', text)
            
            # Clean section references
            text = re.sub(r'Section\s+(\d+[A-Z]*)', r'Section \1', text)
            
            # Remove excessive newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Strip and normalize
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to clean text: {e}")
            return text  # Return original if cleaning fails
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        try:
            sections = []
            
            # Pattern to match sections (e.g., "Section 80C", "Section 10(13A)")
            section_pattern = r'Section\s+(\d+[A-Z]*(?:\([^)]+\))*)\s*[-.:]\s*([^§]+?)(?=Section\s+\d+|$)'
            
            matches = re.finditer(section_pattern, text, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                section_number = match.group(1).strip()
                section_content = match.group(2).strip()
                
                if len(section_content) > 100:  # Only include substantial sections
                    sections.append({
                        "section_number": section_number,
                        "content": section_content,
                        "type": "section"
                    })
            
            logger.info(f"Extracted {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"Failed to extract sections: {e}")
            return []
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            if metadata is None:
                metadata = {}
            
            chunks = self.text_splitter.split_text(text)
            
            chunked_docs = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Only include substantial chunks
                    chunk_metadata = {
                        **metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk)
                    }
                    
                    chunked_docs.append({
                        "content": chunk.strip(),
                        "metadata": chunk_metadata
                    })
            
            logger.info(f"Created {len(chunked_docs)} chunks")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            return []
    
    def process_income_tax_act(self, pdf_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info("Processing Income Tax Act PDF")
            
            # Extract raw text
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Extract sections
            sections = self.extract_sections(cleaned_text)
            
            all_chunks = []
            
            # Process each section
            for section in sections:
                section_metadata = {
                    "source": "Income Tax Act 1961",
                    "section_number": section["section_number"],
                    "document_type": "legal",
                    "category": "tax_law"
                }
                
                # Chunk the section content
                section_chunks = self.chunk_text(section["content"], section_metadata)
                all_chunks.extend(section_chunks)
            
            # Also process the full text in chunks for general queries
            general_chunks = self.chunk_text(
                cleaned_text, 
                {
                    "source": "Income Tax Act 1961",
                    "document_type": "legal",
                    "category": "tax_law",
                    "section_type": "general"
                }
            )
            all_chunks.extend(general_chunks)
            
            logger.info(f"Total processed chunks: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Failed to process Income Tax Act: {e}")
            raise
    
    def process_document(self, file_path: str, doc_type: str = "general") -> List[Dict[str, Any]]:
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() == '.pdf':
                text_content = self.extract_text_from_pdf(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Clean text
            cleaned_text = self.clean_text(text_content)
            
            # Create chunks
            base_metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "document_type": doc_type
            }
            
            chunks = self.chunk_text(cleaned_text, base_metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            raise
    
    def validate_income_tax_act(self, pdf_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(pdf_path):
                return {"valid": False, "error": "File not found"}
            
            # Basic validation
            text_sample = self.extract_text_from_pdf(pdf_path)[:5000]  # First 5000 chars
            
            # Check for key indicators
            indicators = [
                "Income Tax Act",
                "Section",
                "Chapter",
                "1961"
            ]
            
            found_indicators = [ind for ind in indicators if ind.lower() in text_sample.lower()]
            
            validation_result = {
                "valid": len(found_indicators) >= 2,
                "found_indicators": found_indicators,
                "file_size": os.path.getsize(pdf_path),
                "sample_text_length": len(text_sample)
            }
            
            return validation_result
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

# Create global instance
document_processor = DocumentProcessor()
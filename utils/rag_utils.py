import logging
from typing import List, Dict, Any, Optional, Tuple
from models.embeddings import vector_store
from utils.document_processor import document_processor
from utils.web_search import web_search
import re

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.web_search = web_search
        self.min_similarity_threshold = 0.3
        self.max_context_length = 8000
    
    def setup_knowledge_base(self, income_tax_pdf_path: str) -> Dict[str, Any]:
        try:
            logger.info("Setting up Income Tax Act knowledge base")
            
            # Validate PDF
            validation = self.document_processor.validate_income_tax_act(income_tax_pdf_path)
            if not validation["valid"]:
                raise ValueError(f"Invalid Income Tax Act PDF: {validation.get('error', 'Unknown error')}")
            
            # Process document
            chunks = self.document_processor.process_income_tax_act(income_tax_pdf_path)
            
            # Prepare texts and metadata for vector store
            texts = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Clear existing data and add new documents
            self.vector_store.clear()
            self.vector_store.add_documents(texts, metadatas)
            
            # Save the index
            self.vector_store.save_index()
            
            stats = self.vector_store.get_stats()
            logger.info(f"Knowledge base setup complete: {stats}")
            
            return {
                "success": True,
                "stats": stats,
                "validation": validation
            }
            
        except Exception as e:
            logger.error(f"Failed to setup knowledge base: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Retrieving context for query: {query[:100]}...")
            
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query, 
                k=k, 
                threshold=self.min_similarity_threshold
            )
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "No relevant information found in the Income Tax Act."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            section_info = ""
            metadata = doc.get("metadata", {})
            
            # Add section information if available
            if metadata.get("section_number"):
                section_info = f"[Section {metadata['section_number']}] "
            
            # Format the content
            content = doc["text"]
            confidence = doc.get("score", 0)
            
            formatted_chunk = f"{section_info}{content}"
            
            # Check length limits
            if total_length + len(formatted_chunk) > self.max_context_length:
                break
            
            context_parts.append(f"{i}. {formatted_chunk}")
            total_length += len(formatted_chunk)
        
        context = "\n\n".join(context_parts)
        return context
    
    def enhance_query(self, query: str) -> str:
        # Add tax-specific context to queries
        tax_keywords = [
            "income tax", "deduction", "exemption", "section", "chapter", 
            "assessment", "return", "liability", "allowance", "rebate"
        ]
        
        # Check if query already contains tax context
        has_tax_context = any(keyword in query.lower() for keyword in tax_keywords)
        
        if not has_tax_context:
            query = f"Under Income Tax Act 1961, {query}"
        
        return query
    
    def generate_tax_prompt(self, user_query: str, context: str, web_results: Optional[str] = None) -> str:
        base_prompt = f"""You are TaxSahayak, an AI assistant specialized in Indian Income Tax Act 1961. 
You help taxpayers understand tax laws, deductions, exemptions, and filing procedures.

CONTEXT FROM INCOME TAX ACT:
{context}
"""
        
        if web_results:
            base_prompt += f"""
ADDITIONAL CURRENT INFORMATION:
{web_results}
"""
        
        base_prompt += f"""
USER QUESTION: {user_query}

INSTRUCTIONS:
1. Answer based primarily on the Income Tax Act context provided above
2. Cite specific sections, clauses, or rules when applicable
3. If using web information, clearly distinguish it from statutory provisions
4. Provide practical examples where helpful
5. Include relevant calculations if applicable
6. Always mention if professional consultation is recommended for complex cases
7. Use simple language while maintaining legal accuracy

RESPONSE FORMAT:
- Direct answer first
- Legal basis (sections/rules)
- Practical application/examples
- Important notes/exceptions
- Disclaimer about professional advice

Answer:"""
        
        return base_prompt
    
    def process_query(self, user_query: str, enable_web_search: bool = False, num_results: int = 5) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Enhance query for better retrieval
            enhanced_query = self.enhance_query(user_query)
            
            # Retrieve relevant context from Income Tax Act
            retrieved_docs = self.retrieve_context(enhanced_query, k=num_results)
            context = self.format_context(retrieved_docs)
            
            web_results = None
            if enable_web_search:
                try:
                    web_results = self.web_search.search_tax_information(user_query)
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
            
            # Generate the prompt
            prompt = self.generate_tax_prompt(user_query, context, web_results)
            
            return {
                "success": True,
                "prompt": prompt,
                "context": context,
                "web_results": web_results,
                "retrieved_docs": retrieved_docs,
                "enhanced_query": enhanced_query
            }
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def extract_section_references(self, text: str) -> List[str]:
        # Extract section references from text
        section_pattern = r'Section\s+(\d+[A-Z]*(?:\([^)]+\))*)'
        matches = re.findall(section_pattern, text, re.IGNORECASE)
        return list(set(matches))  # Remove duplicates
    
    def get_section_details(self, section_number: str) -> Dict[str, Any]:
        try:
            # Search for specific section
            query = f"Section {section_number}"
            results = self.vector_store.similarity_search(query, k=3, threshold=0.7)
            
            section_content = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("section_number") == section_number:
                    section_content.append(result)
            
            return {
                "section_number": section_number,
                "content": section_content,
                "found": len(section_content) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get section details: {e}")
            return {"section_number": section_number, "content": [], "found": False}
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_stats()
    
    def is_knowledge_base_ready(self) -> bool:
        stats = self.get_knowledge_base_stats()
        return stats["total_documents"] > 0

# Create global instance
rag_system = RAGSystem()
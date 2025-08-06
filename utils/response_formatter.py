import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseFormatter:
    def __init__(self):
        self.section_pattern = r'Section\s+(\d+[A-Z]*(?:\([^)]+\))*)'
        self.calculation_pattern = r'₹[\d,]+'
        
    def format_tax_response(self, response: str, mode: str = "detailed") -> str:
        try:
            if mode == "concise":
                return self._format_concise_response(response)
            else:
                return self._format_detailed_response(response)
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return response  # Return original if formatting fails
    
    def _format_concise_response(self, response: str) -> str:
        # Extract key points for concise mode
        lines = response.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if line and (
                line.startswith('•') or 
                line.startswith('-') or
                line.startswith('1.') or
                'Section' in line or
                '₹' in line
            ):
                key_points.append(line)
        
        if key_points:
            return '\n'.join(key_points[:5])  # Top 5 key points
        else:
            # Fallback: first paragraph
            paragraphs = response.split('\n\n')
            return paragraphs[0] if paragraphs else response[:300]
    
    def _format_detailed_response(self, response: str) -> str:
        # Enhanced formatting for detailed responses
        formatted_response = response
        
        # Highlight section references
        formatted_response = re.sub(
            self.section_pattern,
            r'**Section \1**',
            formatted_response
        )
        
        # Format calculations
        formatted_response = re.sub(
            self.calculation_pattern,
            r'**\g<0>**',
            formatted_response
        )
        
        # Add structure markers
        formatted_response = self._add_structure_markers(formatted_response)
        
        return formatted_response
    
    def _add_structure_markers(self, text: str) -> str:
        # Add emoji markers for better readability
        markers = {
            r'(Direct Answer|Answer:|ANSWER:)': '🎯 **\\1**',
            r'(Legal Basis|Legal|Basis:)': '⚖️ **\\1**',
            r'(Example|For example|Examples:)': '📝 **\\1**',
            r'(Important|Note|Warning|Caution:)': '⚠️ **\\1**',
            r'(Calculation|Calculate|Computed:)': '🧮 **\\1**',
            r'(Deduction|Exemption|Allowance:)': '💰 **\\1**',
            r'(Professional advice|Consult|Professional:)': '👨‍💼 **\\1**'
        }
        
        formatted_text = text
        for pattern, replacement in markers.items():
            formatted_text = re.sub(pattern, replacement, formatted_text, flags=re.IGNORECASE)
        
        return formatted_text
    
    def add_source_citations(self, response: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return response
        
        citations = []
        sections_cited = set()
        
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            section = metadata.get('section_number')
            score = doc.get('score', 0)
            
            if section and section not in sections_cited and score > 0.5:
                citations.append(f"Section {section}")
                sections_cited.add(section)
        
        if citations:
            citation_text = f"\n\n📚 **Sources:** {', '.join(citations[:5])} (Income Tax Act, 1961)"
            response += citation_text
        
        return response
    
    def add_disclaimer(self, response: str) -> str:
        disclaimer = """

---
⚠️ **Important Disclaimer:**
This information is for educational purposes only and should not be considered as professional tax advice. 
Tax laws are complex and subject to frequent changes. For specific situations, please consult a qualified 
Chartered Accountant or tax professional.
"""
        return response + disclaimer
    
    def format_calculation(self, calculation_data: Dict[str, Any]) -> str:
        try:
            calc_text = "🧮 **Tax Calculation:**\n\n"
            
            for item, value in calculation_data.items():
                if isinstance(value, (int, float)):
                    calc_text += f"• **{item}:** ₹{value:,.2f}\n"
                else:
                    calc_text += f"• **{item}:** {value}\n"
            
            return calc_text
        except Exception as e:
            logger.error(f"Failed to format calculation: {e}")
            return ""
    
    def format_section_details(self, section_data: Dict[str, Any]) -> str:
        try:
            if not section_data.get('found'):
                return f"❌ Section {section_data.get('section_number')} not found in our database."
            
            section_text = f"📋 **Section {section_data.get('section_number')} Details:**\n\n"
            
            for content in section_data.get('content', []):
                text = content.get('text', '')[:500]  # Limit length
                section_text += f"{text}...\n\n"
            
            return section_text
        except Exception as e:
            logger.error(f"Failed to format section details: {e}")
            return ""
    
    def format_web_results(self, web_results: Optional[str]) -> str:
        if not web_results:
            return ""
        
        return f"""
🌐 **Latest Information:**
{web_results}
"""
    
    def create_quick_actions(self, query: str) -> List[str]:
        # Generate contextual quick action suggestions
        actions = []
        query_lower = query.lower()
        
        if 'deduction' in query_lower:
            actions.extend([
                "Show all available deductions under Chapter VI-A",
                "Compare 80C vs 80CCD deductions",
                "Calculate maximum deduction limits"
            ])
        
        if 'exemption' in query_lower:
            actions.extend([
                "List HRA exemption conditions",
                "Explain LTA exemption rules",
                "Show agricultural income exemption"
            ])
        
        if 'return' in query_lower or 'filing' in query_lower:
            actions.extend([
                "ITR form selection guide",
                "Document checklist for filing",
                "Important filing dates and deadlines"
            ])
        
        if 'salary' in query_lower:
            actions.extend([
                "Salary income tax calculation",
                "Professional tax vs income tax",
                "Perquisites and fringe benefits"
            ])
        
        # Default actions if no specific matches
        if not actions:
            actions = [
                "Explain basic tax slabs for current year",
                "List common deductions available",
                "Show ITR filing process"
            ]
        
        return actions[:3]  # Limit to 3 suggestions
    
    def format_final_response(self, 
                             response: str, 
                             retrieved_docs: List[Dict[str, Any]] = None,
                             web_results: Optional[str] = None,
                             mode: str = "detailed") -> str:
        
        # Format the main response based on mode
        formatted_response = self.format_tax_response(response, mode)
        
        # Add web results if available
        if web_results and mode == "detailed":
            formatted_response += self.format_web_results(web_results)
        
        # Add citations
        if retrieved_docs and mode == "detailed":
            formatted_response = self.add_source_citations(formatted_response, retrieved_docs)
        
        # Add disclaimer
        if mode == "detailed":
            formatted_response = self.add_disclaimer(formatted_response)
        
        return formatted_response
    
    def extract_key_metrics(self, response: str) -> Dict[str, Any]:
        """Extract key metrics from response for analytics"""
        metrics = {
            "sections_mentioned": len(re.findall(self.section_pattern, response)),
            "calculations_present": len(re.findall(self.calculation_pattern, response)),
            "response_length": len(response),
            "word_count": len(response.split()),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics

# Create global instance
response_formatter = ResponseFormatter()
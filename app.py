import streamlit as st
import asyncio
from typing import List, Dict, Any
import logging
from pathlib import Path
import time
import os

# Import our modules
from config.config import config
from models.llm import llm_manager
from models.embeddings import vector_store
from utils.rag_utils import rag_system
from utils.response_formatter import response_formatter
from utils.web_search import web_search
from utils.document_processor import document_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.app.app_title,
    page_icon=config.app.page_icon,
    layout=config.app.layout,
    initial_sidebar_state="expanded"
)

# Professional CSS with improved typography and icons
st.markdown("""
<style>
    /* Global Typography */
    .stApp {
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Header Text Styling */
    .main-header h1 {
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        color: #1f2937 !important;
        letter-spacing: -0.025em;
        text-align: center;
    }
    
    .main-header p {
        font-size: 14px !important;
        color: #6b7280 !important;
        margin: 0 !important;
        font-weight: 500;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    /* Chat Messages with Professional Icons */
    .stChatMessage {
        font-size: 14px !important;
        line-height: 1.6 !important;
        margin: 1rem 0 !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        border-radius: 18px 18px 6px 18px !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25) !important;
        border: none !important;
    }
    
    .stChatMessage[data-testid="user-message"] .stMarkdown {
        color: white !important;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="assistant-message"] {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px 18px 18px 6px !important;
        color: #1f2937 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Professional Avatar Icons */
    .stChatMessage .stAvatar {
        width: 32px !important;
        height: 32px !important;
        border-radius: 8px !important;
    }
    
    /* User Avatar - Professional User Icon */
    .stChatMessage[data-testid="user-message"] .stAvatar {
        background: linear-gradient(135deg, #1e40af, #1e3a8a) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stChatMessage[data-testid="user-message"] .stAvatar::after {
        content: "👤" !important;
        font-size: 16px !important;
        filter: brightness(0) invert(1);
    }
    
    /* Assistant Avatar - Professional Bot Icon */
    .stChatMessage[data-testid="assistant-message"] .stAvatar {
        background: linear-gradient(135deg, #059669, #047857) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] .stAvatar::after {
        content: "⚡" !important;
        font-size: 16px !important;
    }
    
    /* Input Area Styling */
    .stChatInputContainer {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        margin: 1rem 0 !important;
    }
    
    .stChatInput input {
        font-size: 14px !important;
        color: #374151 !important;
        border: none !important;
        background: transparent !important;
        font-family: inherit !important;
    }
    
    .stChatInput input::placeholder {
        color: #9ca3af !important;
        font-size: 14px !important;
    }
    
    /* Sidebar Typography */
    .stSidebar {
        background: #f9fafb !important;
        border-right: 1px solid #e5e7eb !important;
    }
    
    .stSidebar .stMarkdown h1 {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    
    .stSidebar .stMarkdown h2 {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin-top: 1.5rem !important;
    }
    
    .stSidebar .stMarkdown h3 {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #4b5563 !important;
    }
    
    .stSidebar .stMarkdown p {
        font-size: 12px !important;
        color: #6b7280 !important;
        line-height: 1.4 !important;
    }
    
    /* Professional Buttons */
    .stButton button {
        font-family: inherit !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid #d1d5db !important;
        background: white !important;
        color: #374151 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        background: #f3f4f6 !important;
        border-color: #9ca3af !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Primary Action Buttons */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e3a8a) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Success/Warning/Error Styling */
    .stSuccess {
        background: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-radius: 8px !important;
        color: #166534 !important;
        font-size: 12px !important;
    }
    
    .stWarning {
        background: #fffbeb !important;
        border: 1px solid #fed7aa !important;
        border-radius: 8px !important;
        color: #9a3412 !important;
        font-size: 12px !important;
    }
    
    .stError {
        background: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        border-radius: 8px !important;
        color: #991b1b !important;
        font-size: 12px !important;
    }
    
    /* Form Elements */
    .stSelectbox label {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }
    
    .stCheckbox label {
        font-size: 14px !important;
        color: #374151 !important;
    }
    
    .stRadio label {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }
    
    /* File Uploader */
    .stFileUploader label {
        font-size: 12px !important;
        color: #6b7280 !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 8px !important;
        color: #1e40af !important;
        font-size: 12px !important;
    }
    
    /* Metrics and Stats */
    .metric-container {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .metric-title {
        font-size: 12px !important;
        color: #6b7280 !important;
        font-weight: 500 !important;
    }
    
    .metric-value {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 20px !important;
        }
        
        .main-header p {
            font-size: 12px !important;
        }
        
        .stChatMessage {
            font-size: 13px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_base_loaded" not in st.session_state:
        st.session_state.knowledge_base_loaded = False
    if "selected_model" not in st.session_state:
        available_models = llm_manager.get_available_models()
        st.session_state.selected_model = available_models[0] if available_models else None
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = web_search.is_enabled()
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Detailed"

def display_header():
    """Display the main header"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🧾 TaxSahayak - AI Tax Assistant")
    st.markdown("*Your Personal Income Tax Buddy*")
    st.markdown('</div>', unsafe_allow_html=True)

def display_sidebar():
    """Display and handle sidebar controls"""
    st.sidebar.title("⚙️ Settings")
    
    # Model Selection
    st.sidebar.subheader("🤖 AI Model")
    available_models = llm_manager.get_available_models()
    
    if available_models:
        selected_model = st.sidebar.selectbox(
            "Choose AI Model:",
            available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
        )
        st.session_state.selected_model = selected_model
    else:
        st.sidebar.error("❌ No AI models configured. Please check your API keys.")
        return False
    
    # Web Search Toggle
    st.sidebar.subheader("🌐 Web Search")
    web_search_enabled = st.sidebar.checkbox(
        "Enable Live Web Search",
        value=st.session_state.web_search_enabled,
        help="Get latest tax information from the web"
    )
    st.session_state.web_search_enabled = web_search_enabled
    
    # Response Mode
    st.sidebar.subheader("📝 Response Style")
    response_mode = st.sidebar.radio(
        "Choose response detail level:",
        config.app.response_modes,
        index=config.app.response_modes.index(st.session_state.response_mode)
    )
    st.session_state.response_mode = response_mode
    
    # Knowledge Base Status
    st.sidebar.subheader("📚 Knowledge Base")
    
    # Initialize document list in session state
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    
    if rag_system.is_knowledge_base_ready():
        stats = rag_system.get_knowledge_base_stats()
        st.sidebar.success("✅ Knowledge Base Active")
        st.sidebar.info(f"Documents: {stats['total_documents']}")
        st.session_state.knowledge_base_loaded = True
        
        # Show uploaded documents
        if st.session_state.uploaded_documents:
            st.sidebar.markdown("**Uploaded Documents:**")
            for i, doc in enumerate(st.session_state.uploaded_documents):
                st.sidebar.markdown(f"📄 {doc}")
        
        # Option to add more documents
        st.sidebar.markdown("**Add More Documents:**")
        additional_files = st.sidebar.file_uploader(
            "Upload additional documents:",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            key="additional_files",
            help="Upload more documents to enrich the knowledge base"
        )
        
        if additional_files:
            if st.sidebar.button("➕ Add to Knowledge Base", key="add_docs"):
                add_documents_to_knowledge_base(additional_files)
        
        # Option to clear knowledge base
        if st.sidebar.button("🗑️ Clear Knowledge Base", key="clear_kb"):
            clear_knowledge_base()
            
    else:
        st.sidebar.warning("⚠️ Knowledge Base Empty")
        st.session_state.knowledge_base_loaded = False
        
        # Upload initial documents
        uploaded_files = st.sidebar.file_uploader(
            "Upload documents to create knowledge base:",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload Income Tax Act 1961 PDF and other relevant tax documents"
        )
        
        if uploaded_files:
            if st.sidebar.button("🚀 Create Knowledge Base"):
                setup_knowledge_base_multiple(uploaded_files)
    
    # Quick Actions
    st.sidebar.subheader("🚀 Quick Actions")
    quick_actions = [
        "What are the current income tax slabs?",
        "How to claim HRA exemption?",
        "What documents needed for ITR filing?",
        "Explain 80C deduction limit",
        "Calculate tax on salary income"
    ]
    
    for action in quick_actions:
        if st.sidebar.button(f"💡 {action}"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": action})
            
            # Generate response immediately
            with st.spinner("🤔 Processing your question..."):
                response = generate_response(action)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to show the conversation
            st.rerun()
    
    return True

def setup_knowledge_base_multiple(uploaded_files):
    """Setup knowledge base from multiple uploaded files"""
    try:
        with st.spinner(f"🔄 Processing {len(uploaded_files)} documents..."):
            os.makedirs("./data", exist_ok=True)
            processed_files = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file
                file_path = f"./data/{uploaded_file.name}"
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                processed_files.append(file_path)
                st.session_state.uploaded_documents.append(uploaded_file.name)
            
            # Setup knowledge base with all files
            result = setup_multiple_documents(processed_files)
            
            if result["success"]:
                st.sidebar.success(f"✅ Knowledge Base Created with {len(uploaded_files)} documents!")
                st.session_state.knowledge_base_loaded = True
                st.rerun()
            else:
                st.sidebar.error(f"❌ Setup failed: {result['error']}")
    
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

def add_documents_to_knowledge_base(additional_files):
    """Add more documents to existing knowledge base"""
    try:
        with st.spinner(f"🔄 Adding {len(additional_files)} new documents..."):
            os.makedirs("./data", exist_ok=True)
            new_files = []
            
            for uploaded_file in additional_files:
                # Save uploaded file
                file_path = f"./data/{uploaded_file.name}"
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                new_files.append(file_path)
                st.session_state.uploaded_documents.append(uploaded_file.name)
            
            # Add documents to existing knowledge base
            result = add_documents_to_existing_kb(new_files)
            
            if result["success"]:
                st.sidebar.success(f"✅ Added {len(additional_files)} documents to knowledge base!")
                st.rerun()
            else:
                st.sidebar.error(f"❌ Adding failed: {result['error']}")
    
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

def clear_knowledge_base():
    """Clear the entire knowledge base"""
    try:
        with st.spinner("🗑️ Clearing knowledge base..."):
            # Clear vector store
            vector_store.clear()
            
            # Clear session state
            st.session_state.uploaded_documents = []
            st.session_state.knowledge_base_loaded = False
            
            st.sidebar.success("✅ Knowledge base cleared!")
            st.rerun()
    
    except Exception as e:
        st.sidebar.error(f"❌ Error clearing knowledge base: {str(e)}")

def setup_multiple_documents(file_paths):
    """Process multiple documents and create knowledge base"""
    try:
        all_chunks = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                chunks = document_processor.process_income_tax_act(file_path)
            elif file_path.endswith('.txt'):
                chunks = process_txt_file(file_path)
            else:
                continue  # Skip unsupported files
            
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return {"success": False, "error": "No valid documents processed"}
        
        # Clear existing vector store and add all chunks
        vector_store.clear()
        texts = [chunk["content"] for chunk in all_chunks]
        metadatas = [chunk["metadata"] for chunk in all_chunks]
        
        vector_store.add_documents(texts, metadatas)
        vector_store.save_index()
        
        return {
            "success": True,
            "stats": {"total_chunks": len(all_chunks)}
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def add_documents_to_existing_kb(file_paths):
    """Add documents to existing knowledge base"""
    try:
        new_chunks = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                chunks = document_processor.process_income_tax_act(file_path)
            elif file_path.endswith('.txt'):
                chunks = process_txt_file(file_path)
            else:
                continue  # Skip unsupported files
            
            new_chunks.extend(chunks)
        
        if not new_chunks:
            return {"success": False, "error": "No valid documents processed"}
        
        # Add to existing vector store
        texts = [chunk["content"] for chunk in new_chunks]
        metadatas = [chunk["metadata"] for chunk in new_chunks]
        
        vector_store.add_documents(texts, metadatas)
        vector_store.save_index()
        
        return {
            "success": True,
            "stats": {"new_chunks": len(new_chunks)}
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def process_txt_file(file_path):
    """Process text file and create chunks"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create chunks using document processor
        filename = os.path.basename(file_path)
        base_metadata = {
            "source": filename,
            "file_path": file_path,
            "document_type": "text"
        }
        
        chunks = document_processor.chunk_text(content, base_metadata)
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to process text file {file_path}: {e}")
        return []

def display_chat_interface():
    """Display the main chat interface"""
    if not st.session_state.knowledge_base_loaded:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("📚 Please upload the Income Tax Act PDF in the sidebar to enable full functionality.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your tax question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("🤔 Thinking..."):
                response = generate_response(prompt)
            
            message_placeholder.markdown(response)
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})

def generate_response(user_query: str) -> str:
    """Generate AI response to user query"""
    try:
        if not st.session_state.knowledge_base_loaded:
            return "❌ Please upload the Income Tax Act PDF first to get accurate tax guidance."
        
        # Process query with RAG
        query_result = rag_system.process_query(
            user_query,
            enable_web_search=st.session_state.web_search_enabled,
            num_results=5
        )
        
        if not query_result["success"]:
            return f"❌ Error processing query: {query_result['error']}"
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": query_result["prompt"]},
            {"role": "user", "content": user_query}
        ]
        
        # Get LLM response
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            raw_response = loop.run_until_complete(
                llm_manager.generate_response(
                    st.session_state.selected_model,
                    messages
                )
            )
        finally:
            loop.close()
        
        # Format response
        formatted_response = response_formatter.format_final_response(
            raw_response,
            retrieved_docs=query_result.get("retrieved_docs"),
            web_results=query_result.get("web_results"),
            mode=st.session_state.response_mode.lower()
        )
        
        return formatted_response
    
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return f"❌ Sorry, I encountered an error: {str(e)}"

def display_configuration_issues():
    """Display configuration issues if any"""
    issues = config.validate_config()
    if issues:
        st.error("⚠️ Configuration Issues:")
        for component, issue in issues.items():
            st.error(f"• {component}: {issue}")
        
        st.info("💡 Please check your .env file and ensure all required API keys are configured.")
        return False
    return True


def main():
    """Main application function"""
    # Initialize
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Check configuration
    if not display_configuration_issues():
        return
    
    # Display sidebar and check if models are available
    if not display_sidebar():
        st.error("Cannot proceed without configured AI models.")
        return
    
    # Display main chat interface
    display_chat_interface()

if __name__ == "__main__":
    main()
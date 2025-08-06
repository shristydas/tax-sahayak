# TaxSahayak - AI-Powered Personal Tax Assistant

🧾 **TaxSahayak** is an intelligent personal tax assistant designed specifically for Indian taxpayers. It leverages advanced RAG (Retrieval-Augmented Generation) technology with the complete Income Tax Act 1961 to provide accurate, contextual, and legally grounded tax guidance.

## ✨ Features

### 🤖 Multiple AI Models
- **OpenAI GPT-4/GPT-3.5**: Premium AI models for comprehensive responses
- **Mistral 7B**: Open-source alternative for efficient processing
- **Google Gemini**: Google's advanced language model
- **Grok**: X.AI's latest language model
- **Dynamic Model Switching**: Choose the best model for your needs

### 📚 RAG-Powered Knowledge Base
- **Complete Income Tax Act 1961**: Vector database with full legal text
- **Semantic Search**: Find relevant sections using natural language
- **Section-wise Retrieval**: Accurate citations and references
- **FAISS Vector Store**: Fast and efficient similarity search

### 🌐 Live Web Search Integration
- **Current Information**: Get latest tax updates and policy changes
- **Multiple Providers**: Serper, Google Custom Search, DuckDuckGo
- **Toggle Control**: Enable/disable web search as needed
- **Source Attribution**: Clear distinction between statutory and web information

### 💡 Intelligent Chat Interface
- **Natural Conversations**: Ask questions in plain English
- **Context Retention**: Maintains conversation history
- **Response Modes**: Choose between Concise and Detailed responses
- **Quick Actions**: Pre-built common tax queries
- **Professional Formatting**: Clear, readable responses with citations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher (for local development)
- API keys for at least one AI model (OpenAI, Google, Mistral, or Grok)
- Income Tax Act 1961 PDF document

### Option 1: Streamlit Cloud Deployment (Recommended)

1. **Deploy to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io/)
   - Connect your GitHub repository
   - Select main branch and `app.py` as entry point

2. **Configure Secrets:**
   - Go to your app's settings in Streamlit Cloud
   - Click "Secrets" in the sidebar
   - Add your secrets in TOML format:

```toml
# Required: At least one LLM API key
OPENAI_API_KEY = "your_openai_api_key_here"
MISTRAL_API_KEY = "your_mistral_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"
GROK_API_KEY = "your_grok_api_key_here"

# Optional: Search APIs for web search
SERPER_API_KEY = "your_serper_api_key_here"
GOOGLE_SEARCH_API_KEY = "your_google_search_api_key_here"
GOOGLE_CSE_ID = "your_google_cse_id_here"
```

3. **Save and Deploy** - Your app will automatically restart with the new configuration

### Option 2: Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/shristydas/TaxSahayak.git
cd TaxSahayak
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

4. **Run the application:**
```bash
streamlit run app.py
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# AI Model API Keys (Configure at least one)
OPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROK_API_KEY=your_grok_api_key_here

# Web Search API Keys (Optional but recommended)
SERPER_API_KEY=your_serper_api_key_here
GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here

# Application Settings
DEBUG=False
DATA_PATH=./data
```

### API Keys Setup

#### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and generate an API key
3. Add to `.env` file as `OPENAI_API_KEY`

#### Google Gemini
1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Generate an API key
3. Add to `.env` file as `GOOGLE_API_KEY`

#### Mistral AI
1. Visit [Mistral AI](https://mistral.ai/)
2. Sign up and get API access
3. Add to `.env` file as `MISTRAL_API_KEY`

#### Grok (X.AI)
1. Visit [X.AI](https://x.ai/)
2. Request API access
3. Add to `.env` file as `GROK_API_KEY`

#### Web Search (Optional)
- **Serper**: Visit [serper.dev](https://serper.dev/) for Google Search API
- **Google Custom Search**: Use Google Cloud Console for Custom Search API

## 📖 Usage Guide

### Initial Setup
1. Launch the application using `streamlit run app.py`
2. Upload the Income Tax Act 1961 PDF in the sidebar
3. Wait for the knowledge base to be processed and indexed
4. Select your preferred AI model and response style

### Asking Questions
- **Natural Language**: "How much can I save under Section 80C?"
- **Specific Sections**: "Explain Section 10(13A) HRA exemption"
- **Calculations**: "Calculate tax on 15 lakh salary"
- **Procedures**: "What documents do I need for ITR filing?"

### Response Modes
- **Detailed**: Comprehensive explanations with examples, calculations, and citations
- **Concise**: Quick answers with key points only

### Web Search Toggle
- **Enabled**: Includes latest policy updates and current information
- **Disabled**: Uses only Income Tax Act knowledge base

## 🏗️ Architecture

```
neostats/
├── config/
│   └── config.py              # Configuration management
├── models/
│   ├── llm.py                 # AI model integrations
│   └── embeddings.py          # Vector embeddings and FAISS
├── utils/
│   ├── document_processor.py  # PDF parsing and chunking
│   ├── rag_utils.py          # RAG system logic
│   ├── web_search.py         # Web search integration
│   └── response_formatter.py  # Response formatting
├── data/                      # Data storage
├── app.py                     # Main Streamlit application
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

### Key Components

#### 1. RAG System (`utils/rag_utils.py`)
- Processes user queries
- Retrieves relevant Income Tax Act sections
- Combines with web search results
- Generates contextual prompts for AI models

#### 2. LLM Manager (`models/llm.py`)
- Manages multiple AI model providers
- Handles API calls and error handling
- Supports streaming responses
- Provides model selection interface

#### 3. Vector Store (`models/embeddings.py`)
- FAISS-based vector database
- Efficient similarity search
- Persistent storage
- Configurable embedding models

#### 4. Document Processor (`utils/document_processor.py`)
- PDF text extraction
- Intelligent chunking
- Section-based parsing
- Metadata preservation

## 🎯 Use Cases

### Individual Taxpayers
- Understanding tax slabs and rates
- Claiming deductions and exemptions
- ITR filing guidance
- Tax planning strategies

### Tax Professionals
- Quick reference for tax sections
- Client consultation support
- Latest policy updates
- Complex scenario analysis

### Students and Researchers
- Learning Income Tax Act provisions
- Academic research support
- Legal interpretation guidance
- Comparative analysis

## 🔍 Example Queries

```
"What are the income tax slabs for FY 2024-25?"
"How to calculate HRA exemption with examples?"
"Can I claim both 80C and 80CCD deductions?"
"What is the penalty for late ITR filing?"
"Explain the difference between Section 54 and 54F"
"How much tax will I pay on 12 lakh salary?"
```

## ⚠️ Important Disclaimers

1. **Educational Purpose**: This tool provides information for educational purposes only
2. **Professional Advice**: Always consult qualified Chartered Accountants for complex cases
3. **Accuracy**: While based on Income Tax Act, interpretations may vary
4. **Updates**: Tax laws change frequently; verify with latest notifications
5. **Liability**: Users are responsible for their tax compliance decisions

## 🛡️ Privacy and Security

- **No Data Storage**: Conversations are not permanently stored
- **API Security**: All API keys are encrypted and secure
- **Local Processing**: Vector database runs locally
- **No Personal Data**: No personal tax information is retained

## 📊 Performance

- **Response Time**: < 5 seconds for complex queries
- **Accuracy**: High accuracy with Income Tax Act citations
- **Scalability**: Supports multiple concurrent users
- **Offline Capability**: Core features work without internet

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


**Made with ❤️ for Indian Taxpayers**

*TaxSahayak - Simplifying Tax Compliance Through AI*

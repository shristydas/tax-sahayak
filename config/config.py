import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import streamlit as st

# Load environment variables for local development
load_dotenv()


@dataclass
class ModelConfig:
    name: str
    api_key: Optional[str]
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    enabled: bool = True

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class VectorDBConfig:
    db_type: str = "faiss"
    persist_directory: str = "./data/vector_store"
    collection_name: str = "income_tax_act"

@dataclass
class SearchConfig:
    provider: str = "serper"  # serper, google, duckduckgo
    api_key: Optional[str] = None
    max_results: int = 5
    enabled: bool = True

@dataclass
class AppConfig:
    app_title: str = "TaxSahayak - AI Tax Assistant"
    page_icon: str = "🧾"
    layout: str = "wide"
    sidebar_width: int = 300
    max_conversation_length: int = 50
    response_modes: list = None
    
    def __post_init__(self):
        if self.response_modes is None:
            self.response_modes = ["Concise", "Detailed"]

class Config:
    def __init__(self):
        self.models = self._init_models()
        self.embedding = EmbeddingConfig()
        self.vector_db = VectorDBConfig()
        self.search = self._init_search()
        self.app = AppConfig()
        self.data_path = os.getenv("DATA_PATH", "./data")
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
    
    def _init_models(self) -> Dict[str, ModelConfig]:
        return {
            "OpenAI GPT-4": ModelConfig(
                name="gpt-4o",
                api_key=st.secrets.get("OPENAI_API_KEY"),
                enabled=bool(st.secrets.get("OPENAI_API_KEY"))
            ),
            "OpenAI GPT-4 Mini": ModelConfig(
                name="gpt-4o-mini",
                api_key=get_secret("OPENAI_API_KEY"),
                max_tokens=3000,
                enabled=bool(get_secret("OPENAI_API_KEY"))
            ),
            "Mistral 7B": ModelConfig(
                name="mistral-small-latest",
                api_key=st.secrets.get("MISTRAL_API_KEY"),
                base_url=st.secrets.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
                enabled=bool(st.secrets.get("MISTRAL_API_KEY"))
            ),
            "Google Gemini": ModelConfig(
                name="gemini-1.5-flash",
                api_key=st.secrets.get("GOOGLE_API_KEY"),
                enabled=bool(st.secrets.get("GOOGLE_API_KEY"))
            ),
            "Grok": ModelConfig(
                name="grok-beta",
                api_key=st.secrets.get("GROK_API_KEY"),
                base_url=st.secrets.get("GROK_BASE_URL", "https://api.x.ai/v1"),
                enabled=bool(st.secrets.get("GROK_API_KEY"))
            )
        }
    
    def _init_search(self) -> SearchConfig:
        search_config = SearchConfig()
        
        if st.secrets.get("SERPER_API_KEY"):
            search_config.provider = "serper"
            search_config.api_key = st.secrets.get("SERPER_API_KEY")
        elif st.secrets.get("GOOGLE_SEARCH_API_KEY"):
            search_config.provider = "google"
            search_config.api_key = st.secrets.get("GOOGLE_SEARCH_API_KEY")
        else:
            search_config.provider = "duckduckgo"
            search_config.enabled = True
        
        return search_config
    
    def get_enabled_models(self) -> Dict[str, ModelConfig]:
        return {name: config for name, config in self.models.items() if config.enabled}
    
    def validate_config(self) -> Dict[str, str]:
        issues = {}
        
        enabled_models = self.get_enabled_models()
        if not enabled_models:
            issues["models"] = "No LLM models configured. Please add API keys."
        
        if self.search.enabled and not self.search.api_key and self.search.provider != "duckduckgo":
            issues["search"] = f"Search enabled but no API key for {self.search.provider}"
        
        return issues

config = Config()
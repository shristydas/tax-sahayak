import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
from config.config import config

logger = logging.getLogger(__name__)

class WebSearchProvider:
    def __init__(self):
        self.search_config = config.search
        self.max_results = self.search_config.max_results
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement search method")

class SerperSearchProvider(WebSearchProvider):
    def __init__(self):
        super().__init__()
        self.api_key = self.search_config.api_key
        self.base_url = "https://google.serper.dev/search"
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": self.max_results,
                "gl": "in",  # India
                "hl": "en"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_serper_results(data)
                    else:
                        logger.error(f"Serper API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return []
    
    def _parse_serper_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        
        # Parse organic results
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "serper",
                "type": "web"
            })
        
        return results

class GoogleSearchProvider(WebSearchProvider):
    def __init__(self):
        super().__init__()
        self.api_key = self.search_config.api_key
        self.cse_id = config.search.api_key  # Assuming CSE ID is stored here
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": self.max_results,
                "gl": "in",
                "hl": "en"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_google_results(data)
                    else:
                        logger.error(f"Google Search API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    def _parse_google_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "google",
                "type": "web"
            })
        
        return results

class DuckDuckGoProvider(WebSearchProvider):
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.duckduckgo.com/"
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        try:
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "region": "in-en"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_duckduckgo_results(data)
                    else:
                        logger.error(f"DuckDuckGo API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def _parse_duckduckgo_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        
        # Parse related topics
        for item in data.get("RelatedTopics", [])[:self.max_results]:
            if isinstance(item, dict) and "Text" in item:
                results.append({
                    "title": item.get("Text", "")[:100],
                    "url": item.get("FirstURL", ""),
                    "snippet": item.get("Text", ""),
                    "source": "duckduckgo",
                    "type": "web"
                })
        
        return results

class WebSearch:
    def __init__(self):
        self.providers = self._initialize_providers()
        self.enabled = config.search.enabled
    
    def _initialize_providers(self) -> Dict[str, WebSearchProvider]:
        providers = {}
        
        try:
            if config.search.provider == "serper" and config.search.api_key:
                providers["serper"] = SerperSearchProvider()
            
            if config.search.provider == "google" and config.search.api_key:
                providers["google"] = GoogleSearchProvider()
            
            # DuckDuckGo is always available as fallback
            providers["duckduckgo"] = DuckDuckGoProvider()
            
        except Exception as e:
            logger.warning(f"Failed to initialize some search providers: {e}")
        
        return providers
    
    async def search(self, query: str, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        
        # Use specified provider or default
        provider_name = provider or config.search.provider
        
        # Fallback to DuckDuckGo if primary provider fails
        if provider_name not in self.providers:
            provider_name = "duckduckgo"
        
        if provider_name not in self.providers:
            logger.error("No search providers available")
            return []
        
        try:
            logger.info(f"Searching with {provider_name}: {query}")
            results = await self.providers[provider_name].search(query)
            logger.info(f"Found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Search failed with {provider_name}: {e}")
            
            # Try fallback to DuckDuckGo
            if provider_name != "duckduckgo" and "duckduckgo" in self.providers:
                try:
                    logger.info("Falling back to DuckDuckGo")
                    return await self.providers["duckduckgo"].search(query)
                except Exception as fallback_error:
                    logger.error(f"Fallback search also failed: {fallback_error}")
            
            return []
    
    def search_tax_information(self, user_query: str) -> Optional[str]:
        try:
            # Enhance query for tax-specific search
            tax_query = f"India income tax {user_query} 2024 2025"
            
            # Run search synchronously for now (can be made async later)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self.search(tax_query))
            loop.close()
            
            if not results:
                return None
            
            # Format results for RAG context
            formatted_results = []
            for i, result in enumerate(results[:3], 1):  # Top 3 results
                formatted_result = f"{i}. {result['title']}\n{result['snippet']}\nSource: {result['url']}\n"
                formatted_results.append(formatted_result)
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            logger.error(f"Tax information search failed: {e}")
            return None
    
    def is_enabled(self) -> bool:
        return self.enabled and len(self.providers) > 0
    
    def get_available_providers(self) -> List[str]:
        return list(self.providers.keys())

# Create global instance
web_search = WebSearch()
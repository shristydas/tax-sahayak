import openai
import google.generativeai as genai
from typing import Optional, Dict, Any, AsyncGenerator
import asyncio
import aiohttp
import json
import logging
from config.config import config, ModelConfig

logger = logging.getLogger(__name__)

class LLMProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = config.models.get(model_name)
        if not self.model_config or not self.model_config.enabled:
            raise ValueError(f"Model {model_name} is not configured or enabled")
    
    async def generate_response(self, messages: list, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement generate_response")
    
    async def stream_response(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        raise NotImplementedError("Subclasses must implement stream_response")

class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        openai.api_key = self.model_config.api_key
        self.client = openai.OpenAI(api_key=self.model_config.api_key)
    
    async def generate_response(self, messages: list, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.model_config.max_tokens),
                temperature=kwargs.get('temperature', self.model_config.temperature),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def stream_response(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model_config.name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.model_config.max_tokens),
                temperature=kwargs.get('temperature', self.model_config.temperature),
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=self.model_config.api_key)
        self.model = genai.GenerativeModel(self.model_config.name)
    
    async def generate_response(self, messages: list, **kwargs) -> str:
        try:
            # Convert OpenAI format to Gemini format
            gemini_messages = self._convert_messages(messages)
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(gemini_messages)
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def stream_response(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        try:
            gemini_messages = self._convert_messages(messages)
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(gemini_messages, stream=True)
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
    
    def _convert_messages(self, messages: list) -> str:
        # Convert OpenAI message format to single prompt for Gemini
        prompt_parts = []
        for msg in messages:
            if msg['role'] == 'system':
                prompt_parts.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                prompt_parts.append(f"User: {msg['content']}")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"Assistant: {msg['content']}")
        return "\n".join(prompt_parts)

class MistralProvider(LLMProvider):
    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    async def generate_response(self, messages: list, **kwargs) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.model_config.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_config.name,
                    "messages": messages,
                    "max_tokens": kwargs.get('max_tokens', self.model_config.max_tokens),
                    "temperature": kwargs.get('temperature', self.model_config.temperature)
                }
                
                async with session.post(
                    f"{self.model_config.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Mistral API error {response.status}: {error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    result = await response.json()
                    logger.info(f"Mistral response: {result}")
                    
                    if 'choices' not in result or not result['choices']:
                        logger.error(f"Unexpected Mistral response format: {result}")
                        raise Exception(f"Invalid response format: {result}")
                    
                    return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise
    
    async def stream_response(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.model_config.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_config.name,
                    "messages": messages,
                    "max_tokens": kwargs.get('max_tokens', self.model_config.max_tokens),
                    "temperature": kwargs.get('temperature', self.model_config.temperature),
                    "stream": True
                }
                
                async with session.post(
                    f"{self.model_config.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data = line_str[6:]
                                if data != '[DONE]':
                                    try:
                                        chunk = json.loads(data)
                                        content = chunk['choices'][0]['delta'].get('content')
                                        if content:
                                            yield content
                                    except json.JSONDecodeError:
                                        continue
        except Exception as e:
            logger.error(f"Mistral streaming error: {e}")
            raise

class GrokProvider(LLMProvider):
    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    async def generate_response(self, messages: list, **kwargs) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.model_config.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_config.name,
                    "messages": messages,
                    "max_tokens": kwargs.get('max_tokens', self.model_config.max_tokens),
                    "temperature": kwargs.get('temperature', self.model_config.temperature)
                }
                
                async with session.post(
                    f"{self.model_config.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            raise
    
    async def stream_response(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.model_config.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_config.name,
                    "messages": messages,
                    "max_tokens": kwargs.get('max_tokens', self.model_config.max_tokens),
                    "temperature": kwargs.get('temperature', self.model_config.temperature),
                    "stream": True
                }
                
                async with session.post(
                    f"{self.model_config.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data = line_str[6:]
                                if data != '[DONE]':
                                    try:
                                        chunk = json.loads(data)
                                        content = chunk['choices'][0]['delta'].get('content')
                                        if content:
                                            yield content
                                    except json.JSONDecodeError:
                                        continue
        except Exception as e:
            logger.error(f"Grok streaming error: {e}")
            raise

class LLMManager:
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        for model_name, model_config in config.get_enabled_models().items():
            try:
                if "OpenAI" in model_name:
                    self.providers[model_name] = OpenAIProvider(model_name)
                elif "Gemini" in model_name:
                    self.providers[model_name] = GeminiProvider(model_name)
                elif "Mistral" in model_name:
                    self.providers[model_name] = MistralProvider(model_name)
                elif "Grok" in model_name:
                    self.providers[model_name] = GrokProvider(model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")
    
    def get_available_models(self) -> list:
        return list(self.providers.keys())
    
    async def generate_response(self, model_name: str, messages: list, **kwargs) -> str:
        if model_name not in self.providers:
            raise ValueError(f"Model {model_name} not available")
        return await self.providers[model_name].generate_response(messages, **kwargs)
    
    async def stream_response(self, model_name: str, messages: list, **kwargs):
        if model_name not in self.providers:
            raise ValueError(f"Model {model_name} not available")
        async for chunk in self.providers[model_name].stream_response(messages, **kwargs):
            yield chunk

llm_manager = LLMManager()
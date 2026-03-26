"""
model_factory.py — Unified Chat Model factory for multi-provider support.
Updated to include local translation via Hugging Face Pipeline.
"""

from typing import Optional, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.config import PROVIDERS, TEMPERATURE, MAX_TOKENS
from modules.local_translator import get_local_pipeline

def get_chat_model(provider: str, api_key: str) -> Any:
    """
    Returns a unified translation engine (LLM or Pipeline) for the given provider.
    """
    conf = PROVIDERS.get(provider)
    if not conf:
        raise ValueError(f"Unsupported provider: {provider}")

    model_name = conf["chat_model"]

    # 1. Local Model Case
    if provider == "Local (Helsinki-NLP)":
        return get_local_pipeline()

    # 2. API-Based Models
    if provider == "OpenAI":
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    elif provider == "Anthropic":
        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    elif provider == "Google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

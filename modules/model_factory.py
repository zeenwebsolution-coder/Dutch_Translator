"""
model_factory.py — Unified Chat Model factory for multi-provider support.
Supports OpenAI, Anthropic (Claude), and Google (Gemini).
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.config import PROVIDERS, TEMPERATURE, MAX_TOKENS

def get_chat_model(provider: str, api_key: str):
    """
    Returns a LangChain Chat Model instance for the given provider.
    """
    conf = PROVIDERS.get(provider)
    if not conf:
        raise ValueError(f"Unsupported provider: {provider}")

    model_name = conf["chat_model"]

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
        raise ValueError(f"Unknown provider logic for: {provider}")

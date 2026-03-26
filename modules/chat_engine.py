"""
chat_engine.py — Interactive Dutch Assistant integrated with RAG/SENDERUM.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from modules.rag_engine import RAGStore, retrieve_tone_context
from modules.config import FORMALITY_OPTIONS

class DutchAssistant:
    def __init__(self, llm: Any, rag_store: RAGStore, domain: str, formality: str):
        self.llm = llm
        self.rag_store = rag_store
        self.domain = domain
        self.formality = formality
        self.formality_note = FORMALITY_OPTIONS.get(formality, "")

    def generate_response(self, user_query: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate a response using the persona and tone context."""
        # 1. Retrieve tone context for the user query
        tone_context = retrieve_tone_context([user_query], self.rag_store, top_k=3)
        
        # 2. Build the System Prompt
        system_prompt = f"""You are the SENDERUM Dutch Business Assistant.
DOMAIN: {self.domain}
FORMALITY: {self.formality_note}

TONE REFERENCE:
\"\"\"
{tone_context}
\"\"\"

INSTRUCTIONS:
- Answer primarily in Dutch (with English explanations if requested).
- Use the SENDERUM brand tone of voice.
- Be professional, helpful, and concise.
"""
        
        # 3. Assemble Messages
        messages = [SystemMessage(content=system_prompt)]
        
        # Add history
        for msg in chat_history[-5:]: # Keep last 5 for context
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
                
        # Add current query
        messages.append(HumanMessage(content=user_query))
        
        # 4. Invoke LLM
        try:
            response = self.llm.invoke(messages)
            return str(response.content)
        except Exception as e:
            return f"❌ Assistant Error: {e}"

"""
main.py — FastAPI backend for Dutch Translator
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import io
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Import your existing modules
from modules.model_factory import get_chat_model
from modules.translator import translate_batch, translate_single
from modules.rag_engine import build_rag_store, RAGStore
from modules.cache_manager import CacheManager
from modules.excel_handler import read_word_entries as read_excel_entries, write_translations as write_excel_translations, unique_words as unique_excel_words
from modules.word_handler import read_word_entries as read_docx_entries, write_translations as write_docx_translations, unique_words as unique_docx_words
from modules.zip_handler import extract_excel_files, pack_single, pack_zip
from modules.config import FORMALITY_OPTIONS, DOMAINS, PROVIDERS

app = FastAPI(title="Dutch Translator API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tone text (load once at startup)
TONE_TEXT = os.getenv("TONE_TEXT", "")
if not TONE_TEXT:
    try:
        from modules.tone_loader import load_tone_from_path
        TONE_TEXT = load_tone_from_path("SENDERUM-tone of voice (1).docx")
    except:
        TONE_TEXT = "Professional Dutch business translation. Use formal language."

# Global RAG store (load once)
rag_store = None

@app.on_event("startup")
async def startup_event():
    global rag_store
    # You can pre-load RAG if needed
    pass

# ── Request/Response Models ─────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: List[str]
    domain: str = "General Business"
    formality: str = "Formal (u-form)"
    provider: str = "OpenAI"
    api_key: str
    user_name: str = "default"

class TranslateResponse(BaseModel):
    translations: Dict[str, str]
    cache_hits: int = 0

class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []
    domain: str = "General Business"
    formality: str = "Formal (u-form)"
    provider: str = "OpenAI"
    api_key: str
    user_name: str = "default"

class ChatResponse(BaseModel):
    response: str

# ── API Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Dutch Translator API",
        "version": "1.0.0",
        "endpoints": {
            "/translate": "POST - Translate text",
            "/translate/file": "POST - Translate file",
            "/chat": "POST - Chat assistant",
            "/domains": "GET - List domains",
            "/formality": "GET - List formality options",
            "/providers": "GET - List providers"
        }
    }

@app.get("/domains")
async def get_domains():
    return {"domains": DOMAINS}

@app.get("/formality")
async def get_formality():
    return {"formality_options": list(FORMALITY_OPTIONS.keys())}

@app.get("/providers")
async def get_providers():
    return {"providers": list(PROVIDERS.keys())}

@app.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """Translate a list of English texts to Dutch"""
    try:
        # Initialize components
        cache = CacheManager(request.api_key, request.user_name)
        llm = get_chat_model(request.provider, request.api_key)
        
        is_local = "Local" in request.provider
        rag = None
        
        if not is_local and TONE_TEXT:
            rag = build_rag_store(TONE_TEXT, request.provider, request.api_key)
        
        # Translate
        results = translate_batch(
            request.text,
            llm,
            rag,
            request.domain,
            request.formality,
            cache=cache
        )
        
        return TranslateResponse(
            translations=results,
            cache_hits=sum(1 for word in request.text if cache.get(request.domain, request.formality, word))
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/file")
async def translate_file(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    user_name: str = Form("default"),
    domain: str = Form("General Business"),
    formality: str = Form("Formal (u-form)"),
    provider: str = Form("OpenAI")
):
    """Translate an Excel, Word, or ZIP file"""
    try:
        # Read file
        file_bytes = await file.read()
        
        # Extract sources
        from modules.zip_handler import extract_excel_files
        sources = extract_excel_files(file)
        
        # Initialize components
        cache = CacheManager(api_key, user_name)
        llm = get_chat_model(provider, api_key)
        is_local = "Local" in provider
        rag = None
        
        if not is_local and TONE_TEXT:
            rag = build_rag_store(TONE_TEXT, provider, api_key)
        
        translated_outputs = []
        
        for source in sources:
            file_name = source.name.lower()
            
            # Read entries
            if file_name.endswith(".xlsx"):
                sheet_entries = read_excel_entries(source.data)
                all_unique = unique_excel_words(sheet_entries)
            elif file_name.endswith(".docx"):
                sheet_entries = read_docx_entries(source.data)
                all_unique = unique_docx_words(sheet_entries)
            else:
                continue
            
            # Translate
            translation_cache = {}
            from modules.config import compute_batch_size
            batch_size = 10 if is_local else compute_batch_size(len(all_unique))
            
            for i in range(0, len(all_unique), batch_size):
                batch = all_unique[i:i + batch_size]
                result = translate_batch(batch, llm, rag, domain, formality, cache=cache)
                translation_cache.update(result)
            
            # Write output
            if file_name.endswith(".xlsx"):
                translated_bytes = write_excel_translations(source.data, sheet_entries, translation_cache)
            else:
                translated_bytes = write_docx_translations(source.data, translation_cache)
            
            translated_outputs.append((source, translated_bytes))
        
        # Create download
        if len(translated_outputs) == 1:
            source, t_bytes = translated_outputs[0]
            dl_bytes, dl_name = pack_single(t_bytes, source.name)
        else:
            dl_bytes, dl_name = pack_zip(translated_outputs)
        
        return StreamingResponse(
            io.BytesIO(dl_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={dl_name}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the Dutch assistant"""
    try:
        from modules.chat_engine import DutchAssistant
        
        cache = CacheManager(request.api_key, request.user_name)
        llm = get_chat_model(request.provider, request.api_key)
        
        is_local = "Local" in request.provider
        rag = None
        
        if not is_local and TONE_TEXT:
            rag = build_rag_store(TONE_TEXT, request.provider, request.api_key)
        
        assistant = DutchAssistant(llm, rag, request.domain, request.formality)
        response = assistant.generate_response(request.message, request.chat_history)
        
        return ChatResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
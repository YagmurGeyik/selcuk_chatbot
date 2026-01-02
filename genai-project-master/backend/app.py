# backend/app.py
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import OpenAI
from pymilvus import connections, Collection, utility

# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rules_qa")
VECTOR_FIELD = os.getenv("VECTOR_FIELD", "vector")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "3"))

# ✅ Alaka eşiği: düşükse "alakasız" say ve kaynak döndürme
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))  # 0.20 - 0.35 arası deneyebilirsin

# PDF/DOC servis ayarları
DOCS_DIR = Path(os.getenv("DOCS_DIR", "documents")).resolve()
DOCS_URL_PREFIX = os.getenv("DOCS_URL_PREFIX", "/docs")  # URL path prefix

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyanı kontrol et.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# FASTAPI
# -----------------------
app = FastAPI(title="Selcuk Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PROD: ["https://www.selcuk.edu.tr"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ PDF/DOC static serve
# documents/ klasörü varsa: http://localhost:8787/docs/<dosya.pdf>
if DOCS_DIR.exists():
    app.mount(DOCS_URL_PREFIX, StaticFiles(directory=str(DOCS_DIR)), name="docs")

# -----------------------
# MILVUS INIT
# -----------------------
def init_milvus():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(f"'{COLLECTION_NAME}' koleksiyonu bulunamadı. Önce ingest.py çalıştır.")

    col = Collection(COLLECTION_NAME)

    field_names = {f.name for f in col.schema.fields}
    if VECTOR_FIELD not in field_names:
        raise RuntimeError(f"VECTOR_FIELD='{VECTOR_FIELD}' koleksiyonda yok. Alanlar: {sorted(field_names)}")

    if "context" not in field_names:
        raise RuntimeError(f"Koleksiyonda 'context' alanı yok. Alanlar: {sorted(field_names)}")

    # index yoksa oluştur
    if len(col.indexes) == 0:
        col.create_index(
            field_name=VECTOR_FIELD,
            index_params={"metric_type": "IP", "index_type": "AUTOINDEX", "params": {}},
        )
        while True:
            progress = utility.index_building_progress(COLLECTION_NAME)
            if progress.get("indexed_rows", 0) == progress.get("total_rows", 1):
                break
            time.sleep(1)

    col.load()
    has_source = "source" in field_names
    has_header = "header" in field_names
    return col, has_source, has_header

collection, HAS_SOURCE, HAS_HEADER = init_milvus()

# -----------------------
# SCHEMAS
# -----------------------
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []  # [{"role":"user/assistant","content":"..."}]

class SourceItem(BaseModel):
    name: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []

# -----------------------
# RAG HELPERS
# -----------------------
GREETING_RE = re.compile(r"^\s*(merhaba|selam|günaydın|iyi\s*günler|iyi\s*akşamlar|hello|hi)\b", re.I)

def embed_text(text: str) -> List[float]:
    emb = client.embeddings.create(model=EMBED_MODEL, input=text)
    return emb.data[0].embedding

def search_milvus(query_text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    vec = embed_text(query_text)

    output_fields = ["context"]
    if HAS_SOURCE:
        output_fields.append("source")
    if HAS_HEADER:
        output_fields.append("header")

    results = collection.search(
        data=[vec],
        anns_field=VECTOR_FIELD,
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=output_fields,
    )

    hits: List[Dict[str, Any]] = []
    for hit in results[0]:
        hits.append(
            {
                "context": hit.entity.get("context"),
                "source": hit.entity.get("source") if HAS_SOURCE else None,
                "header": hit.entity.get("header") if HAS_HEADER else None,
                "score": float(hit.distance),  # IP: büyük daha iyi
            }
        )
    return hits

def build_context_text(contexts: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(contexts):
        header = (c.get("header") or "").strip()
        ctx = (c.get("context") or "").strip()
        if header:
            parts.append(f"{i+1}) {header}\n{ctx}")
        else:
            parts.append(f"{i+1}) {ctx}")
    return "\n\n".join(parts)

def ask_llm(question: str, contexts: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
    context_text = build_context_text(contexts)

    prompt = f"""
Aşağıdaki yönetmelik parçalarını kullanarak soruyu cevapla.

YÖNETMELİK PARÇALARI:
{context_text}

SORU: {question}

KURALLAR:
- Cevap Türkçe, kısa ve net olsun.
- Sadece Selçuk Üniversitesi ile ilgili yönetmelik/işlem sorularına cevap ver.
- Okulla ilgisizse aynen şunu söyle: "Üzgünüm yalnızca Selçuk Üniversitesi ile ilgili sorulara cevap verebilirim."
- Cevapta dosya adı / PDF adı / köşeli parantezli kaynak etiketi yazma.
- Gereksiz uzun maddeler yazma; en fazla 5 madde.

YANIT:
""".strip()

    messages = [{"role": "system", "content": "Sen Selçuk Üniversitesi öğrenci işlerinde uzman bir asistansın."}]

    for m in history[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
    )
    answer = completion.choices[0].message.content.strip()

    answer = re.sub(r"\[[^\]]+\.pdf\]", "", answer, flags=re.I).strip()
    return answer

def extract_sources(contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    seen = set()

    for c in contexts:
        raw = (c.get("source") or "").strip()
        if not raw:
            continue

        name = os.path.basename(raw)
        if name in seen:
            continue
        seen.add(name)

        file_path = (DOCS_DIR / name)
        if file_path.exists() and DOCS_DIR.exists():
            url = f"{DOCS_URL_PREFIX}/{name}"
        else:
            url = ""

        sources.append({"name": name, "url": url})

    return sources

# -----------------------
# ENDPOINTS
# -----------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = (req.message or "").strip()
    if not q:
        return ChatResponse(answer="Bir soru yazar mısın?", sources=[])

    # Selamlaşma: Milvus/OpenAI çağırmadan sabit cevap
    if GREETING_RE.match(q):
        return ChatResponse(
            answer="Merhaba 👋 Selçuk Üniversitesi ile ilgili bir sorunuz varsa yardımcı olabilirim.",
            sources=[],
        )

    contexts = search_milvus(q, top_k=TOP_K)
    if not contexts:
        return ChatResponse(
            answer="Bu konuda yönetmeliklerde net bir bilgi bulamadım. Soruyu biraz daha detaylandırır mısın?",
            sources=[],
        )

    # ✅ Alakasız soru filtresi: skor düşükse kaynak da dönme, LLM'e de gitme
    best_score = float(contexts[0].get("score", 0.0))
    if best_score < MIN_SCORE:
        return ChatResponse(
            answer="Üzgünüm yalnızca Selçuk Üniversitesi ile ilgili sorulara cevap verebilirim.",
            sources=[],
        )

    answer = ask_llm(q, contexts, req.history)
    sources = extract_sources(contexts)

    return ChatResponse(answer=answer, sources=sources)

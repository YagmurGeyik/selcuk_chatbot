import os
import time
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from pymilvus import Collection, connections, utility
from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Senin .env dosyanda COLLECTION_NAME var.
# Bazı örnek projelerde MILVUS_COLLECTION kullanılıyor.
COLLECTION_NAME = (
    os.getenv("COLLECTION_NAME")
    or os.getenv("MILVUS_COLLECTION")
    or "rules_qa"
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

TOP_K = int(os.getenv("TOP_K", "3"))

# Selamlaşma / kısa mesaj yakalama
GREETING_KEYWORDS = {"merhaba", "selam", "hello", "hi", "iyi günler", "iyi akşamlar", "günaydın"}

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Milvus Q&A Chatbot", page_icon="💬", layout="wide")
st.title("💬 Üniversite Soru-Cevap Asistanı")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY bulunamadı. .env dosyanı kontrol et.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# INIT MILVUS
# -----------------------
@st.cache_resource
def init_milvus():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        st.error(f"'{COLLECTION_NAME}' koleksiyonu bulunamadı. Önce 'python ingest.py' çalıştır.")
        st.stop()

    col = Collection(COLLECTION_NAME)

    # Şemadaki alanları oku (header var mı? vektör alan adı ne?)
    field_names = {f.name for f in col.schema.fields}

    # Projene göre vektör alanı bazen "vector", bazen "vector_context" oluyor.
    if "vector_context" in field_names:
        vector_field = "vector_context"
    elif "vector" in field_names:
        vector_field = "vector"
    else:
        st.error(f"Koleksiyon şemasında vektör alanı bulunamadı. Bulunan alanlar: {sorted(field_names)}")
        st.stop()

    has_header = "header" in field_names
    has_source = "source" in field_names
    has_context = "context" in field_names

    if not has_context:
        st.error(f"Koleksiyon şemasında 'context' alanı yok. Bulunan alanlar: {sorted(field_names)}")
        st.stop()

    # Index yoksa oluştur (vektör alanına göre)
    if len(col.indexes) == 0:
        with st.spinner("🔧 Index oluşturuluyor..."):
            col.create_index(
                field_name=vector_field,
                index_params={"metric_type": "IP", "index_type": "AUTOINDEX", "params": {}}
            )
            # index oluşana kadar bekle
            while True:
                progress = utility.index_building_progress(COLLECTION_NAME)
                if progress.get("indexed_rows", 0) == progress.get("total_rows", 1):
                    break
                time.sleep(1)

    col.load()
    return col, vector_field, has_header, has_source

collection, VECTOR_FIELD, HAS_HEADER, HAS_SOURCE = init_milvus()

# -----------------------
# RAG
# -----------------------
def embed_text(text: str):
    emb = client.embeddings.create(model=EMBED_MODEL, input=text)
    return emb.data[0].embedding

def search_milvus(query_text: str, top_k: int = TOP_K):
    query_vector = embed_text(query_text)

    # output_fields şemaya göre seçilsin (header yoksa istemeyelim)
    output_fields = ["context"]
    if HAS_HEADER:
        output_fields.append("header")
    if HAS_SOURCE:
        output_fields.append("source")

    results = collection.search(
        data=[query_vector],
        anns_field=VECTOR_FIELD,
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=output_fields
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "context": hit.entity.get("context"),
            "header": hit.entity.get("header") if HAS_HEADER else None,
            "source": hit.entity.get("source") if HAS_SOURCE else None,
            "score": float(hit.distance),
        })
    return hits

def ask_gpt(question: str, contexts):
    # ✅ Kullanıcıya pdf göstermeyeceğiz → prompt içinde de dosya adı istemiyoruz
    # Contextleri sadece içerik olarak veriyoruz.
    parts = []
    for i, c in enumerate(contexts):
        if c.get("header"):
            parts.append(f"{i+1}) {c['header']}\n{c['context']}")
        else:
            parts.append(f"{i+1}) {c['context']}")
    context_text = "\n\n".join(parts)

    prompt = f"""
Aşağıdaki yönetmelik parçalarını kullanarak soruyu cevapla.

YÖNETMELİK PARÇALARI:
{context_text}

SORU: {question}

KURALLAR:
- Cevap Türkçe, kısa ve net olsun.
- Sadece Selçuk Üniversitesi ile ilgili yönetmelik/işlem sorularına cevap ver.
- Okulla ilgisizse aynen şunu söyle: "Üzgünüm yalnızca Selçuk Üniversitesi ile ilgili sorulara cevap verebilirim."
- Cevapta dosya adı, PDF adı, köşeli parantez (örn. [xxx.pdf]) veya kaynak etiketi yazma.
- Selamlaşma gibi mesajlarda kullanıcıyı yönlendir.

YANIT:
"""

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Sen Selçuk Üniversitesi öğrenci işlerinde uzman bir asistansın."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()

# -----------------------
# UI CHAT
# -----------------------
st.markdown("Sorunu yaz 👇 Yönetmeliklerden bulup cevaplayacağım.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("🎓 Soru:", placeholder="Örn: Ders kaydı nasıl yapılır?")

if st.button("🚀 Gönder") and question:
    q = question.strip()
    q_lower = q.lower().strip()

    with st.spinner("Yanıt hazırlanıyor..."):
        # 1) Selamlaşma yakala (Milvus araması yapmadan)
        if q_lower in GREETING_KEYWORDS:
            answer = "Merhaba 👋 Selçuk Üniversitesi ile ilgili bir sorunuz varsa yardımcı olabilirim."
        else:
            contexts = search_milvus(q, top_k=TOP_K)

            # Context gelmezse (çok nadir) güvenli cevap
            if not contexts:
                answer = "Bu konuda yönetmeliklerde net bir bilgi bulamadım. Sorunu biraz daha detaylandırabilir misin?"
            else:
                answer = ask_gpt(q, contexts)

        st.session_state.chat_history.append(("👤", q))
        st.session_state.chat_history.append(("🤖", answer))

for role, text in st.session_state.chat_history:
    if role == "👤":
        st.markdown(f"**{role}**: {text}")
    else:
        st.success(f"**{role}**: {text}")

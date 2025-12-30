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
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "rules_qa")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Milvus Q&A Chatbot", page_icon="💬", layout="wide")
st.title("💬 Üniversite Soru-Cevap Asistanı (Milvus + GPT-4o)")

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

    # index yoksa oluştur
    if len(col.indexes) == 0:
        with st.spinner("🔧 Index oluşturuluyor..."):
            col.create_index(
                field_name="vector_context",
                index_params={"metric_type": "IP", "index_type": "AUTOINDEX", "params": {}}
            )
            while True:
                progress = utility.index_building_progress(COLLECTION_NAME)
                if progress.get("indexed_rows", 0) == progress.get("total_rows", 1):
                    break
                time.sleep(1)

    col.load()
    return col

collection = init_milvus()

# -----------------------
# RAG
# -----------------------
def embed_text(text: str):
    emb = client.embeddings.create(model=EMBED_MODEL, input=text)
    return emb.data[0].embedding

def search_milvus(query_text: str, top_k: int = 3):
    query_vector = embed_text(query_text)

    results = collection.search(
        data=[query_vector],
        anns_field="vector_context",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["source", "header", "context"]
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "source": hit.entity.get("source"),
            "header": hit.entity.get("header"),
            "context": hit.entity.get("context"),
            "score": float(hit.distance),
        })
    return hits

def ask_gpt(question: str, contexts):
    # okul dışı sorular filtresi (prompt içinde de var)
    context_text = "\n\n".join(
        [f"{i+1}) [{c['source']}] {c['header']}\n{c['context']}" for i, c in enumerate(contexts)]
    )

    prompt = f"""
Aşağıdaki yönetmelik parçalarını kullanarak soruyu cevapla.

KAYNAKLAR:
{context_text}

SORU: {question}

KURALLAR:
- Cevap Türkçe, kısa ve net olsun.
- Sadece Selçuk Üniversitesi ile ilgili yönetmelik/işlem sorularına cevap ver.
- Okulla ilgisizse aynen şunu söyle: "Üzgünüm yalnızca Selçuk Üniversitesi ile ilgili sorulara cevap verebilirim."
- Cevapta en az 1 kaynak göster: [dosya_adı] biçiminde.

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
    return completion.choices[0].message.content

# -----------------------
# UI CHAT
# -----------------------
st.markdown("Sorunu yaz 👇 Yönetmeliklerden bulup cevaplayacağım.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("🎓 Soru:", placeholder="Örn: Ders kaydı nasıl yapılır?")

if st.button("🚀 Gönder") and question:
    with st.spinner("Yanıt aranıyor..."):
        contexts = search_milvus(question, top_k=3)
        answer = ask_gpt(question, contexts)

        st.session_state.chat_history.append(("👤", question))
        st.session_state.chat_history.append(("🤖", answer))

for role, text in st.session_state.chat_history:
    if role == "👤":
        st.markdown(f"**{role}**: {text}")
    else:
        st.success(f"**{role}**: {text}")

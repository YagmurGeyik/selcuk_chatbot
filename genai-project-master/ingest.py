import os
import re
from tqdm import tqdm
from dotenv import load_dotenv

from pypdf import PdfReader
import docx

from openai import OpenAI
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType, Collection
)

load_dotenv()

# -----------------------
# CONFIG
# -----------------------
DOCS_DIR = os.getenv("DOCS_DIR", "documents")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "rules_qa")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))

# Eğer true yaparsan koleksiyon silinip sıfırdan yüklenir
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "false").lower() == "true"

# batch insert (hız)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

# chunk ayarları
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# -----------------------
# UTILS
# -----------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return clean_text(" ".join((p.extract_text() or "") for p in reader.pages))

def read_docx(path: str) -> str:
    d = docx.Document(path)
    return clean_text(" ".join(p.text for p in d.paragraphs))

def ensure_collection():
    if RESET_COLLECTION and utility.has_collection(COLLECTION_NAME):
        print("🧹 RESET_COLLECTION=true -> koleksiyon siliniyor:", COLLECTION_NAME)
        utility.drop_collection(COLLECTION_NAME)

    if not utility.has_collection(COLLECTION_NAME):
        print("📦 Yeni koleksiyon oluşturuluyor:", COLLECTION_NAME)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

            # Kaynak göstermek için
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),

            # Chunk başlığı / etiketi (dosya + chunk no)
            FieldSchema(name="header", dtype=DataType.VARCHAR, max_length=512),

            # Metin parçası
            FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535),

            # Embedding alanı (test.py ile uyumlu)
            FieldSchema(name="vector_context", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        ]

        schema = CollectionSchema(fields, "Selçuk Üniversitesi Yönetmelikleri - RAG")
        collection = Collection(COLLECTION_NAME, schema)

        collection.create_index(
            field_name="vector_context",
            index_params={"metric_type": "IP", "index_type": "AUTOINDEX", "params": {}}
        )
        print("✅ Index oluşturuldu.")
    else:
        collection = Collection(COLLECTION_NAME)

    collection.load()
    return collection

# -----------------------
# MAIN
# -----------------------
def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyanı kontrol et.")

    print("🔌 Milvus'a bağlanılıyor...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    collection = ensure_collection()

    client = OpenAI(api_key=OPENAI_API_KEY)

    # 1) Dosyaları oku ve chunk'la
    records = []  # (source, header, context)
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)

        if file.lower().endswith(".pdf"):
            text = read_pdf(path)
        elif file.lower().endswith(".docx"):
            text = read_docx(path)
        else:
            continue

        chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, ch in enumerate(chunks, start=1):
            header = f"{file} - Parça {i}"
            records.append((file, header, ch))

    print(f"📄 Toplam {len(records)} parça oluşturuldu")

    # 2) Embed + batch insert
    sources_batch, headers_batch, contexts_batch, vectors_batch = [], [], [], []

    for (source, header, chunk) in tqdm(records, desc="Embedding + Insert"):
        emb = client.embeddings.create(model=EMBED_MODEL, input=chunk).data[0].embedding

        sources_batch.append(source)
        headers_batch.append(header)
        contexts_batch.append(chunk)
        vectors_batch.append(emb)

        if len(sources_batch) >= BATCH_SIZE:
            collection.insert([sources_batch, headers_batch, contexts_batch, vectors_batch])
            sources_batch, headers_batch, contexts_batch, vectors_batch = [], [], [], []

    # kalanlar
    if sources_batch:
        collection.insert([sources_batch, headers_batch, contexts_batch, vectors_batch])

    collection.flush()
    collection.load()

    # küçük bilgi
    try:
        print("📌 Koleksiyon kayıt sayısı:", collection.num_entities)
    except Exception:
        pass

    print("✅ Veri yükleme tamamlandı!")

if __name__ == "__main__":
    main()

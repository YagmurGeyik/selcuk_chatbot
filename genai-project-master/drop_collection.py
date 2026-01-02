import os
from dotenv import load_dotenv
from pymilvus import connections, utility

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rules_qa")

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    print(f"✅ Collection silindi: {COLLECTION_NAME}")
else:
    print(f"ℹ️ Collection bulunamadı: {COLLECTION_NAME}")

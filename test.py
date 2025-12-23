import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "legal_docs"
MODEL = "all-MiniLM-L6-v2"

embed = SentenceTransformer(MODEL)

client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DIR,
        chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False
    )
)

col = client.get_collection(COLLECTION_NAME)

question = "Đối tượng không chịu thuế"

query_embedding = embed.encode(question).tolist()

res = col.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

for i, (doc, meta, dist) in enumerate(
    zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
):
    print(f"\n===== RESULT {i} =====")
    print("DISTANCE:", dist)
    print("FILE    :", meta.get("file"))
    print("ARTICLE :", meta.get("article"))
    print(doc[:500])

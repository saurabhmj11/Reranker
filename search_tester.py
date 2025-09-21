import json
import pickle
import faiss
import numpy as np

# Paths
BASE_DIR = r"C:\Users\Saura\Downloads\book images\reranker\data"
INDEX_PATH = BASE_DIR + r"\index.faiss"
META_PATH = BASE_DIR + r"\index.pkl"
SOURCES_PATH = BASE_DIR + r"\sources.json"

# Load FAISS index and metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# Load sources.json
with open(SOURCES_PATH, "r", encoding="utf-8") as f:
    sources = {src["id"]: src for src in json.load(f)}

# Example: search query
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "What are ISO 10218 safety standards?"
q_emb = model.encode([query], convert_to_numpy=True)

# Search top 5 results
D, I = index.search(q_emb, k=5)

# Print results
print(f"\n🔎 Query: {query}\n")
for rank, (score, idx) in enumerate(zip(D[0], I[0])):
    if idx == -1:
        continue
    chunk = metadata[idx + 1]  # chunk_id starts from 1 in DB
    source_id = chunk["source_id"]
    src_info = sources.get(source_id, {})

    print(f"Result {rank+1}: (Score: {score:.4f})")
    print(f"File: {chunk['file']}")
    if src_info:
        print(f"Title: {src_info.get('title')}")
        print(f"URL: {src_info.get('url')}")
    print(f"Text: {chunk['text'][:500]}...\n")

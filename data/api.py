import os
import pickle
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from fastapi import FastAPI
from pydantic import BaseModel

# ===================================================================
# 1. DATA LOADING AND MODEL INITIALIZATION
# ===================================================================

# Get the absolute path of the directory where this script (api.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full, absolute paths to your data files
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "index.pkl")
SOURCES_PATH = os.path.join(BASE_DIR, "sources.json")

# Config
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALPHA = 0.6          # Weight for vector vs. BM25 scores
CANDIDATE_POOL = 20  # How many initial FAISS hits to consider for reranking

# Load metadata (list aligned with FAISS embeddings)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# Prepare BM25 model
tokenized_corpus = [doc["text"].split() for doc in metadata]
bm25 = BM25Okapi(tokenized_corpus)

# Load FAISS index and sentence embedding model
index = faiss.read_index(INDEX_PATH)
embedder = SentenceTransformer(EMB_MODEL)

# Load sources mapping
with open(SOURCES_PATH, "r", encoding="utf-8") as f:
    sources = {s["id"]: s for s in json.load(f)}


# ===================================================================
# 2. SEARCH AND RERANKING FUNCTIONS
# ===================================================================

def hybrid_rerank(query, top_k=5):
    """
    Performs a hybrid search using FAISS for initial retrieval and BM25 for reranking.
    """
    # 1. Initial retrieval with FAISS (Vector Search)
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_embedding, CANDIDATE_POOL)

    # Filter out invalid indices (-1)
    valid_indices = [int(i) for i in indices[0] if i != -1]
    if not valid_indices:
        return []
    
    vec_scores = np.array([float(d) for d in distances[0][:len(valid_indices)]])

    # 2. Score the candidates with BM25 (Keyword Search)
    query_tokens = query.split()
    bm25_all_scores = bm25.get_scores(query_tokens)
    bm25_scores = np.array([float(bm25_all_scores[i]) for i in valid_indices])

    # 3. Normalize and combine scores for reranking
    def normalize_scores(scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    vec_scores_norm = normalize_scores(vec_scores)
    bm25_scores_norm = normalize_scores(bm25_scores)

    # Combine scores: higher is better for both after normalization
    # Note: FAISS distances are lower is better, so we use (1 - normalized_distance)
    combined_scores = ALPHA * (1 - vec_scores_norm) + (1 - ALPHA) * bm25_scores_norm

    # 4. Assemble and sort the final results
    results = []
    for i, idx in enumerate(valid_indices):
        meta = metadata[idx]
        results.append({
            "combined_score": float(combined_scores[i]),
            "text": meta["text"],
            "file": meta["file"],
            "vec_score": float(vec_scores[i]),
            "bm25_score": float(bm25_scores[i]),
        })
    
    # Sort by the combined score in descending order and return top_k
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:top_k]


# ===================================================================
# 3. API SERVER SETUP (FastAPI)
# ===================================================================

# This is the "app" object uvicorn needs to run
app = FastAPI(
    title="Hybrid Search API",
    description="An API that uses FAISS and BM25 for hybrid search.",
    version="1.0.0",
)

# Pydantic model to define the structure of the request body
class AskRequest(BaseModel): # Renamed from SearchRequest
    query: str
    top_k: int = 5

# Root endpoint for a basic health check
@app.get("/")
def read_root():
    return {"status": "API is running"}

# Main search endpoint
@app.post("/ask") # <-- THE FIX IS HERE
def ask(request: AskRequest): # <-- Renamed function and request model
    """
    Performs a hybrid search on the document corpus.
    """
    search_results = hybrid_rerank(query=request.query, top_k=request.top_k)
    return {"results": search_results}
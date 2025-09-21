import sqlite3
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
from rank_bm25 import BM25Okapi

# Paths
BASE_DIR = r"C:\Users\Saura\Downloads\book images\reranker\data"
DB_PATH = os.path.join(BASE_DIR, "chunks.db")
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "index.pkl")
BM25_PATH = os.path.join(BASE_DIR, "bm25_corpus.pkl")

def main():
    # Load chunks from DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, source_id, file, chunk_index, text FROM chunks")
    rows = cur.fetchall()
    conn.close()

    print(f"📄 Loaded {len(rows)} chunks from {DB_PATH}")

    # Prepare data
    ids = []
    texts = []
    metadata = []   # stays aligned with FAISS embeddings

    for row in rows:
        chunk_id, source_id, file, chunk_index, text = row
        ids.append(chunk_id)
        texts.append(text)
        metadata.append({
            "chunk_id": chunk_id,
            "source_id": source_id,
            "file": file,
            "chunk_index": chunk_index,
            "text": text
        })

    # Load embedding model
    print("⚡ Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index + metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index saved to {INDEX_PATH}")
    print(f"✅ Metadata saved to {META_PATH}")

    # Build BM25 corpus
    print("⚡ Building BM25 corpus...")
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"✅ BM25 corpus saved to {BM25_PATH}")

if __name__ == "__main__":
    main()

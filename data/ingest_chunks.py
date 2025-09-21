# ingest_chunks.py
import sqlite3
from PyPDF2 import PdfReader
import json
import os
import re

# --- Paths ---
EXTRACT_DIR = r"C:\Users\Saura\Downloads\book images\reranker\data\industrial-safety-pdfs"  # extracted PDFs folder
SOURCES_JSON = "sources.json"
DB_PATH = "chunks.db"

# --- Helpers ---
def clean_text(text: str) -> str:
    """Collapse whitespace and trim."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, max_words: int = 120):
    """Split long text into smaller chunks of ~max_words."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# --- Main ---
def main():
    # Make sure folder exists
    if not os.path.isdir(EXTRACT_DIR):
        print(f"❌ ERROR: Folder not found: {EXTRACT_DIR}")
        return

    # Load sources.json
    if not os.path.exists(SOURCES_JSON):
        print(f"❌ ERROR: sources.json not found at {SOURCES_JSON}")
        return

    with open(SOURCES_JSON, "r", encoding="utf-8") as f:
        sources_data = json.load(f)

    # Initialize database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute("""
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER,
        file TEXT,
        chunk_index INTEGER,
        text TEXT
    )
    """)

    # Process each PDF
    for src in sources_data:
        file_path = os.path.join(EXTRACT_DIR, src["file"])
        print(f"📄 Processing: {src['file']}")

        if not os.path.exists(file_path):
            print(f"⚠️ Skipping {src['file']} (not found in {EXTRACT_DIR})")
            continue

        try:
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

            # Split into paragraphs
            paragraphs = [
                clean_text(p)
                for p in full_text.split("\n")
                if len(p.strip()) > 40
            ]

            # Further split into chunks
            chunks = []
            for p in paragraphs:
                if len(p.split()) > 120:
                    chunks.extend(chunk_text(p))
                else:
                    chunks.append(p)

            # Insert into DB
            for idx, chunk in enumerate(chunks):
                cur.execute(
                    "INSERT INTO chunks (source_id, file, chunk_index, text) VALUES (?, ?, ?, ?)",
                    (src["id"], src["file"], idx, chunk)
                )

            print(f"✅ Saved {len(chunks)} chunks from {src['file']}")

        except Exception as e:
            print(f"❌ Error reading {src['file']}: {e}")

    conn.commit()
    conn.close()
    print(f"\n🎉 Done! All chunks stored in {DB_PATH}")


if __name__ == "__main__":
    main()

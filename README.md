# Reranker Project

This project implements a search reranking system using FAISS and Sentence Transformers. It allows users to build a FAISS index, perform search queries, and rerank results for better accuracy.

---

## Features
- Build and manage FAISS indices.
- Rerank search results using Sentence Transformers.
- Includes BM25 corpus support.

---

## Prerequisites
- Python 3.10 or higher
- Virtual environment (recommended)

---

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-folder>


    Install Dependencies: Create and activate a virtual environment:
      python -m venv venv
venv\Scripts\activate  # On Windows

Install required packages:
pip install -r requirements.txt

Generate FAISS Index: If data/index.faiss and data/index.pkl are missing, generate them:
python data/build_index.py

Usage
Run the API: Start the API server:
python data/api.py

Test Search: Use search_tester.py to test search functionality:
python search_tester.py

python evalute for evaluation

File Structure
data/: Contains data files and scripts for index building and API.
build_index.py: Script to generate FAISS index.
bm25_corpus.pkl: Preprocessed BM25 corpus.
api.py: API implementation.
script/: Additional utility scripts.
requirements.txt: Python dependencies.
Contributing
Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request.
License

This project is licensed under the MIT License. See LICENSE for details.

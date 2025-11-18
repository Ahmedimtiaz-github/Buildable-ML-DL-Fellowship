# build_rag_index.py
"""
Build RAG embeddings and FAISS index.

Saves:
  - models/rag_embeddings.npy
  - models/rag_chunks_meta.jsonl
  - models/rag_faiss.index

Usage (PowerShell):
  .\venv\Scripts\Activate.ps1
  python .\build_rag_index.py

Notes/assumptions:
 - Input chunks expected at: data/processed/rag/chunks.jsonl
   Each line is a JSON object with at least a 'text' or 'content' field.
 - Default embedding model: sentence-transformers/all-MiniLM-L6-v2
 - FAISS Index: IndexFlatIP (requires normalized vectors)
"""

import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import sys

# --- Config (change if needed) ---
REPO_ROOT = Path(__file__).resolve().parent
CHUNKS_FILE = REPO_ROOT / "data" / "processed" / "rag" / "chunks.jsonl"
OUT_DIR = REPO_ROOT / "models"
EMB_OUT = OUT_DIR / "rag_embeddings.npy"
META_OUT = OUT_DIR / "rag_chunks_meta.jsonl"
FAISS_OUT = OUT_DIR / "rag_faiss.index"
EMBEDDING_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")  # sentence-transformers model

os.makedirs(OUT_DIR, exist_ok=True)

def load_chunks(p):
    chunks = []
    meta = []
    if not p.exists():
        print(f"[ERROR] chunks file not found: {p}", file=sys.stderr)
        sys.exit(1)
    with open(p, "r", encoding="utf8") as fh:
        for i, line in enumerate(fh):
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] skipping invalid json line {i}: {e}")
                continue
            # support multiple possible field names
            text = obj.get("text") or obj.get("content") or obj.get("chunk") or obj.get("body") or ""
            if not text:
                # if no text field, stringify object (last resort)
                text = json.dumps(obj, ensure_ascii=False)
            chunks.append(text)
            # ensure each meta has an id field
            if "id" not in obj:
                obj["id"] = f"chunk_{i}"
            meta.append(obj)
    return chunks, meta

def build_embeddings(chunks, model_name):
    print(f"[INFO] loading embedding model '{model_name}' (this may download weights first time)...")
    model = SentenceTransformer(model_name)
    # encode
    print("[INFO] encoding chunks ...")
    emb = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, batch_size=32)
    return emb, model

def build_faiss_index(embeddings, faiss_path):
    # normalize embeddings for IP search
    print("[INFO] normalizing embeddings for IndexFlatIP similarity (faiss.normalize_L2) ...")
    xb = embeddings.copy()
    faiss.normalize_L2(xb)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    faiss.write_index(index, str(faiss_path))
    return index

def save_meta(meta_list, meta_path):
    with open(meta_path, "w", encoding="utf8") as fh:
        for m in meta_list:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

def main():
    print("[STEP] load chunks")
    chunks, meta = load_chunks(CHUNKS_FILE)
    print(f"[INFO] loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    emb, model = build_embeddings(chunks, EMBEDDING_MODEL)
    print(f"[INFO] embeddings shape: {emb.shape}  dtype={emb.dtype}")

    print(f"[STEP] saving embeddings to {EMB_OUT}")
    np.save(EMB_OUT, emb)

    print(f"[STEP] saving metadata to {META_OUT}")
    save_meta(meta, META_OUT)

    print("[STEP] building faiss index")
    index = build_faiss_index(emb, FAISS_OUT)
    print(f"[INFO] saved faiss index to {FAISS_OUT}")

    # Print summary and sample
    print("\n--- Summary ---")
    print(f"num_chunks: {len(chunks)}")
    print(f"embedding_dim: {emb.shape[1]}")
    sample_meta = meta[0] if len(meta) else {}
    print("sample_meta (first chunk):")
    print(json.dumps(sample_meta, ensure_ascii=False, indent=2)[:1000])
    print("\nDone.")

if __name__ == "__main__":
    main()

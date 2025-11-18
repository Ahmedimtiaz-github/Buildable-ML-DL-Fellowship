Phase D — RAG quick-start (local, PowerShell)

How to run (from repo root):

1) Activate venv (you already have .\venv):
   .\venv\Scripts\Activate.ps1

2) Install packages:
   pip install --upgrade pip
   pip install sentence-transformers faiss-cpu numpy openai transformers torch

3) Build embeddings + FAISS index:
   python .\build_rag_index.py

   Output files created:
    - models\rag_embeddings.npy
    - models\rag_chunks_meta.jsonl
    - models\rag_faiss.index

4) Run demo:
   # Option A: Use OpenAI (set API key in current session)
   $env:OPENAI_API_KEY = "sk-..."
   python .\src\rag_demo.py

   # Option B: No OpenAI key (uses local transformers)
   python .\src\rag_demo.py

5) Report output:
   - reports\rag_demo_output.txt

Note: If your chunks.jsonl uses a different field name than 'text' or 'content', update build_rag_index.py load_chunks().

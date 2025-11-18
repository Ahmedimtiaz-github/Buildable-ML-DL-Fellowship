# src/rag_tool.py
"""
RAG retrieval + QA wrapper with robust OpenAI fallback.

Behavior:
 - If OPENAI_API_KEY is not set -> uses local transformers LLM.
 - If OPENAI_API_KEY is set -> attempts to call OpenAI (modern client preferred).
   If any authentication or client error occurs, logs the issue and falls back to local LLM.
 - Deduplicates retrieved snippets (whitespace-normalized).
 - Prompt instructs model to use only context and list source ids.

This file is safe to run locally without an OpenAI key.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

# Lazy/optional imports handled inside functions to avoid heavy import-time failures
try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
EMB_FILE = MODELS_DIR / "rag_embeddings.npy"
META_FILE = MODELS_DIR / "rag_chunks_meta.jsonl"
FAISS_FILE = MODELS_DIR / "rag_faiss.index"
EMBED_MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

_embed_model = None
_index = None
_meta = None

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Use ONLY the following CONTEXT to answer the user's question. "
    "If the answer is not contained in the context, reply: \"I don't know.\" Do not hallucinate. "
    "Answer concisely (1-3 short paragraphs). At the end, list the source chunk ids you used in this format: [id1, id2].\n\n"
    "<<CONTEXT>>\n\nUser: {question}\nAssistant:"
)

def _ensure_dependencies():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available. Install with: pip install sentence-transformers")
    if faiss is None:
        raise RuntimeError("faiss not available. Install with: pip install faiss-cpu")

def _load_meta():
    global _meta
    if _meta is None:
        if not META_FILE.exists():
            raise FileNotFoundError(f"Missing meta file: {META_FILE}")
        with open(META_FILE, "r", encoding="utf8") as fh:
            _meta = [json.loads(l) for l in fh if l.strip()]
    return _meta

def _load_index_and_model():
    global _embed_model, _index
    if _embed_model is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed.")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    if _index is None:
        if not FAISS_FILE.exists():
            raise FileNotFoundError(f"Missing FAISS index file: {FAISS_FILE}")
        if faiss is None:
            raise RuntimeError("faiss not installed.")
        _index = faiss.read_index(str(FAISS_FILE))
    return _embed_model, _index

def _normalize_text_for_key(text: str) -> str:
    return " ".join(text.split()).strip().lower()

def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    _ensure_dependencies()
    model, index = _load_index_and_model()
    meta = _load_meta()

    qvec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, top_k * 6)
    seen_keys = set()
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        text = m.get("text") or m.get("content") or m.get("chunk") or ""
        snippet = text.strip()
        key = _normalize_text_for_key(snippet)[:1000]
        if not key:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        results.append({
            "id": m.get("id", f"chunk_{idx}"),
            "score": float(score),
            "text_snippet": snippet[:3000]
        })
        if len(results) >= top_k:
            break
    return results

# -----------------------
# LLM backends
# -----------------------

def _generate_with_openai(prompt: str, llm_key: str = None, max_tokens: int = 256) -> str:
    """
    Try modern OpenAI client first. If any auth/client error occurs, raise an Exception to allow fallback.
    """
    llm_key = llm_key or os.environ.get("OPENAI_API_KEY")
    if not llm_key:
        raise RuntimeError("No OpenAI API key provided")

    # Try the new client if available
    try:
        from openai import OpenAI
        client = OpenAI(api_key=llm_key)
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            resp = client.chat.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return getattr(resp.choices[0].message, "content", resp.choices[0].message).strip()
    except Exception as e_new_client:
        # Fallback to legacy openai package interface
        try:
            import openai as openai_legacy
            openai_legacy.api_key = llm_key
            resp = openai_legacy.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip() if hasattr(resp.choices[0], "message") else resp.choices[0].text.strip()
        except Exception as e_legacy:
            raise RuntimeError(f"OpenAI generation failed. new_client_err={e_new_client}; legacy_client_err={e_legacy}")

def _generate_with_local_transformers(prompt: str, local_model: str = None, max_new_tokens: int = 80) -> str:
    """
    Safe local generation using transformers, with truncation + error handling.
    """
    import os
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except Exception as e:
        return f"[ERROR] transformers not installed: {str(e)}"

    try:
        if local_model is None:
            local_model = os.environ.get("RAG_LOCAL_MODEL", "distilgpt2")

        tok = AutoTokenizer.from_pretrained(local_model)

        # Truncate to tokenizer max length to avoid index errors
        max_len = tok.model_max_length
        encoded = tok(prompt, truncation=True, max_length=max_len)
        prompt = tok.decode(encoded["input_ids"], skip_special_tokens=True)

        model = AutoModelForCausalLM.from_pretrained(local_model)
        gen = pipeline("text-generation", model=model, tokenizer=tok, device=-1)

        out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0].get("generated_text", "")

        # Remove prompt prefix if present
        if out.startswith(prompt):
            return out[len(prompt):].strip()

        return out.strip()

    except Exception as e:
        return f"[ERROR] Local LLM failed: {str(e)}"
def rag_qa_tool(query: str, top_k: int = 5, llm: str = "auto", llm_key: str = None, max_context_chars: int = 3000) -> Dict:
    """
    Top-level RAG wrapper.
    llm: 'auto' (default), 'openai', or 'local'
    - 'auto': uses OpenAI only if OPENAI_API_KEY set and valid, otherwise local.
    """
    # 1) retrieval
    retrieved = retrieve(query, top_k=top_k)

    # 2) build context
    parts = []
    total = 0
    for r in retrieved:
        text = r["text_snippet"]
        if total + len(text) > max_context_chars:
            remaining = max_context_chars - total
            if remaining > 50:
                parts.append(text[:remaining])
                total += remaining
            break
        parts.append(text)
        total += len(text)
    context_text = "\n\n---\n\n".join(parts) if parts else ""
    prompt = PROMPT_TEMPLATE.replace("<<CONTEXT>>", context_text).format(question=query)

    # 3) choose LLM
    chosen_llm = llm
    if llm == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            chosen_llm = "openai"
        else:
            chosen_llm = "local"

    answer = None
    if chosen_llm == "openai":
        try:
            answer = _generate_with_openai(prompt, llm_key)
        except Exception as e:
            print(f"[WARN] OpenAI generation failed ({str(e)}). Falling back to local LLM.", flush=True)
            try:
                answer = _generate_with_local_transformers(prompt)
            except Exception as e_local:
                answer = f"[ERROR] local LLM generation also failed: {str(e_local)}"
    elif chosen_llm == "local":
        try:
            answer = _generate_with_local_transformers(prompt)
        except Exception as e:
            answer = f"[ERROR] local LLM generation failed: {str(e)}"
    else:
        raise ValueError("llm must be 'auto', 'openai', or 'local'")

    return {
        "answer": answer,
        "sources": retrieved,
        "prompt": prompt
    }

# helper
def list_rag_files():
    return [p.name for p in MODELS_DIR.glob("rag_*")]

if __name__ == "__main__":
    q = input("Query> ")
    out = rag_qa_tool(q, top_k=5, llm="auto")
    print("ANSWER:\n", out["answer"][:1000])
    print("\nSOURCES:")
    for s in out["sources"]:
        print(s["id"], f"score={s['score']:.4f}", s['text_snippet'][:200])



# src/rag_demo.py (ultra-robust loader)
"""
Robust demo loader for RAG tool.

Loads src/rag_tool.py by file path, finds a suitable callable (prefers rag_qa_tool),
runs demo queries and writes reports/rag_demo_output.txt.

This script tries multiple fallbacks so it won't fail due to package import oddities.
"""
from pathlib import Path
import sys
import importlib.util
import inspect
import json
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_FILE = REPO_ROOT / "src" / "rag_tool.py"
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = REPORTS_DIR / "rag_demo_output.txt"

SAMPLE_QUERIES = [
    "best time to plant rice",
    "how to treat early blight on tomato",
    "recommended fertilizer for wheat at heading stage",
]

def load_module_from_path(path: Path, mod_name: str = "local_rag_tool"):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # show traceback but keep module (may be partly loaded)
        print("[WARN] Exception while executing module:", e)
        traceback.print_exc()
    return module

def find_rag_callable(module):
    # 1) exact name
    if hasattr(module, "rag_qa_tool") and callable(getattr(module, "rag_qa_tool")):
        return getattr(module, "rag_qa_tool")
    # 2) alternate common name
    for name in ("rag_tool", "ragQA", "rag_qa", "ragqa"):
        if hasattr(module, name) and callable(getattr(module, name)):
            return getattr(module, name)
    # 3) search for any callable with 'rag' and 'qa' in name
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        lname = name.lower()
        if "rag" in lname and "qa" in lname:
            return obj
    # 4) search for any callable that when called with (str) returns dict-like
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        try:
            # attempt a dry run with a tiny query, but capture exceptions
            if obj is None:
                continue
            sig = inspect.signature(obj)
            # only try callables that accept at least one positional argument
            if len(sig.parameters) >= 1:
                # call but with safe kwargs if named parameters
                try:
                    res = obj("test query for probe", top_k=1, llm="local")
                except TypeError:
                    # try calling with only query
                    try:
                        res = obj("test query for probe")
                    except Exception:
                        continue
                # check response shape lightly
                if isinstance(res, dict) and ("answer" in res or "sources" in res):
                    return obj
        except Exception:
            continue
    # none found
    return None

def run_demo(rag_callable):
    outputs = []
    for q in SAMPLE_QUERIES:
        print(f"[QUERY] {q}")
        try:
            # prefer signature with (query, top_k=..., llm=...)
            try:
                res = rag_callable(q, top_k=5, llm="auto")
            except TypeError:
                try:
                    res = rag_callable(q)
                except Exception as e:
                    res = {"answer": f"[ERROR running callable] {e}", "sources": []}
        except Exception as e:
            res = {"answer": f"[ERROR] {e}", "sources": []}
        # print short summary
        ans = res.get("answer") if isinstance(res, dict) else str(res)
        print("-> answer (first 300 chars):")
        print((ans or "<no answer>")[:300])
        print("-> sources (ids/scores):")
        for s in (res.get("sources") or []):
            sid = s.get("id", "<no-id>") if isinstance(s, dict) else str(s)
            score = s.get("score", "") if isinstance(s, dict) else ""
            print(f"   {sid}  score={score}")
        print("-" * 60)
        outputs.append({"query": q, "result": res})
    # save to file
    with open(OUT_FILE, "w", encoding="utf8") as fh:
        json.dump(outputs, fh, ensure_ascii=False, indent=2)
    print(f"[SAVED] demo outputs to {OUT_FILE}")

def main():
    print(f"[INFO] Loading module from {SRC_FILE}")
    module = load_module_from_path(SRC_FILE)
    print("[INFO] Module loaded. Available attributes:")
    print(sorted([a for a in dir(module) if not a.startswith("__")])[:200])
    rag_callable = find_rag_callable(module)
    if rag_callable is None:
        print("[ERROR] Could not find a rag callable in module.")
        print("Please check src/rag_tool.py contains a function named 'rag_qa_tool' or similar.")
        return
    print(f"[INFO] Using callable: {rag_callable.__name__}")
    run_demo(rag_callable)

if __name__ == "__main__":
    main()

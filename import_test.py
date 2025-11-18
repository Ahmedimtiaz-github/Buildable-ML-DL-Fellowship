# import_test.py - safe import check for src.agent_pipeline and src.rag_tool
import importlib, sys, traceback

print("Python executable:", sys.executable)
print("sys.path sample:", sys.path[:4])

ok = True
try:
    m = importlib.import_module("src.agent_pipeline")
    print("OK: imported src.agent_pipeline")
    print("Available names (sample):", [n for n in dir(m) if not n.startswith('_')][:50])
except Exception:
    ok = False
    print("ERROR importing src.agent_pipeline:")
    traceback.print_exc()

try:
    r = importlib.import_module("src.rag_tool")
    print("OK: imported src.rag_tool. rag_qa_tool exists?:", hasattr(r, "rag_qa_tool"))
except Exception:
    ok = False
    print("ERROR importing src.rag_tool:")
    traceback.print_exc()

if not ok:
    sys.exit(2)

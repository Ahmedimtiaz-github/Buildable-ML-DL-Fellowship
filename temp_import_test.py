import sys
print('Python executable:', sys.executable)
try:
    from src.agent_pipeline import query_agent
    print('OK: imported src.agent_pipeline, query_agent is present')
    names = [n for n in dir() if not n.startswith('_')]
    print('Sample names length:', len(names))
except Exception as e:
    import traceback
    traceback.print_exc()
    raise

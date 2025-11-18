# main.py - RAG agent wrapper (auto-generated)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Any, Dict
import tempfile, inspect, os
from pathlib import Path

# import query_agent from your pipeline
try:
    from src.agent_pipeline import query_agent
except Exception as e:
    raise RuntimeError(f"Failed to import src.agent_pipeline.query_agent: {e}")

app = FastAPI(title="Agent Pipeline API")

class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    llm: Optional[str] = "auto"

def _call_query_agent(question: str, top_k: int = 5, llm: str = "auto", image_path: Optional[str] = None, image_bytes: Optional[bytes] = None) -> Any:
    """
    Call query_agent adaptively based on its signature (works with different implementations).
    """
    sig = inspect.signature(query_agent)
    kwargs = {}
    if 'question' in sig.parameters:
        kwargs['question'] = question
    if 'top_k' in sig.parameters:
        kwargs['top_k'] = top_k
    if 'llm' in sig.parameters:
        kwargs['llm'] = llm
    # prefer image_path if tool supports it; otherwise try image_bytes
    if image_path and 'image_path' in sig.parameters:
        kwargs['image_path'] = image_path
    if image_bytes and 'image_bytes' in sig.parameters:
        kwargs['image_bytes'] = image_bytes

    # If we have built kwargs that match query_agent, call with them
    try:
        if kwargs:
            return query_agent(**kwargs)
        # fallback: try common positional call pattern
        return query_agent(question, top_k, llm)
    except TypeError:
        # last-resort: try calling with fewer args
        try:
            return query_agent(question)
        except Exception as e:
            raise

@app.get("/")
def root():
    return {"status": "ok", "note": "Use /agent-json for JSON requests, /agent for form-data (file uploads)."}

@app.post("/agent-json")
def agent_json(req: QARequest):
    """
    Accept JSON: { "question": "...", "top_k": 3, "llm": "local" }
    Returns whatever query_agent returns (should include tool_used/tool_output/reasoning_text).
    """
    try:
        result = _call_query_agent(req.question, req.top_k, req.llm)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent")
async def agent_form(
    question: str = Form(...),
    top_k: int = Form(5),
    llm: str = Form("auto"),
    file: Optional[UploadFile] = File(None)
):
    """
    Accept multipart/form-data. Use form field 'question' and optional file upload named 'file'.
    """
    tmp_path = None
    image_bytes = None
    try:
        if file is not None:
            contents = await file.read()
            image_bytes = contents
            # write a temporary file path for tools that expect a file path
            fd, tmp_path = tempfile.mkstemp(suffix='_' + (file.filename or "upload"))
            os.close(fd)
            with open(tmp_path, "wb") as fh:
                fh.write(contents)
        result = _call_query_agent(question, top_k, llm, image_path=tmp_path, image_bytes=image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

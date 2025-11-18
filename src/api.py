# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

# import rag_qa_tool from your module
from src.rag_tool import rag_qa_tool

app = FastAPI(title="RAG QA API")

class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    llm: Optional[str] = "auto"   # 'auto', 'openai', or 'local'

@app.post("/qa")
def qa(req: QARequest):
    try:
        res = rag_qa_tool(req.question, top_k=req.top_k or 5, llm=req.llm or "auto")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

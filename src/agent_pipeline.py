import os, json, traceback
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(r"C:\\Users\\ec\\Documents\\Buildable-ML-DL-Fellowship\\final project")
MODELS_DIR = REPO_ROOT / "models"

# --------------------------------------------
#  TOOL 1 — Disease Detector Tool (placeholder)
# --------------------------------------------
def disease_detector_tool(query: str, image_path: str = None) -> Dict[str, Any]:
    model_path = MODELS_DIR / "disease_efficientnet_final.h5"
    if not model_path.exists():
        return {
            "error": "disease model not found",
            "model_path": str(model_path)
        }
    # placeholder simple return (replace with real model inference)
    return {
        "prediction": "healthy",
        "confidence": 0.95
    }

# --------------------------------------------
#  TOOL 2 — Crop Predictor Tool (placeholder)
# --------------------------------------------
def crop_predictor_tool(query: str, features: Dict[str, float] = None) -> Dict[str, Any]:
    model_path = MODELS_DIR / "crop_predictor.pkl"
    if not model_path.exists():
        return {
            "error": "crop model not found",
            "model_path": str(model_path)
        }
    # placeholder simple return
    return {
        "recommended_crop": "wheat",
        "confidence": 0.88
    }

# --------------------------------------------
#  TOOL 3 — RAG Q/A Tool
#  (Uses existing src.rag_tool.rag_qa_tool)
# --------------------------------------------
from src.rag_tool import rag_qa_tool

# --------------------------------------------
# Routing logic: choose a tool by simple rules
# --------------------------------------------
def query_agent(question: str, file_path: str = None) -> Dict[str, Any]:
    q = question.lower() if question else ""

    # Rule 1: if an image provided or disease/leaf keywords -> disease_detector_tool
    if file_path or any(k in q for k in ("leaf","disease","spot","fungus","infection")):
        out = disease_detector_tool(question, file_path)
        return {
            "tool_used": "disease_detector_tool",
            "tool_input": {"query": question, "image_path": file_path},
            "tool_output": out,
            "reasoning_text": "Detected plant/leaf/disease keywords or file attached — routed to disease_detector_tool."
        }

    # Rule 2: soil/weather/planting keywords -> crop_predictor_tool
    if any(k in q for k in ("soil","weather","plant","fertilizer","grow","season","yield")):
        out = crop_predictor_tool(question, {})
        return {
            "tool_used": "crop_predictor_tool",
            "tool_input": {"query": question, "features": {}},
            "tool_output": out,
            "reasoning_text": "Detected soil/weather/planting keywords — routed to crop_predictor_tool."
        }

    # Default: RAG QA
    rag_out = rag_qa_tool(question, top_k=5, llm="local")
    return {
        "tool_used": "rag_qa_tool",
        "tool_input": {"query": question},
        "tool_output": rag_out,
        "reasoning_text": "No specialized keywords found — defaulting to RAG retrieval QA."
    }

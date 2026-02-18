# 🌾 Buildable ML/DL Fellowship — AI-Powered Farming Assistant

An end-to-end Machine Learning & Deep Learning project built during the **Buildable ML/DL Fellowship**. This project combines **crop prediction**, **plant disease detection**, and a **Retrieval-Augmented Generation (RAG)** question-answering system into a unified AI agent, all served via a **FastAPI** backend.

---

## 📌 Project Overview

This repository implements a multi-tool AI agent for agriculture that can:

1. **Predict optimal crops** based on soil and climate features (N, P, K, temperature, humidity, pH, rainfall) using classical ML models (Logistic Regression, Random Forest, XGBoost).
2. **Detect plant diseases** from leaf images using deep learning (CNN via TensorFlow/PyTorch).
3. **Answer farming questions** using a RAG pipeline that retrieves relevant context from a knowledge base (500 farming FAQs) and generates answers using OpenAI or a local transformer model.

The agent automatically routes queries to the appropriate tool based on the user's question.

---

## 🗂️ Repository Structure

```
Buildable-ML-DL-Fellowship/
├── data/                          # Raw and test datasets
│   ├── rag_knowledge/             # Farming FAQ knowledge base (CSV/TXT)
│   ├── processed/                 # Preprocessed data (crop splits, RAG chunks)
│   ├── agent_test.csv             # Agent routing test cases
│   ├── crop_test.csv              # Crop prediction test data
│   ├── disease_test.csv           # Disease detection test data
│   └── rag_test.csv               # RAG retrieval test data
├── eval/                          # Evaluation scripts and results
├── models/                        # Trained models & FAISS index artifacts
├── notebooks/                     # Jupyter notebooks (EDA, experiments)
├── report/                        # Project documentation
├── reports/                       # Generated output reports
├── scripts/                       # Utility scripts (e.g., NaN checks)
├── src/                           # Core source code
│   ├── api.py                     # FastAPI app for RAG QA endpoint
│   ├── agent_pipeline.py          # Multi-tool agent router
│   ├── crop_training.py           # Crop prediction model training
│   ├── crop_tool.py               # Crop prediction inference tool
│   ├── rag_preprocess.py          # RAG knowledge base preprocessing
│   ├── rag_tool.py                # RAG retrieval + QA tool
│   └── rag_demo.py                # RAG demo runner
├── build_rag_index.py             # Build FAISS index from chunked data
├── main.py                        # FastAPI agent wrapper (multi-tool API)
├── start_server.ps1               # PowerShell script to start the server
├── requirements.txt               # Python dependencies
├── Final_project_Buildables_DEEPLearning.ipynb  # Deep learning final notebook
└── FInal project report Buildables-M Ahmed Imtiaz(1).pdf  # Final report
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- A virtual environment is recommended

### 1. Clone the Repository

```bash
git clone https://github.com/Ahmedimtiaz-github/Buildable-ML-DL-Fellowship.git
cd Buildable-ML-DL-Fellowship
```

### 2. Create & Activate Virtual Environment

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔧 Usage

### Phase A & B — Crop Prediction (ML)

Train crop prediction models (Logistic Regression, Random Forest, XGBoost):

```bash
python src/crop_training.py
```

This reads processed data from `data/processed/crop/` and saves the best model artifact to `models/crop_predictor.pkl`.

### Phase C — Disease Detection (Deep Learning)

Refer to the Jupyter notebook for the deep learning pipeline:

```
Final_project_Buildables_DEEPLearning.ipynb
```

### Phase D — RAG Pipeline

#### Step 1: Preprocess Knowledge Base

```bash
python src/rag_preprocess.py
```

Reads files from `data/rag_knowledge/` and outputs chunked data to `data/processed/rag/chunks.jsonl`.

#### Step 2: Build Embeddings & FAISS Index

```bash
python build_rag_index.py
```

Generates:
- `models/rag_embeddings.npy` — sentence embeddings
- `models/rag_chunks_meta.jsonl` — chunk metadata
- `models/rag_faiss.index` — FAISS similarity search index

#### Step 3: Run RAG Demo

```bash
# Uses OpenAI if OPENAI_API_KEY is set, otherwise falls back to local transformers
python src/rag_demo.py
```

Output is saved to `reports/rag_demo_output.txt`.

### Phase E — Unified Agent API

Start the FastAPI server that routes queries to the appropriate tool:

```powershell
# PowerShell (one-liner)
.\start_server.ps1
```

Or manually:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

#### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check & usage info |
| `/agent-json` | POST | JSON request — auto-routes to the right tool |
| `/agent` | POST | Form-data request with optional file upload (e.g., leaf images) |
| `/qa` | POST | Direct RAG QA endpoint |
| `/health` | GET | Health status |

#### Example Request

```bash
curl -X POST http://127.0.0.1:8000/agent-json \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the best time to plant rice?", "top_k": 5}'
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **ML Frameworks** | scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow, Keras, PyTorch, torchvision |
| **NLP / Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`), transformers (Hugging Face) |
| **Vector Search** | FAISS (`faiss-cpu`) |
| **LLM Integration** | OpenAI API (optional), local transformers fallback |
| **Orchestration** | LangChain, LlamaIndex |
| **API Framework** | FastAPI + Uvicorn |
| **Data Processing** | NumPy, Pandas, OpenCV |
| **Visualization** | Matplotlib |

---

## 📊 Dataset

The RAG knowledge base uses the **Farming FAQ Assistant Dataset**:

- **500** question-answer pairs covering agriculture topics
- Topics include planting schedules, pest management, fertilizer recommendations, and more
- Designed for powering conversational AI assistants for farming

---

## 📝 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (optional — enables GPT-based answers) | *not set* (uses local model) |
| `RAG_EMBED_MODEL` | Sentence-transformers embedding model name | `all-MiniLM-L6-v2` |
| `RAG_LOCAL_MODEL` | Local transformer model for text generation | `distilgpt2` |

---

## 👤 Author

**M. Ahmed Imtiaz**
- GitHub: [@Ahmedimtiaz-github](https://github.com/Ahmedimtiaz-github)

---

## 📄 License

This project is for educational and research purposes as part of the Buildable ML/DL Fellowship program.
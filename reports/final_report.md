# Final Project Report — Agentic RAG-Based Smart Farming Advisor

## 1. Problem statement
(Write problem statement here)

## 2. Datasets & preprocessing
- List datasets: models/, data/
- Note preprocessing steps (resizing images, normalization, train/test splits)

## 3. Models & hyperparameters
- Crop predictor: (type, features, hyperparams)
- Disease model: (EfficientNet-type, training epochs, batch size)
- RAG: embedding model, chunk size, FAISS index
- Agent: rule-based routing logic

## 4. Results
### Crop model
- Metrics: link to `reports/eval/crop/crop_metrics.json`
![confusion](./reports/eval/crop/crop_confusion_matrix.png)

### Disease model
- Metrics: `reports/eval/disease/disease_metrics.json`
![confusion](./reports/eval/disease/disease_confusion_matrix.png)

### RAG retrieval
- Metrics: `reports/eval/rag/rag_metrics.json`
- Examples: `reports/eval/rag/rag_examples.csv`

### Agent
- Metrics: `reports/eval/agent/agent_metrics.json`
- Confusion: `reports/eval/agent/agent_confusion_matrix.csv`

## 5. Agent flow diagram
(Insert a small diagram or screenshot — you can draw in any tool and add here)

## 6. Limitations & future work
- List points: dataset size, domain shift, hallucination risk in RAG, runtime perf.

## 7. Reproducibility & README pointers
- Commands to run evaluations: `python ./eval/eval_*.py --test ...`
- To run the agent server: `python -m uvicorn main:app --reload`


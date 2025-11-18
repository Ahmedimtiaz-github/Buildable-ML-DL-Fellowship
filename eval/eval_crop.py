# --- INSERTED: robust_model_loader for crop eval ---
import joblib, pickle, os
def robust_load_crop_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"crop model not found: {path}")
    # try joblib then pickle
    try:
        m = joblib.load(path)
    except Exception:
        with open(path, "rb") as fh:
            m = pickle.load(fh)
    # if dict, attempt to pull actual model and helpers
    if isinstance(m, dict):
        # common keys
        for k in ("model","estimator","clf","sklearn_model"):
            if k in m:
                real = m[k]
                # keep metadata too
                m = {"model": real, **m}
                break
        # if still dict but has model as key, nothing to do
        if "model" in m and not hasattr(m["model"], "predict"):
            # last attempt: if the object itself is a scikit wrapper under a known key
            for k in m.keys():
                if hasattr(m[k], "predict"):
                    m = {"model": m[k], **m}
                    break
    # normalize: return dict with 'model' and optional scaler/feature_columns/encoders
    if hasattr(m, "predict"):
        return {"model": m}
    if isinstance(m, dict) and "model" in m and hasattr(m["model"], "predict"):
        return m
    raise RuntimeError(f"Loaded crop model object from {path} does not expose .predict(); type={type(m)}")
# --- END INSERT ---
# eval/eval_crop.py
import os, json, argparse, joblib
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

def main(args):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model_path = args.model or "models/crop_predictor.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Exiting.")
        return
# --- safe model load inserted by helper ---
import joblib, pickle, os
try:
    model = joblib.load(model_path)
except Exception:
    with open(model_path,'rb') as fh:
        model = pickle.load(fh)

# if the saved object is a dict with the real estimator under 'model' or 'estimator', extract it.
if isinstance(model, dict):
    for k in ('model','estimator','clf','sklearn_model'):
        if k in model:
            model = model[k]
            break

if not hasattr(model, 'predict'):
    raise RuntimeError(f\"Loaded object from {model_path} has no predict() - type={type(model)}\")
# --- end safe load ---

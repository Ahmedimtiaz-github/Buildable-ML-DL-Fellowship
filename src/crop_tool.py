import os, joblib, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "crop_predictor.pkl"

def _reconstruct_label_encoder(classes):
    le = LabelEncoder()
    le.classes_ = np.array(classes, dtype=object)
    return le

def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run training first.")
    artifact = joblib.load(MODEL_PATH)
    return artifact

artifact_cache = None
def predict_crop(input_dict):
    """
    input_dict: dictionary with numeric keys e.g.
    {
      "N": 90,
      "P": 42,
      "K": 43,
      "temperature": 20.5,
      "humidity": 80.0,
      "ph": 6.5,
      "rainfall": 200
      # plus optional categorical fields if present in encoders
    }
    returns: dict with predicted crop, top3, probabilities, and reasoning
    """
    global artifact_cache
    if artifact_cache is None:
        artifact_cache = load_artifact()
    art = artifact_cache
    model = art["model"]
    feature_columns = art["feature_columns"]
    scaler = art.get("scaler", None)
    encoders = art.get("encoders", None)
    classes = art.get("classes", [])

    # Build a single-row DataFrame with features
    df = pd.DataFrame([{}])
    # copy raw numeric keys if provided
    for k in ["N","P","K","temperature","humidity","ph","rainfall"]:
        if k in input_dict:
            df[k] = input_dict[k]
    # compute derived features if not present
    if {"N","P","K"}.issubset(set(df.columns)) or all(k in input_dict for k in ["N","P","K"]):
        try:
            n = float(input_dict.get("N", df.get("N", [np.nan])[0]))
            p = float(input_dict.get("P", df.get("P", [np.nan])[0]))
            k_ = float(input_dict.get("K", df.get("K", [np.nan])[0]))
            df["soil_fertility"] = (n + p + k_) / 3.0
        except:
            df["soil_fertility"] = np.nan
    if "temperature" in input_dict and "humidity" in input_dict:
        try:
            df["climate_index"] = float(input_dict.get("temperature")) * float(input_dict.get("humidity"))
        except:
            df["climate_index"] = np.nan

    # Ensure all feature columns exist; fill missing numeric with 0
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0

    # Apply encoders for categorical features if encoders available
    if encoders:
        for col, meta in encoders.items():
            if meta.get("type") == "label":
                # reconstruct label encoder
                classes_meta = meta.get("classes", [])
                le = _reconstruct_label_encoder(classes_meta)
                val = input_dict.get(col, "")
                try:
                    df[col] = le.transform([str(val)])[0] if str(val) in list(le.classes_) else 0
                except Exception:
                    df[col] = 0
            elif meta.get("type") == "onehot":
                # meta.cols lists the one-hot column names created during preprocessing
                # Set those columns to 0, then set the matching one to 1 if provided
                onehot_cols = meta.get("cols", [])
                # ensure they exist
                for oc in onehot_cols:
                    if oc not in df.columns:
                        df[oc] = 0
                provided = input_dict.get(col, None)
                if provided is not None:
                    candidate_col = f"{col}_{provided}"
                    if candidate_col in df.columns:
                        df[candidate_col] = 1

    # Now scale numeric features (if scaler exists)
    if scaler is not None:
        # scaler expects numeric columns that were used during preprocessing
        # we apply scaler to intersection of scaler.feature_names_in_ and df.columns if available
        try:
            if hasattr(scaler, "feature_names_in_"):
                cols_to_scale = [c for c in scaler.feature_names_in_ if c in df.columns]
            else:
                # fallback: scale numeric columns
                cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        except Exception as e:
            # if scaling fails, continue with raw values (but warn)
            print("Warning: scaler transform failed:", e)

    # Align columns to model feature order
    X = df[feature_columns].fillna(0.0)
    # Predict
    try:
        probs = model.predict_proba(X)[0]
        idxs = np.argsort(probs)[::-1][:3]
        top3 = [(model.classes_[i], float(probs[i])) for i in idxs]
        pred = model.classes_[np.argmax(probs)]
    except Exception:
        # some models (e.g., certain sklearn wrappers) might not have predict_proba for label encoding issues
        pred = model.predict(X)[0]
        top3 = [(pred, 1.0)]

    # create reasoning
    reasoning_parts = []
    if "soil_fertility" in X.columns:
        reasoning_parts.append(f\"soil_fertility={float(X['soil_fertility'].iloc[0]):.3f}\")
    if "climate_index" in X.columns:
        reasoning_parts.append(f\"climate_index={float(X['climate_index'].iloc[0]):.3f}\")
    reasoning = \"; \".join(reasoning_parts) if reasoning_parts else \"Used model features to predict.\"

    return {
        "predicted_crop": str(pred),
        "top3": [{"crop": t[0], "probability": t[1]} for t in top3],
        "reasoning": f\"{reasoning}; model={art.get('model_name')}\"
    }

# allow direct CLI test
if __name__ == '__main__':
    # example test dictionary (user can modify)
    example = {"N": 90, "P": 42, "K": 43, "temperature": 20.5, "humidity": 80, "ph": 6.5, "rainfall": 200}
    print("Testing with example input:", example)
    print(predict_crop(example))

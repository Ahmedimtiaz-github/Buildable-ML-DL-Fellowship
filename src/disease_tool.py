#!/usr/bin/env python3
"""
disease_tool.py
Usage:
  python disease_tool.py /path/to/image.jpg
  python disease_tool.py /path/to/image.jpg --model "../models/disease_efficientnet_final.h5"
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

def default_paths():
    repo_root = Path(__file__).resolve().parents[1]
    model_default = repo_root / "models" / "disease_efficientnet_final.h5"
    classes_default = repo_root / "models" / "classes_unified.json"
    return model_default, classes_default

_model = None
_classes = None

def _to_path(p):
    # accept None, Path, or str -> return Path or None
    if p is None:
        return None
    if isinstance(p, Path):
        return p
    return Path(str(p))

def load_model_and_classes(model_path=None, classes_path=None):
    global _model, _classes
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    # convert to Path
    model_path = _to_path(model_path)
    classes_path = _to_path(classes_path)

    if model_path is None or classes_path is None:
        mdef, cdef = default_paths()
        model_path = model_path or mdef
        classes_path = classes_path or cdef

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if _model is None:
        _model = load_model(str(model_path), compile=False)

    if _classes is None:
        if classes_path.exists():
            with open(classes_path, "r", encoding="utf8") as f:
                _classes = json.load(f)
        else:
            _classes = [str(i) for i in range(_model.output_shape[-1])]

    return _model, _classes

def preprocess_image(path_or_pil, target_size=(224,224)):
    if isinstance(path_or_pil, (str, Path)):
        img = Image.open(str(path_or_pil)).convert("RGB")
    elif isinstance(path_or_pil, Image.Image):
        img = path_or_pil.convert("RGB")
    else:
        raise ValueError("Unsupported image input")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_disease(image_path, model_path=None, classes_path=None, top_k=3):
    model, classes = load_model_and_classes(model_path, classes_path)
    x = preprocess_image(image_path, target_size=(model.input_shape[1], model.input_shape[2]))
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    t1 = time.perf_counter()
    idxs = np.argsort(probs)[::-1][:top_k]
    topk = [{"label": classes[i] if i < len(classes) else str(i), "probability": float(probs[i])} for i in idxs]
    return {
        "predicted_label": topk[0]["label"],
        "top_k": topk,
        "inference_time_ms": (t1 - t0) * 1000.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", help="Path to Keras model (.h5). Default: repo/models/disease_efficientnet_final.h5", default=None)
    parser.add_argument("--classes", help="Path to classes json. Default: repo/models/classes_unified.json", default=None)
    parser.add_argument("--topk", type=int, default=5, help="Return top-k predictions")
    args = parser.parse_args()

    img_path = args.image
    model_path = args.model
    classes_path = args.classes

    try:
        out = predict_disease(img_path, model_path=model_path, classes_path=classes_path, top_k=args.topk)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("ERROR:", str(e))

if __name__ == "__main__":
    main()

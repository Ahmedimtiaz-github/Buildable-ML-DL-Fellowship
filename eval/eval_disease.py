# --- safe keras load (compile=False fallback) ---
from tensorflow.keras.models import load_model as _tf_load_model
def safe_load_keras_model(path):
    try:
        return _tf_load_model(path, compile=False)
    except Exception as _e:
        try:
            return _tf_load_model(path)
        except Exception as e2:
            raise RuntimeError(f"Failed to load keras model {path}: {_e} | {e2}")
# use safe_load_keras_model(model_path) where load_model(model_path) used earlier
# eval/eval_disease.py
import os, json, argparse, time
import numpy as np, pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PIL import Image
import tqdm

def load_image(path, target_size):
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = np.array(img) / 255.0
    return arr

def main(args):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    model_path = args.model or "models/disease_efficientnet_final.h5"
    if not os.path.exists(model_path):
        print(f"Disease model not found: {model_path}")
        return

    try:\n    model = safe_load_keras_model(, compile=False)\nexcept Exception as _e:\n    # fallback: try without compile flag (older TF/Keras versions)\n    model = safe_load_keras_model()\n

    # Test list CSV: columns 'image_path' and 'label' (label numeric or str)
    test_csv = args.test or "data/disease_test.csv"
    if not os.path.exists(test_csv):
        print(f"Test CSV not found: {test_csv}")
        return

    df = pd.read_csv(test_csv)
    if not {'image_path','label'}.issubset(df.columns):
        print("Test CSV must contain 'image_path' and 'label'.")
        return

    X_paths = df['image_path'].tolist()
    y_true = df['label'].tolist()

    # load target size
    # try to infer input shape
    try:
        input_shape = model.input_shape
        # shape like (None, h, w, c)
        target_size = (input_shape[1], input_shape[2])
    except Exception:
        target_size = (224,224)
    print("Using target_size:", target_size)

    # run predictions and time inference
    y_pred = []
    times = []
    for p in tqdm.tqdm(X_paths, desc="Inferring"):
        if not os.path.exists(p):
            print("Missing image:", p)
            y_pred.append(None)
            continue
        img = load_image(p, target_size)
        img_batch = np.expand_dims(img, axis=0).astype(np.float32)
        t0 = time.time()
        logits = model.predict(img_batch)
        t1 = time.time()
        times.append((t1-t0)*1000.0)  # ms
        pred = np.argmax(logits, axis=-1)[0]
        y_pred.append(pred)

    # filter out None predictions
    mask = [yp is not None for yp in y_pred]
    y_true_f = [y_true[i] for i, m in enumerate(mask) if m]
    y_pred_f = [y_pred[i] for i, m in enumerate(mask) if m]

    acc = accuracy_score(y_true_f, y_pred_f)
    clf_report = classification_report(y_true_f, y_pred_f, output_dict=True)
    cm = confusion_matrix(y_true_f, y_pred_f)
    metrics = {"accuracy": float(acc), "inference_time_ms_avg": float(np.mean(times)) if times else None}

    with open(os.path.join(out_dir, "disease_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("Saved disease metrics.")

    # confusion matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap='Reds')
    plt.title("Disease confusion matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "disease_confusion_matrix.png"))
    plt.close()
    print("Saved confusion matrix.")

    # training curves: if training_history.json exists in models folder, try to plot
    hist_path = "models/training_summary.json"
    if os.path.exists(hist_path):
        try:
            with open(hist_path) as fh:
                hist = json.load(fh)
            # expect keys 'loss','val_loss','accuracy','val_accuracy'
            plt.figure()
            if 'loss' in hist and 'val_loss' in hist:
                plt.plot(hist['loss'], label='loss'); plt.plot(hist['val_loss'], label='val_loss')
                plt.legend(); plt.title('Loss curves'); plt.savefig(os.path.join(out_dir,'loss_curve.png')); plt.close()
            if 'accuracy' in hist and 'val_accuracy' in hist:
                plt.figure()
                plt.plot(hist['accuracy'], label='acc'); plt.plot(hist['val_accuracy'], label='val_acc')
                plt.legend(); plt.title('Accuracy curves'); plt.savefig(os.path.join(out_dir,'acc_curve.png')); plt.close()
            print("Plotted training curves from training_summary.json")
        except Exception as e:
            print("Failed to plot training curves:", e)
    else:
        print("No training_summary.json found in models/ - skipping training curves")
    # save timings to CSV
    pd.DataFrame({"image": [p for p in X_paths if os.path.exists(p)], "time_ms": times}).to_csv(os.path.join(out_dir,"disease_inference_times.csv"), index=False)
    print("Saved inference times CSV.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--out", default="reports/eval/disease")
    args = parser.parse_args()
    main(args)



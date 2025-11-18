import os, joblib, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "crop"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load processed csvs
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

# load feature columns (expects file created in preprocessing)
feat_file = DATA_DIR / "feature_columns.csv"
if feat_file.exists():
    fc = pd.read_csv(feat_file)
    if "feature_columns" in fc.columns:
        feature_columns = fc["feature_columns"].dropna().tolist()
    else:
        feature_columns = [c for c in train_df.columns if c != "label"]
else:
    feature_columns = [c for c in train_df.columns if c != "label"]

# X/y
X_train = train_df[feature_columns].copy()
y_train = train_df["label"].astype(str).copy()
X_val   = val_df[feature_columns].copy()
y_val   = val_df["label"].astype(str).copy()
X_test  = test_df[feature_columns].copy()
y_test  = test_df["label"].astype(str).copy()

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Unique labels in training set:", sorted(y_train.unique()))

# === Imputation step (fix NaNs) ===
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [c for c in feature_columns if c not in numeric_cols]

# Compute medians / modes on training set
medians = X_train[numeric_cols].median()
modes = {}
for c in non_numeric_cols:
    try:
        modes[c] = X_train[c].mode().iloc[0]
    except Exception:
        modes[c] = ""

# Fill NaNs in train/val/test using training statistics
X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
X_val[numeric_cols]   = X_val[numeric_cols].fillna(medians)
X_test[numeric_cols]  = X_test[numeric_cols].fillna(medians)

for c in non_numeric_cols:
    X_train[c] = X_train[c].fillna(modes.get(c, ""))
    X_val[c]   = X_val[c].fillna(modes.get(c, ""))
    X_test[c]  = X_test[c].fillna(modes.get(c, ""))

# Quick check
print("NaNs after imputation (train,val,test):", X_train.isna().sum().sum(), X_val.isna().sum().sum(), X_test.isna().sum().sum())

# === Label encoding for classifiers that need numeric labels (XGBoost etc.) ===
le = LabelEncoder()
le.fit(y_train)
y_train_enc = le.transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

# define models to train
models = {
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "xgb": XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
}

reports_dir = OUT_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

results = {}

for name, model in models.items():
    print(f"Training {name} ...")
    # Train using encoded labels
    model.fit(X_train, y_train_enc)
    # evaluate on val and test (use encoded labels)
    yv_pred_enc = model.predict(X_val)
    yt_pred_enc = model.predict(X_test)
    # compute metrics
    f1_val = f1_score(y_val_enc, yv_pred_enc, average="macro")
    acc_val = accuracy_score(y_val_enc, yv_pred_enc)
    f1_test = f1_score(y_test_enc, yt_pred_enc, average="macro")
    acc_test = accuracy_score(y_test_enc, yt_pred_enc)
    results[name] = {
        "model": model,
        "f1_val": float(f1_val),
        "acc_val": float(acc_val),
        "f1_test": float(f1_test),
        "acc_test": float(acc_test)
    }
    # save classification report text files using readable class names
    with open(reports_dir / f"{name}_val_report.txt", "w", encoding="utf8") as f:
        f.write(f"Validation report for {name}\n")
        f.write(classification_report(y_val_enc, yv_pred_enc, target_names=list(le.classes_), digits=4))
    with open(reports_dir / f"{name}_test_report.txt", "w", encoding="utf8") as f:
        f.write(f"Test report for {name}\n")
        f.write(classification_report(y_test_enc, yt_pred_enc, target_names=list(le.classes_), digits=4))
    # confusion matrix on test (use numeric labels for plotting)
    cm = confusion_matrix(y_test_enc, yt_pred_enc, labels=list(range(len(le.classes_))))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=False, fmt="d")
    except Exception:
        plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion matrix (test) - {name}")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{name}_confusion_test.png")
    plt.close()
    print(f"{name}: val f1={f1_val:.4f} test f1={f1_test:.4f}")

# pick best model by validation f1
best_name = max(results.keys(), key=lambda k: results[k]["f1_val"])
best_model = results[best_name]["model"]
print("Best model on validation (f1_macro):", best_name, results[best_name]["f1_val"])

# load scaler and encoders artifacts so wrapper can use them (if present)
scaler_path = DATA_DIR / "scaler.joblib"
encoders_path = DATA_DIR / "encoders.joblib"
scaler = joblib.load(scaler_path) if scaler_path.exists() else None
encoders = joblib.load(encoders_path) if encoders_path.exists() else None

# Save a single artifact blob containing model + metadata
artifact = {
    "model": best_model,
    "model_name": best_name,
    "feature_columns": feature_columns,
    "scaler": scaler,
    "encoders": encoders,
    "classes": list(le.classes_),
    "label_classes_encoded": list(range(len(le.classes_)))
}
artifact_path = OUT_DIR / "crop_predictor.pkl"
joblib.dump(artifact, artifact_path)
print("Saved final model artifact to", artifact_path)

# Save a JSON summary of model metrics
summary = {k: {"f1_val": results[k]["f1_val"], "acc_val": results[k]["acc_val"], "f1_test": results[k]["f1_test"], "acc_test": results[k]["acc_test"]} for k in results}
with open(OUT_DIR / "training_summary.json", "w", encoding="utf8") as f:
    json.dump({"best_model": best_name, "results": summary}, f, indent=2)

print("Training complete. Reports in", reports_dir)

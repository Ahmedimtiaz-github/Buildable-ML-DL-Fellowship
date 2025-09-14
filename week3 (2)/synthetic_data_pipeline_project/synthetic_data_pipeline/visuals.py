import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def _ensure_dir(path):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

def save_hist(csv_path, column, out_path):
    df = pd.read_csv(csv_path)
    _ensure_dir(out_path)
    plt.figure()
    if column not in df.columns:
        raise KeyError(f"Column {column} not found in {csv_path}")
    df[column].hist(bins=20)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def save_corr_heatmap(csv_path, out_path):
    df = pd.read_csv(csv_path)
    _ensure_dir(out_path)
    plt.figure(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=["number"]).corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def save_bar(csv_path, column, out_path):
    df = pd.read_csv(csv_path)
    _ensure_dir(out_path)

    if column in df.columns:
        counts = df[column].value_counts()
    else:
        # attempt to find dummy/one-hot columns like column_*
        dummy_cols = [c for c in df.columns if c.startswith(column + "_")]
        if dummy_cols:
            # reconstruct category per-row by argmax across dummy columns
            sub = df[dummy_cols]
            # If sub is empty, fallback
            if sub.shape[1] == 0:
                raise KeyError(f"No dummy columns for '{column}' found.")
            # idxmax returns the column name (dummy) with highest value; convert to label
            labels = sub.idxmax(axis=1).apply(lambda s: s.replace(column + "_", ""))
            counts = labels.value_counts()
        else:
            raise KeyError(f"Column '{column}' not found and no dummy columns detected.")
    plt.figure()
    counts.plot(kind="bar")
    plt.title(f"Bar plot of {column}")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def save_scatter(csv_path, col_x, col_y, out_path):
    df = pd.read_csv(csv_path)
    _ensure_dir(out_path)
    if col_x not in df.columns or col_y not in df.columns:
        raise KeyError(f"Columns {col_x} or {col_y} not found in {csv_path}")
    plt.figure()
    plt.scatter(df[col_x], df[col_y], alpha=0.6)
    plt.xlabel(col_x); plt.ylabel(col_y)
    plt.title(f"{col_y} vs {col_x}")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

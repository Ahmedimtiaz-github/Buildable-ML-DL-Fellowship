import numpy as np
import pandas as pd
import os

def mean(series):
    return float(np.nanmean(np.array(series, dtype=float)))

def median(series):
    return float(np.nanmedian(np.array(series, dtype=float)))

def std(series):
    return float(np.nanstd(np.array(series, dtype=float), ddof=1))

def save_stats(df, cols, out_path="results/stats.csv"):
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        rows.append({
            "column": c,
            "mean": mean(df[c]),
            "median": median(df[c]),
            "std": std(df[c])
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

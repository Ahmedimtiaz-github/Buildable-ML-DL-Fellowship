# create_files.ps1
# Run this from:
# C:\Users\ec\Documents\Buildable-ML-DL-Fellowship\week4\synthetic_data_pipeline_project

# Ensure directories exist
New-Item -ItemType Directory -Force -Path .\synthetic_data_pipeline | Out-Null
New-Item -ItemType Directory -Force -Path .\src | Out-Null
New-Item -ItemType Directory -Force -Path .\data\raw | Out-Null
New-Item -ItemType Directory -Force -Path .\data\processed | Out-Null
New-Item -ItemType Directory -Force -Path .\plots | Out-Null
New-Item -ItemType Directory -Force -Path .\models | Out-Null
New-Item -ItemType Directory -Force -Path .\logs | Out-Null
New-Item -ItemType Directory -Force -Path .\results | Out-Null

function Write-UTF8File($path, $content) {
    $content | Out-File -FilePath $path -Encoding utf8 -Force
}

# exceptions.py
$exceptions = @'
# exceptions.py
class PipelineError(Exception):
    """Base class for pipeline errors."""
    pass

class InvalidPathError(PipelineError):
    """Raised when a file path is invalid or inaccessible."""
    pass

class DataGenerationError(PipelineError):
    """Raised when data generation fails due to wrong parameters."""
    pass
'@
Write-UTF8File -path ".\synthetic_data_pipeline\exceptions.py" -content $exceptions

# generator.py
$generator = @'
# generator.py
import os
import numpy as np
import pandas as pd
from .exceptions import InvalidPathError, DataGenerationError
import logging
from typing import Optional

# configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "errors.txt"),
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s: %(message)s"
)

class DataGenerator:
    def __init__(self, n_rows: int = 500, random_state: Optional[int] = 42):
        if n_rows < 1:
            raise DataGenerationError("n_rows must be >= 1")
        self.n_rows = n_rows
        self.rng = np.random.default_rng(random_state)

    def _generate_numerical(self):
        # example numerical features
        age = self.rng.integers(18, 80, size=self.n_rows)
        income = self.rng.normal(50000, 15000, size=self.n_rows).round(2)
        score1 = self.rng.normal(70, 10, size=self.n_rows).clip(0,100)
        score2 = self.rng.normal(50, 15, size=self.n_rows).clip(0,100)
        visits = self.rng.poisson(3, size=self.n_rows)
        return {
            "age": age,
            "income": income,
            "score1": score1,
            "score2": score2,
            "visits": visits
        }

    def _generate_categorical(self):
        genders = self.rng.choice(["Male", "Female", "Other"], size=self.n_rows, p=[0.48,0.48,0.04])
        product = self.rng.choice(["A","B","C"], size=self.n_rows, p=[0.5,0.3,0.2])
        return {
            "gender": genders,
            "product": product
        }

    def _generate_target(self, num_features_dict):
        income = num_features_dict["income"]
        score1 = num_features_dict["score1"]
        visits = num_features_dict["visits"]
        probs = 1 / (1 + np.exp(-0.00004*(income-40000) + 0.03*(score1-50) + 0.2*(visits-2)))
        target = self.rng.random(self.n_rows) < probs
        return target.astype(int)

    def generate(self, save_path: str):
        save_dir = os.path.dirname(save_path) or "."
        if not os.path.exists(save_dir):
            msg = f"Directory does not exist: {save_dir}"
            logging.error(msg)
            raise InvalidPathError(msg)

        try:
            nums = self._generate_numerical()
            cats = self._generate_categorical()
            target = self._generate_target(nums)
            df = pd.DataFrame({**nums, **cats})
            df["target"] = target

            if self.n_rows >= 5:
                idx = self.rng.choice(self.n_rows, size=5, replace=False)
                df.loc[idx, "income"] = np.nan

            df.to_csv(save_path, index=False)
            return df
        except Exception as e:
            logging.exception("Failed to generate dataset")
            raise DataGenerationError(f"Data generation failed: {e}")
'@
Write-UTF8File -path ".\synthetic_data_pipeline\generator.py" -content $generator

# stats.py
$stats = @'
# stats.py
import numpy as np

def mean(arr):
    arr = np.array(arr, dtype=float)
    return np.nanmean(arr)

def median(arr):
    arr = np.array(arr, dtype=float)
    return np.nanmedian(arr)

def std(arr):
    arr = np.array(arr, dtype=float)
    return np.nanstd(arr, ddof=1)
'@
Write-UTF8File -path ".\synthetic_data_pipeline\stats.py" -content $stats

# augment.py
$augment = @'
# augment.py
import pandas as pd
import numpy as np

def augment_with_jitter(df: pd.DataFrame, numeric_cols: list, factor: float = 2.0, jitter_scale: float = 0.01, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    orig_n = df.shape[0]
    target_n = int(orig_n * factor)
    if target_n <= orig_n:
        return df.copy()

    samples = []
    indices = rng.integers(0, orig_n, size=(target_n - orig_n))
    for idx in indices:
        row = df.iloc[idx].copy()
        for col in numeric_cols:
            if pd.isna(row[col]):
                continue
            std = df[col].std() if df[col].std() > 0 else 1.0
            noise = rng.normal(0, jitter_scale * std)
            row[col] = row[col] + noise
        samples.append(row)
    augmented = pd.concat([df, pd.DataFrame(samples)], ignore_index=True)
    return augmented
'@
Write-UTF8File -path ".\synthetic_data_pipeline\augment.py" -content $augment

# visuals.py
$visuals = @'
# visuals.py
import matplotlib.pyplot as plt
import pandas as pd
import os

def save_hist(df: pd.DataFrame, column: str, out_path: str):
    plt.figure()
    df[column].hist()
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_bar(df: pd.DataFrame, column: str, out_path: str):
    plt.figure()
    df[column].value_counts().plot(kind="bar")
    plt.title(f"Bar plot of {column}")
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_scatter(df: pd.DataFrame, x: str, y: str, out_path: str):
    plt.figure()
    plt.scatter(df[x], df[y], alpha=0.6)
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_corr_heatmap(df, numeric_columns, out_path):
    plt.figure(figsize=(6,5))
    corr = df[numeric_columns].corr()
    import numpy as np
    plt.imshow(corr, interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45, ha="right")
    plt.yticks(range(len(numeric_columns)), numeric_columns)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
'@
Write-UTF8File -path ".\synthetic_data_pipeline\visuals.py" -content $visuals

# trainer.py
$trainer = @'
# trainer.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "errors.txt"),
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s: %(message)s"
)

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, target_col: str = "target"):
        self.df = df.copy()
        self.target_col = target_col
        if target_col not in df.columns:
            raise ValueError(f"target column \'{target_col}\' not found in dataframe")

    def prepare(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        X = pd.get_dummies(X, drop_first=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def train_and_evaluate(self, out_model_dir: str, metrics_path: str):
        os.makedirs(out_model_dir, exist_ok=True)
        models = {
            "logistic": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        results = []
        for name, model in models.items():
            try:
                model.fit(self.X_train, self.y_train)
                preds = model.predict(self.X_test)
                probs = model.predict_proba(self.X_test)[:,1] if hasattr(model, "predict_proba") else None
                res = {
                    "model": name,
                    "accuracy": accuracy_score(self.y_test, preds),
                    "precision": precision_score(self.y_test, preds, zero_division=0),
                    "recall": recall_score(self.y_test, preds, zero_division=0),
                    "f1": f1_score(self.y_test, preds, zero_division=0),
                    "roc_auc": roc_auc_score(self.y_test, probs) if probs is not None else None
                }
                results.append(res)
                joblib.dump(model, os.path.join(out_model_dir, f"{name}.joblib"))
            except Exception as e:
                logging.exception(f"Failed training {name}")
        import csv
        keys = results[0].keys() if results else []
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            writer.writerows(results)
        return results
'@
Write-UTF8File -path ".\synthetic_data_pipeline\trainer.py" -content $trainer

# __init__.py
$init = @'
# __init__.py
__all__ = ["generator", "stats", "augment", "visuals", "trainer", "exceptions"]
'@
Write-UTF8File -path ".\synthetic_data_pipeline\__init__.py" -content $init

# src/generate.py
$generate = @'
# generate.py
from synthetic_data_pipeline.generator import DataGenerator
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
SAVE_PATH = os.path.join(RAW_DIR, "generated_data.csv")

if __name__ == "__main__":
    gen = DataGenerator(n_rows=1000, random_state=42)
    df = gen.generate(save_path=SAVE_PATH)
    print("Generated:", SAVE_PATH)
    print(df.head())
'@
Write-UTF8File -path ".\src\generate.py" -content $generate

# src/prepare.py
$prepare = @'
# prepare.py
import pandas as pd
import os

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "generated_data.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_data.csv")

def main():
    df = pd.read_csv(RAW)
    print("Before cleaning, missing values per column:")
    print(df.isna().sum())

    num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)
    for c in cat_cols:
        df[c].fillna(df[c].mode().iloc[0], inplace=True)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Saved cleaned dataset to:", OUT)
    print("After cleaning, missing values per column:")
    print(df.isna().sum())

if __name__ == "__main__":
    main()
'@
Write-UTF8File -path ".\src\prepare.py" -content $prepare

# src/augment_run.py
$augment_run = @'
# augment_run.py
import os
import pandas as pd
from synthetic_data_pipeline.augment import augment_with_jitter

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_data.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "augmented_data.csv")

def main():
    df = pd.read_csv(IN_PATH)
    numeric_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")
    augmented = augment_with_jitter(df, numeric_cols=numeric_cols, factor=2.0, jitter_scale=0.02)
    augmented.to_csv(OUT_PATH, index=False)
    print("Saved augmented dataset:", OUT_PATH)
    print("Original rows:", df.shape[0], "Augmented rows:", augmented.shape[0])

if __name__ == "__main__":
    main()
'@
Write-UTF8File -path ".\src\augment_run.py" -content $augment_run

# src/visualize_run.py
$visualize_run = @'
# visualize_run.py
import os
import pandas as pd
from synthetic_data_pipeline.visuals import save_hist, save_bar, save_scatter, save_corr_heatmap

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_data.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    numeric_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    save_hist(df, numeric_cols[0], os.path.join(PLOTS_DIR, f"{numeric_cols[0]}_hist.png"))
    cat_col = None
    for c in df.columns:
        if df[c].nunique() <= 10 and c not in numeric_cols:
            cat_col = c
            break
    if cat_col:
        save_bar(df, cat_col, os.path.join(PLOTS_DIR, f"{cat_col}_bar.png"))
    save_corr_heatmap(df, numeric_cols, os.path.join(PLOTS_DIR, "corr_heatmap.png"))
    if len(numeric_cols) >= 2:
        save_scatter(df, numeric_cols[0], numeric_cols[1], os.path.join(PLOTS_DIR, f"{numeric_cols[0]}_vs_{numeric_cols[1]}.png"))
    print("Saved plots to", PLOTS_DIR)

if __name__ == "__main__":
    main()
'@
Write-UTF8File -path ".\src\visualize_run.py" -content $visualize_run

# src/train.py
$train = @'
# train.py
import os
import pandas as pd
from synthetic_data_pipeline.trainer import ModelTrainer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "augmented_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.csv")

def main():
    df = pd.read_csv(DATA_PATH)
    trainer = ModelTrainer(df, target_col="target")
    trainer.prepare()
    results = trainer.train_and_evaluate(out_model_dir=MODEL_DIR, metrics_path=METRICS_PATH)
    print("Training completed. Metrics saved to", METRICS_PATH)
    print(results)

if __name__ == "__main__":
    main()
'@
Write-UTF8File -path ".\src\train.py" -content $train

# requirements.txt
$req = @'
numpy
pandas
scikit-learn
matplotlib
joblib
'@
Write-UTF8File -path ".\requirements.txt" -content $req

# README.md
$readme = @'
# Synthetic Data Pipeline Project (Week 3)

Folders:
- `src/` - executable scripts for each pipeline stage.
- `synthetic_data_pipeline/` - package modules.
- `data/raw`, `data/processed` - datasets.
- `plots/` - saved plots.
- `models/` - saved models.
- `logs/` - error logs.

## Steps to run
1. Install requirements:
```powershell
pip install -r requirements.txt

python src/generate.py

python src/prepare.py

python src/augment_run.py

python src/visualize_run.py

python src/train.py
## Notes
- Error logs written to `logs/errors.txt`.
- Models saved into `models/`.
- Metrics saved into `results/metrics.csv`.
'@
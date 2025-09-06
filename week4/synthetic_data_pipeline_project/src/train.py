import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Paths
INPUT_PATH = "data/processed/augmented_data.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RESULTS_DIR, "metrics.csv")


def encode_categoricals(df):
    """
    Encode categorical columns (e.g., 'gender_original') into numeric.
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        print(f"🔄 Encoding categorical columns: {list(categorical_cols)}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def train_and_evaluate():
    # Load data
    df = pd.read_csv(INPUT_PATH)

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Encode categoricals
    X = encode_categoricals(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        print(f"✅ {name} trained & saved.")

    # Save metrics to CSV
    pd.DataFrame(metrics).T.to_csv(OUTPUT_PATH)
    print(f"📊 Training results saved to {OUTPUT_PATH}")
    print(metrics)


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        with open("logs/errors.txt", "a") as f:
            f.write(f"❌ Error in train.py: {e}\n")
        print(f"❌ Error in train.py: {e}")

# synthetic_data_pipeline/trainer.py
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelTrainer:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate(self):
        # Train
        self.model.fit(self.X_train, self.y_train)

        # Predict
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1] if hasattr(
            self.model, "predict_proba") else None

        # Metrics
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, zero_division=0),
            "recall": recall_score(self.y_test, y_pred, zero_division=0),
            "f1": f1_score(self.y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(self.y_test, y_prob) if y_prob is not None else None,
        }

        # Save results
        os.makedirs("results", exist_ok=True)
        pd.DataFrame([metrics]).to_csv("results/metrics.csv", index=False)

        return metrics

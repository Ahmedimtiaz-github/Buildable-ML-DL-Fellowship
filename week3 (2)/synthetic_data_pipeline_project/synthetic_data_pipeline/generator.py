# synthetic_data_pipeline/generator.py
import numpy as np
import pandas as pd
import random
import os


class DataGenerator:
    def __init__(self, n_rows=500, random_state=42):
        self.n_rows = n_rows
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

    def generate(self):
        data = {
            "age": np.random.randint(18, 70, size=self.n_rows),
            "income": np.random.randint(20000, 120000, size=self.n_rows),
            "score": np.random.normal(50, 15, size=self.n_rows).astype(int),
            "experience": np.random.randint(0, 40, size=self.n_rows),
            "hours_per_week": np.random.randint(20, 60, size=self.n_rows),
            "gender": np.random.choice(["Male", "Female"], size=self.n_rows),
            "product_type": np.random.choice(["A", "B", "C"], size=self.n_rows),
            "target": np.random.choice([0, 1], size=self.n_rows),
        }
        return pd.DataFrame(data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.generate()
        df.to_csv(path, index=False)
        return df

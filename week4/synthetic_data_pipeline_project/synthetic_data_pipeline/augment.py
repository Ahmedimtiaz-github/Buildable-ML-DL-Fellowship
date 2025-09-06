import pandas as pd
import numpy as np
import os

def augment_with_jitter(input_path="data/processed/cleaned_data.csv",
                        out_path="data/processed/augmented_data.csv",
                        factor: float = 2.0,
                        jitter_scale: float = 0.02,
                        random_state: int = 42):
    """
    Duplicate rows with small Gaussian jitter on numeric columns until dataset size = orig * factor.
    """
    df = pd.read_csv(input_path)
    rng = np.random.default_rng(random_state)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")

    orig_n = df.shape[0]
    target_n = int(orig_n * factor)
    if target_n <= orig_n:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"✅ Augmented data saved to {out_path} (no augmentation needed)")
        return df

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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    augmented.to_csv(out_path, index=False)
    print(f"✅ Augmented data saved to {out_path}")
    return augmented

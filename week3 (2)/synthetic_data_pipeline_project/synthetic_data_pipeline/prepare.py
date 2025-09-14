import pandas as pd
import os

def clean_data(input_path="data/raw/generated_data.csv", out_path="data/processed/cleaned_data.csv"):
    df = pd.read_csv(input_path)
    df = df.copy()

    # Basic cleaning
    df = df.dropna()
    if "income" in df.columns:
        df = df[df["income"] > 0]

    # Keep a copy of original categorical 'gender' if present for plotting convenience
    if "gender" in df.columns:
        df["gender_original"] = df["gender"]

    # One-hot encode categorical columns except any _original copies
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if not c.endswith("_original")]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Cleaned data saved to {out_path}")
    return df

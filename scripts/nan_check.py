import pandas as pd
from pathlib import Path
ROOT = Path.cwd()
p = ROOT / "data" / "processed" / "crop"
for fn in ["train.csv","val.csv","test.csv"]:
    path = p / fn
    if path.exists():
        df = pd.read_csv(path)
        print(f"\n--- {fn} shape: {df.shape} ---")
        print(df.isna().sum().sort_values(ascending=False).head(20))
    else:
        print(f"{fn} not found at {path}")

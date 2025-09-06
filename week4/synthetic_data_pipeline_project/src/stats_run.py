import os
import traceback
import pandas as pd
from synthetic_data_pipeline.stats import save_stats

def main():
    try:
        df = pd.read_csv("data/processed/augmented_data.csv")
        save_stats(df, ["age", "income"], out_path="results/stats.csv")
        print("✅ Stats saved to results/stats.csv")
    except Exception as e:
        os.makedirs("logs", exist_ok=True)
        with open("logs/errors.txt","a") as f:
            f.write("Error in src/stats_run.py:\n")
            f.write(str(e) + "\n")
            import traceback as tb; f.write(tb.format_exc() + "\n")
        print("❌ Error logged to logs/errors.txt")

if __name__ == "__main__":
    main()

import os
import traceback
from synthetic_data_pipeline.visuals import save_hist, save_corr_heatmap, save_bar, save_scatter
import pandas as pd

def main():
    try:
        csv = "data/processed/augmented_data.csv"
        # histogram
        save_hist(csv, "age", "plots/histogram_age.png")
        # correlation heatmap
        save_corr_heatmap(csv, "plots/corr_heatmap.png")
        # bar plot (try "gender" original or reconstructed from dummies)
        try:
            save_bar(csv, "gender", "plots/bar_gender.png")
        except Exception as e:
            # fallback to first categorical-like dummy
            df = pd.read_csv(csv)
            # attempt to pick any original-col stored like 'gender_original'
            if "gender_original" in df.columns:
                df["gender_original"].to_csv("plots/placeholder_gender.csv", index=False)
                save_bar(csv, "gender_original", "plots/bar_gender.png")
            else:
                # try product if present
                if "product" in df.columns:
                    save_bar(csv, "product", "plots/bar_product.png")
        # scatter age vs income
        save_scatter(csv, "age", "income", "plots/scatter_age_income.png")
        print("✅ Plots saved to plots/")
    except Exception as e:
        os.makedirs("logs", exist_ok=True)
        with open("logs/errors.txt", "a") as f:
            f.write("Error in src/visualize_run.py:\n")
            f.write(str(e) + "\n")
            import traceback as tb; f.write(tb.format_exc() + "\n")
        print("❌ Error logged to logs/errors.txt")

if __name__ == "__main__":
    main()

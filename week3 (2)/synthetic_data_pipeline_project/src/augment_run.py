import os
import traceback
from synthetic_data_pipeline.augment import augment_with_jitter

def main():
    try:
        augment_with_jitter(input_path="data/processed/cleaned_data.csv",
                            out_path="data/processed/augmented_data.csv",
                            factor=2.0, jitter_scale=0.02)
    except Exception as e:
        os.makedirs("logs", exist_ok=True)
        with open("logs/errors.txt", "a") as f:
            f.write("Error in src/augment_run.py:\n")
            f.write(str(e) + "\n")
            import traceback as tb; f.write(tb.format_exc() + "\n")
        print("❌ Error logged to logs/errors.txt")

if __name__ == "__main__":
    main()

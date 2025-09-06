import os
import traceback
from synthetic_data_pipeline.prepare import clean_data

def main():
    try:
        clean_data(input_path="data/raw/generated_data.csv",
                   out_path="data/processed/cleaned_data.csv")
    except Exception as e:
        os.makedirs("logs", exist_ok=True)
        with open("logs/errors.txt", "a") as f:
            f.write("Error in src/prepare.py:\n")
            f.write(str(e) + "\n")
            import traceback as tb; f.write(tb.format_exc() + "\n")
        print("❌ Error logged to logs/errors.txt")

if __name__ == "__main__":
    main()

# src/generate.py
import os
import traceback
from synthetic_data_pipeline.generator import DataGenerator

def main():
    try:
        gen = DataGenerator(n_rows=500)
        gen.save("data/raw/generated_data.csv")
        print("✅ Data generated and saved to data/raw/generated_data.csv")
    except Exception as e:
        os.makedirs("logs", exist_ok=True)
        with open("logs/errors.txt", "a") as f:
            f.write("Error in generate.py:\n")
            f.write(str(e) + "\n")
            f.write(traceback.format_exc() + "\n\n")
        # Also print the error to terminal for debugging
        print("❌ Error in generate.py:", e)

if __name__ == "__main__":
    main()

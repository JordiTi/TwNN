import os
import pandas as pd
from pathlib import Path
import sys

"""
Normalize the trajectory data by subtracting the initial position from all subsequent positions.
This ensures that the trajectory starts from the origin (0,0).
This is the final step before the data is ready for training.
"""

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

INPUT_DIR = ROOT_DIR / "data/handwriting/digits_upsampled"
OUTPUT_DIR = ROOT_DIR / "data/handwriting/digits_normalized/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize(df):
    # Get the initial position (first row)
    initial_x = df['x'].iloc[0]
    initial_y = df['y'].iloc[0]
    
    # Normalize by subtracting initial position
    df['x'] = df['x'] - initial_x
    df['y'] = df['y'] - initial_y

    return df


def main():
    # Process each CSV file
    files = INPUT_DIR.glob("digit*.csv")
    for file in files:

        filename = str(file).split("/")[-1]
        
        print(f" Processing {file}")

        output_path = os.path.join(OUTPUT_DIR, filename.replace("upsampled", "normalized"))

        df = pd.read_csv(file)
        df_normalized = normalize(df)

        # Save to output directory
        df_normalized.to_csv(output_path, index=False)

        if not df_normalized.empty:
            print(f"Processed {filename}")
        else:
            raise Exception("Empty dataframe")

if __name__ == "__main__":
    main()
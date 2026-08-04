import pandas as pd
from pathlib import Path
import glob
import sys
import matplotlib.pyplot as plt

"""
Plots the normalized digit trajectories
"""

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
INPUT_DIR = ROOT_DIR / "data/handwriting/digits_normalized"



def plot(axes, idx, df):
# Plot the trajectory (x vs y)
        axes[0, idx].plot(df['x'], df['y'], 'b-', linewidth=2)
        axes[0, idx].set_aspect('equal')
        axes[0, idx].invert_yaxis()  # Invert y-axis as typical for image coordinates
        axes[0, idx].set_title(f"Trajectory {idx}")
        axes[0, idx].grid(True, alpha=0.3)
        
        # Plot x stroke (x over time)
        axes[1, idx].plot(df['x'], 'r-', linewidth=2)
        axes[1, idx].set_title(f"X Stroke {idx}")
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].set_ylabel('X Position')
        
        # Plot y stroke (y over time)
        axes[2, idx].plot(df['y'], 'g-', linewidth=2)
        axes[2, idx].set_title(f"Y Stroke {idx}")
        axes[2, idx].grid(True, alpha=0.3)
        axes[2, idx].set_ylabel('Y Position')
        axes[2, idx].set_xlabel('Time Step')

def main():
    # Get all trajectory files
    trajectory_files = INPUT_DIR.glob("*.csv")

    # Create a figure with subplots (3 rows: trajectory, x stroke, y stroke)
    _, axes = plt.subplots(3, 5, figsize=(15, 10))
    for idx, file in enumerate(trajectory_files):
        if idx < 5:
            continue
        idx = idx - 5
        # Read the CSV file

        df = pd.read_csv(file)

        plot(axes, idx, df)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
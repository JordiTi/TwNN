import pandas as pd
from pathlib import Path
import glob

import matplotlib.pyplot as plt

# Get all trajectory files
trajectory_files = sorted(glob.glob('trajectory_*.csv'))

# Create a figure with subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for idx, file in enumerate(trajectory_files):
    if idx >= 10:
        break
    
    # Read the CSV file
    df = pd.read_csv(file)
    # Plot the trajectory
    axes[idx].plot(df['x'], df['y'], 'b-', linewidth=2)
    axes[idx].set_aspect('equal')
    axes[idx].invert_yaxis()  # Invert y-axis as typical for image coordinates
    axes[idx].set_title(f"{idx}")
    axes[idx].grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(trajectory_files), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig("digits.png")
plt.show()
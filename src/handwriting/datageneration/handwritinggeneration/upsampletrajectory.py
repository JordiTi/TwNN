import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import sys

"""
Upsamples pen trajectory to x number of points. 
"""


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

INPUT_DIR = ROOT_DIR / "data/handwriting/digits_raw/"
OUTPUT_DIR = ROOT_DIR / "data/handwriting/digits_upsampled/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def upsample_trajectory(trajectory, target_points=800):

    n_points = len(trajectory)
    
    # Create interpolation function for each coordinate
    t_original = np.linspace(0, 1, n_points)
    t_new = np.linspace(0, 1, target_points)
    
    # Interpolate x and y separately
    f_x = interp1d(t_original, trajectory[:, 0], kind='cubic')
    f_y = interp1d(t_original, trajectory[:, 1], kind='cubic')
    
    upsampled_x = f_x(t_new)
    upsampled_y = f_y(t_new)
    
    return np.column_stack([upsampled_x, upsampled_y])

# Load trajectory from CSV file
def load_trajectory_from_csv(filepath):
    """Load pen trajectory from CSV file with time, x, y columns."""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return data[:, 2:4]  # Return only x, y columns

def getfilename(filepath):
    return str(filepath).split("/")[-1]


def main():
    # Load and upsample

    print(" Loading files...")
    trajectory_files = INPUT_DIR.glob('digit*.csv')

    for file in trajectory_files:

        print(f" Processing {file}")
        data = load_trajectory_from_csv(file)
        upsampled = upsample_trajectory(data, target_points=800)

        # Save upsampled trajectory
        filename = getfilename(file)
        output_path = OUTPUT_DIR / str(filename).replace('.csv', '_upsampled.csv')
        np.savetxt(output_path, upsampled, delimiter=',', header='x,y', comments='')
        print(f" Processed {filename}")

if __name__ == "__main__":
    main()
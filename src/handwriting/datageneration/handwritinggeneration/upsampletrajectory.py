import numpy as np
from scipy.interpolate import interp1d
import os
import glob

def upsample_trajectory(trajectory, target_points=800):
    """
    Upsample pen-tip trajectory to target number of points.
    
    Args:
        trajectory: numpy array of shape (n, 2) with x, y coordinates
        target_points: desired number of points in output
    
    Returns:
        upsampled trajectory of shape (target_points, 2)
    """
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


# Example usage:
# trajectory = np.array([[x1, y1], [x2, y2], ...])
# upsampled = upsample_trajectory(trajectory, target_points=800)
# Load trajectory from CSV file
def load_trajectory_from_csv(filepath):
    """Load pen trajectory from CSV file with time, x, y columns."""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return data[:, 1:3]  # Return only x, y columns

# Load and upsample
trajectory_files = sorted(glob.glob('trajectory_*.csv'))
for idx, file in enumerate(trajectory_files):
    data = load_trajectory_from_csv(file)
    upsampled = upsample_trajectory(data, target_points=800)

    # Save upsampled trajectory
    output_path = os.path.join(
        os.path.dirname('./'),
        file.replace('.csv', '_upsampled.csv')
    )
    np.savetxt(output_path, upsampled, delimiter=',', header='x,y', comments='')
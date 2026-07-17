import sys
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

'''
Visualizes sine, gaussian and block pulse
Images are plotted in root/data/basefunctions/
'''

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data/basefunctions/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_base_function_paths() -> list[Path]:

    return list(DATA_DIR.glob("*.txt"))

def plotbasefunctions():

    file_paths = get_base_function_paths()

    for path in file_paths:
        print(DATA_DIR, path.stem)
        try:
            # 1. Load the data from the text file
            data = np.loadtxt(path)
            x = data[:, 0]
            y = data[:, 1]
            
            # 2. Create new window
            plt.figure(figsize=(8, 5))
            
            # 3. Plot the data and customize this specific chart
            plt.plot(x, y, color="blue", linewidth=2)
            plt.title(f"Base Function: {path.stem}")  # Displays file name as title
            plt.xlabel("X")
            plt.ylabel("Amplitude")
            plt.grid(True)
            
            # 4. Display the window and pause execution until you close it

            plt.savefig(f"{str(DATA_DIR)}/{path.stem}.png")
            plt.clf()
            
        except Exception as e:
            print(f"Could not plot file {path.name} due to error: {e}")
    
if __name__ == "__main__":
    plotbasefunctions()
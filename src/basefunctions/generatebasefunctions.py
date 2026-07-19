import sys
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

'''
Generates sine, gaussian or block pulse
Ensures signal starts at 1/4th of total trial time to allow early outputs
Ensures signal ends at 3/4th of total trial time to allow late outputs

Parameters:
-----------
name : name of function to generate
       choose from "sine, gaussian, or square"

length : total length of the signal in datapoints

width : 1/(2*var)
'''

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data/basefunctions/inputfiles"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generatetargetfunction(name: str, length: int, width: float):

    output = np.zeros(length)
    if name in ["sine", "all"]:
        x = np.linspace(0, 2*np.pi, int(length*0.5)) # Start from 0, end with 2*PI
        output[int(length*0.25):int(length*0.75)] = np.sin(x) # Shift to the middle of the signal
        output = output/np.sum(abs(output))
        np.savetxt(str(DATA_DIR / f"sine_{length}.txt"), np.array([np.linspace(0, len(output), len(output)), output]).T)
        print(f"Saved sine in {str(DATA_DIR / f"sine_{length}.txt")}")
    
    if name in ["gauss", "all"]:
        x = np.linspace(-length/2, length/2, int(length/2))
        output[int(length*0.25):int(length*0.75)] = np.exp(-width*(x)**2)
        output = output/np.sum(abs(output))
        np.savetxt(str(DATA_DIR / f"gauss_l{length}_w{width}.txt"), np.array([np.linspace(0, len(output), len(output)), output]).T)
        print(f"Saved gauss in {str(DATA_DIR / f"gauss_{length}_{width}.txt")}")
    
    if name in ["block", "all"]:
        output[int(length*0.25):int(length*0.75)] = 1
        output = output/np.sum(abs(output))
        np.savetxt(str(DATA_DIR / f"block_{length}.txt"), np.array([np.linspace(0, len(output), len(output)), output]).T)
        print(f"Saved block in {str(DATA_DIR / f"block_{length}.txt")}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Define what flags this specific script accepts
    parser.add_argument("--name", type=str, default="all")
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--width", type=float, default=0.0001)
    args = parser.parse_args()

    name = args.name
    length = args.length
    width = args.width
    generatetargetfunction(name, length, width)
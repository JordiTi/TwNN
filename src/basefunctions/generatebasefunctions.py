import sys
import numpy as np
from pathlib import Path

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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data/basefunctions/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generatetargetfunction(name, length, width):

    output = np.zeros(length)
    if name == "sine":
        x = np.linspace(0, 2*np.pi, int(length*0.5)) # Start from 0, end with 2*PI
        output[int(length*0.25):int(length*0.75)] = np.sin(x) # Shift to the middle of the signal
        np.savetxt(DATA_DIR / f"sine_{length}.txt")
        return output
    elif name == "gaussian":
        x = np.linspace(-length/2, length/2, int(length/2))
        output[int(length*0.25):int(length*0.75)] = np.exp(-width*(x)**2)
        np.savetxt(DATA_DIR / f"gauss_l{length}_w{width}")
        return  output
    elif name == "square":
        output[int(length*0.25):int(length*0.75)] = 1
        np.savetxt(DATA_DIR / f"block_{length}")
        return output
    
if __name__ == "__main__":
    generatetargetfunction()
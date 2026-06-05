import numpy as np
import math

# Function that can generate sine+1, gaussian, sinc, square
# Functions start at x = 1/4*length, and end at 3/4*length, so that the function starts at 0 and ends at 0, and is in the middle of the trial
def generatetargetfunction(name, length, width):
    # Sine + 1 for positive signal
    # Sine(1/2*pi) so sine starts at 0
    output = np.zeros(length)
    if name == "sine":
        x = np.linspace(0, 2*np.pi, int(length*0.5))
        output[int(length*0.25):int(length*0.75)] = np.sin(x) 
        return output
    elif name == "gaussian":
        x = np.linspace(-length/2, length/2, int(length/2))
        output[int(length*0.25):int(length*0.75)] = np.exp(-width*(x)**2)
        return  output
    elif name == "square":
        output[int(length*0.25):int(length*0.75)] = 1
        return output
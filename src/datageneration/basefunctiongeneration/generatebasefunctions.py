import numpy as np
import math

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
def generatetargetfunction(name, length, width):

    output = np.zeros(length)
    if name == "sine":
        x = np.linspace(0, 2*np.pi, int(length*0.5)) # Start from 0, end with 2*PI
        output[int(length*0.25):int(length*0.75)] = np.sin(x) # Shift to the middle of the signal
        np.savetxt(f"sine_{length}.txt")
        return output
    elif name == "gaussian":
        x = np.linspace(-length/2, length/2, int(length/2))
        output[int(length*0.25):int(length*0.75)] = np.exp(-width*(x)**2)
        return  output
    elif name == "square":
        output[int(length*0.25):int(length*0.75)] = 1
        return output
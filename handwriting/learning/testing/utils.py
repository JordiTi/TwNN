import numpy as np
import math

# Generates the target jet
def generatetargetjet(runin, maxjetspeed, flyspeed, totallength, dt, delay=0):
    totaltime = totallength/flyspeed + delay*dt
    timesteps = np.arange(0, totaltime, dt)
    targetjet = np.zeros(len(timesteps) + delay)
    flyposhist = np.zeros(len(timesteps) + delay)

    diagonaldistance = 0.3/np.sin(0.25*np.pi)

    pos_fly = 0
    fly_eta = 0.3/flyspeed

    for idx, t in enumerate(timesteps):

        if diagonaldistance/maxjetspeed < fly_eta and t > runin:

            jetspeed_required = diagonaldistance/fly_eta
        else:
            jetspeed_required = 0
        fly_eta -= dt
        pos_fly += dt * flyspeed

        flyposhist[idx] = pos_fly
        targetjet[idx] = jetspeed_required
    return targetjet
    
def flyinput_guassian_deterministic(flyspeed, dt, exponent, neuronthreshold, nsensorneurons, totallength, delay=200):
    sensorcenters = np.linspace(-totallength/3, totallength + totallength/3, nsensorneurons)
    dhist = []
    thist = []
    simulationtime = totallength/flyspeed + delay*dt
    timesteps = np.arange(0, simulationtime, dt)
    sensoryneuronsignals = [] # Every row is one timestep
    # Create Gaussian values that map fly position to neuron input
    for t in timesteps:
        d = flyspeed * t
        thist.append(t)
        dhist.append(d)
        sensoroutputs = []
        for s in sensorcenters:
            sensoroutputs.append(math.exp(exponent*(s-d)**2))

        sensoryneuronsignals.append(sensoroutputs.copy())
    sensoryneuronsignals = np.array(sensoryneuronsignals)

    # Now convert gaussian to inputs
    neuroncharges = np.zeros(nsensorneurons)
    spikehist = np.zeros(sensoryneuronsignals.shape)
    for i in range(len(sensoryneuronsignals)):
        gaussianvalues = sensoryneuronsignals[i, :]
        neuroncharges += gaussianvalues
        spikes = neuroncharges > neuronthreshold
        spikehist[i,:] = spikes
        neuroncharges[neuroncharges > neuronthreshold] -= neuronthreshold
    return spikehist

# Converts velocity curve to droplet end x-positions
def curvetodroplet(velocitycurve, dt, fishangle, flyspeed, targetflyx, targetflyy=0.3):
    droplet_xpositions = []
    droplet_ypositions = []
    timesteps = np.arange(0, len(velocitycurve)*dt, dt)
    for idx, speed in enumerate(velocitycurve):
        t = idx * dt
        currenttime = timesteps[idx]
        remaining_flydistance = targetflyx - flyspeed * currenttime
        droplet_yspeed = np.sin(fishangle) * speed
        droplet_xspeed = np.cos(fishangle) * speed

        eta_fly = targetflyx / flyspeed - t
        travel_droplet_x = eta_fly * droplet_xspeed
        travel_droplet_y = eta_fly * droplet_yspeed
        droplet_xpositions.append(travel_droplet_x)
        droplet_ypositions.append(travel_droplet_y)

    return droplet_xpositions, droplet_ypositions

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
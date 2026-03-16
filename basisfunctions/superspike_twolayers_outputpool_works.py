import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append('../')
import utils

"""Script for fitting the fish output using pools of motor units with different time constants and amplitudes."""

class Neurons:
    def __init__(self, ninputs, noutputs, amplitudes):
        self.weights = np.random.uniform(-1, 1, [ninputs, noutputs]) # input weights
        self.threshold = 2  # Membrane threshold
        self.urest = 0  # rest potential
        self.Vs = np.ones(noutputs)*self.urest  # Membrane potentail
        self.Isyns = np.ones(noutputs)*self.urest  # Synaptic input current, evolves with exponential decay
        self.taumem = 30
        self.trest = 0
        self.tausyn = 15
        self.tauref = 50
        self.tau_rise = 20
        self.tau_decay = 15
        self.axonaldelay = 0
        self.spike = 0
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.amplitudes = amplitudes

        self.actionpotential = 0
        self.Isynhist = []
        self.vhist = []
        self.tracehist = []
        self.vderivhist = []
        self.lambdahist = []
        self.lambdahist2 = []

        self.pre_trace1 = np.zeros(ninputs, dtype=np.float32)
        self.pre_trace2 = np.zeros(ninputs, dtype=np.float32)

        self.surrogategradient = 0
        self.lambda_1 = np.zeros(self.ninputs)
        self.lambda_2 = np.zeros(self.ninputs)

        self.beta = 1
        self.dothist = 0
        self.dwhist = []

        self.chunktime = 0
        self.finalchunktime = 53

        self.dw_running = np.zeros([self.ninputs, self.noutputs])



    def update_state(self, S, dt):
        # Input current streams in
        self.pre_trace1 = self.pre_trace1*np.exp(-dt/self.tausyn) +  S
        self.pre_trace2 = self.pre_trace2*np.exp(-dt/self.tau_decay) + (1-np.exp(-dt/self.tausyn))*self.pre_trace1
        self.tracehist.append(self.pre_trace2.copy())
        # self.Isynhist.append(self.Isyn.copy())
        # Update membrane potential
        # Membrane voltage increase
        self.Isyns = self.Isyns*np.exp(-dt/self.tausyn) + np.dot(self.weights.T, S)
        self.Vs = self.Vs*np.exp(-dt/self.taumem) + self.Isyns * (1-np.exp(-dt/self.taumem))
        self.vhist.append(self.Vs.copy())

        # Surrogate gradient
        self.surrogategradient = 1/((1+np.abs(self.beta*(self.Vs - self.threshold)))**2)
        # self.vderivhist.append(self.surrogategradient)

        # Outer product of presynaptic trace with postsynaptic partial derivative
        outer_hebbian = np.outer(self.pre_trace2, self.surrogategradient)
        outer_hebbian[outer_hebbian < 1e-7] = 0
        self.lambda_1 = self.lambda_1*np.exp(-dt/self.tau_rise) + ((1-np.exp(-dt/self.tau_rise))*outer_hebbian).T
        self.lambda_2 = self.lambda_2*np.exp(-dt/self.tau_decay) + (1-np.exp(-dt/self.tau_decay))*self.lambda_1
        # self.lambdahist.append(self.lambda_2)
        self.lambdahist2.append(self.lambda_2)

        # Voltage has crossed threshold. Now wait before transmitting and initialize refactory period
        spikes = self.Vs > self.threshold
        self.Vs[self.Vs > self.threshold] = 0
        return spikes

    def update_weight(self, error, lr, dt):
        dw = np.multiply(self.lambda_2.T , error)

        self.chunktime += dt
        if self.chunktime >= self.finalchunktime:
            self.chunktime = 0
            self.weights += lr *  self.dw_running 
            self.weights = np.clip(self.weights, -1, 1)
            self.dw_running = np.zeros([self.ninputs, self.noutputs])

        self.dw_running += dw

    def reset(self):
        self.Vs = np.ones(self.noutputs)*self.urest
        self.Isyn = 0
        self.actionpotential = 0
        self.dwhist = []
        self.Isynhist = []
        self.vhist = []
        self.tracehist = []
        self.vderivhist = []
        self.lambdahist = []
        self.lambdahist2 = []
        self.previousv = 0
        self.vderiv = 0
        self.surrogategradient = 0
        self.lambda_1 = np.zeros(self.ninputs)
        self.lambda_2 = np.zeros(self.ninputs)

        self.pre_trace1 = np.zeros(self.ninputs, dtype=np.float32)
        self.pre_trace2 = np.zeros(self.ninputs, dtype=np.float32)

# Initialize animals
# Note that the speed of the water needs to be similar to the insect fly speed, in order to involve timing
# housefly can go up to 7 kmh, 1.94 m/s. Let's take 2
# 5.8-6.8 m/s is the water jet maximum speed (gerullis 2014). Let's take 6.
# If the angle w.r.t. the water is always 45 degrees, and the insect flies 30 cm above the fish, the the water needs to travel 42.4 cm before hitting the fly
# So it needs 0.07s or 14cm of housefly flight time/distance given that it can immediately fire. Does it work when it
# The fly will travel 16cm or 0.08s before the fish needs to shoot. What about 50cm environment, so 0.25s sim time.
# Environment: 50 cm
# Fly speed: 2 m/s
# Simulation time: 250ms
# Dt: 1ms
# Fish location: 0
# Fly hit distance (Diagonal): 42.4cm
# Max water jet speed: 6m/s
# Shoot until: 0.0706s before fly reaches

# Generate target jet

neuronthreshold = 5
nsensorneurons = 50
nalphamneurons = 100
lr = 0.1

timesteps = 1000
# function generator
sine = utils.generatetargetfunction("sine", timesteps, 1)
gaussian = utils.generatetargetfunction("gaussian", timesteps, 0.0001)
square = utils.generatetargetfunction("square",timesteps, 1)

# Normalize to area of 1
sine = sine/np.sum(np.abs(sine))
gaussian = gaussian/np.sum(gaussian)
square = square/np.sum(square)

# Print the areas under the functions
print("Sine area:", np.sum(np.abs(sine)))
print("Gaussian area:", np.sum(gaussian))
print("Square area:", np.sum(square))
# fig, axs = plt.subplots(2,2, figsize=(10,10))
# axs[0,0].plot(sine, color="red")
# axs[0,1].plot(gaussian, color="green")
# axs[1,1].plot(square)
# plt.show()
functions = {"sine": sine, "gaussian": gaussian, "square": square}
functiontofit = "sine"
target = functions[functiontofit]
amplitudes = np.random.uniform(0.05, 0.1, nalphamneurons)
# Half of the amplitudes are negative
amplitudes[:int(nalphamneurons/2)] *= -1
# Create neural network
outputs = Neurons(nsensorneurons, nalphamneurons, amplitudes)
inputs = np.random.choice([0, 1],[nsensorneurons, timesteps], p=[0.99, 0.01])


tau_rises = np.random.uniform(20, 50, nalphamneurons)
tau_decays = np.random.uniform(20, 50, nalphamneurons)
outputs.tau_rises = tau_rises
outputs.tau_decays = tau_decays

ntrials = 1000
trialtime = timesteps
losses = np.zeros(ntrials)
dt = 1
for t in range(ntrials):
    if t%99 == 0:
        print(t)


    spikehist = np.zeros([nalphamneurons, int(trialtime/dt)])
    errorhist = []
    spikefilterhist = np.zeros([nalphamneurons, int(trialtime/dt)])
    totalerr = 0
    error1 = 0
    error2 = 0

    spikefilter1 = np.zeros(nalphamneurons)
    spikefilter2 = np.zeros(nalphamneurons)

    for idx, inp in enumerate(inputs.T):
        spikes = outputs.update_state(inp, dt)

        spikefilter1 = spikefilter1*np.exp(-dt/outputs.tau_rises) + (1-np.exp(-dt/outputs.tau_rises) ) * spikes * amplitudes
        spikefilter2 = spikefilter2*np.exp(-dt/outputs.tau_decays) + (1-np.exp(-dt/outputs.tau_decays) ) * spikefilter1
        spikefilterhist[:,idx] = spikefilter2.copy()
        outputs.update_weight((target[idx] - np.sum(spikefilter2.copy()))* amplitudes/0.1, lr, dt)
        errorhist.append(target[idx] - np.sum(spikefilter2.copy()))

    # plt.plot(np.array(spikefilterhist).T)
    # plt.show()
    if (t <100 and t%10 == 0) or (t%100 == 0):
        plt.plot(sum(spikefilterhist), label="out")
        plt.plot(target,label="target")
        plt.legend()
        plt.savefig(f"../data/images/basisfunctions/{functiontofit}/individual_{t}")
        plt.clf()
        plt.plot(sum(spikefilterhist))
        plt.plot(spikefilterhist.T)
        plt.savefig(f"../data/images/basisfunctions/{functiontofit}/sum_{t}")
        plt.clf()
        plt.plot(errorhist)
        plt.savefig(f"../data/images/basisfunctions/{functiontofit}/error_{t}")
        plt.clf()
        plt.plot(np.array(outputs.lambdahist2)[:,1,1:5])
        plt.savefig(f"../data/images/basisfunctions/{functiontofit}/lambda_{t}")
        plt.clf()
    losses[t] = np.array(1/2*np.array(errorhist)**2).mean()
    # if t == ntrials-1:
    #     np.savetxt("data/outputpool/vs.txt", np.array(outputs.vhist))
    #     plt.clf()
    #     plt.plot(np.array(outputs.vhist))
    #     plt.savefig("imgs/outputpool/lastvs.png")
    #     plt.clf()

    outputs.reset()

# plt.plot(sum(spikefilterhist), label="out")
# plt.plot(target_out,label="target")
# plt.legend()
# plt.savefig(f"imgs/outputpool/{t}")
# plt.clf()
# plt.plot(sum(spikefilterhist))
# plt.plot(spikefilterhist.T)
# plt.savefig(f"imgs/outputpool/spikefilter_{t}")
# plt.clf()

# plt.plot(losses)
# plt.savefig("imgs/outputpool/loss")



import numpy as np
import copy
import sys
import utils
import getopt
import os
import matplotlib.pyplot as plt
import pandas as pd

"""Script for fitting the fish output using pools of motor units with different time constants and amplitudes."""

class Neurons:
    def __init__(self, ninputs, noutputs, amplitudes):
        self.weights = np.random.uniform(-1, 1, [ninputs, noutputs]) # input weights
        self.threshold = 10  # Membrane threshold
        self.urest = 0  # rest potential
        self.Vs = np.ones(noutputs)*self.urest  # Membrane potentail
        self.Isyns = np.ones(noutputs)*self.urest  # Synaptic input current, evolves with exponential decay

        # Prresynaptic traces
        self.tau_pre_1 = 25
        self.tau_pre_2 = 25

        # Presynaptic membrane constants
        self.tau_syn = 25
        self.tau_mem = 25

        # eligibility trace time constants
        self.tau_elig_rise = 25
        self.tau_elig_decay = 25

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
        self.finalchunktime = 523

        self.dw_running = np.zeros([self.ninputs, self.noutputs])


    def update_state(self, S, dt):
        # Input current streams in
        self.pre_trace1 = self.pre_trace1*np.exp(-dt/self.tau_pre_1) +  S
        self.pre_trace2 = self.pre_trace2*np.exp(-dt/self.tau_pre_2) + (1-np.exp(-dt/self.tau_pre_2))*self.pre_trace1
        self.tracehist.append(self.pre_trace2.copy())
        # self.Isynhist.append(self.Isyn.copy())
        # Update membrane potential
        # Membrane voltage increase
        self.Isyns = self.Isyns*np.exp(-dt/self.tau_syn) + np.dot(self.weights.T, S)
        self.Vs = self.Vs*np.exp(-dt/self.tau_mem) + self.Isyns * (1-np.exp(-dt/self.tau_mem))
        self.vhist.append(self.Vs.copy())

        # Surrogate gradient
        self.surrogategradient = 1/((1+np.abs(self.beta*(self.Vs - self.threshold)))**2)
        # self.vderivhist.append(self.surrogategradient)

        # Outer product of presynaptic trace with postsynaptic partial derivative
        outer_hebbian = np.outer(self.pre_trace2, self.surrogategradient)
        outer_hebbian[outer_hebbian < 1e-7] = 0
        self.lambda_1 = self.lambda_1*np.exp(-dt/self.tau_elig_rise) + ((1-np.exp(-dt/self.tau_elig_rise))*outer_hebbian).T
        self.lambda_2 = self.lambda_2*np.exp(-dt/self.tau_elig_decay) + (1-np.exp(-dt/self.tau_elig_decay))*self.lambda_1
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

# Command line parsing 
opts, args = getopt.getopt(sys.argv[1:], "-l:-m:-a:")
for o, a in opts:
    if o == "-l":
        lr = float(a)
    if o == "-m":
        amplitude_minimum = float(a)
    if o == "-a":
        amplitude_maximum = float(a)

# Load audio data
melfolderpath = "../audio/audio_cut/mel_spectrograms"
melfolder = os.listdir(melfolderpath)
audiodict = {}
for file in melfolder:
    if file.endswith(".txt"):
        audiolabel = file.split("_")[0]
        # 256 * 970
        audiodata = np.loadtxt(melfolderpath + "/" + file)

        # Normalize
        audiodict[audiolabel] = audiodata/-100

# Load handwiring data
textfolderpath = "../text/text_normalized/"
textfolder = os.listdir(textfolderpath)
textdict = {}
for file in textfolder:
    if file.endswith(".csv"):
        textlabel = file.split(".")[0]
        
        # 2*800
        textdata = pd.read_csv(textfolderpath + "/" + file)

        # Append 135 on each side to make the data equally long, also normalize to be between -1,1
        #Convert to np
        textdata = textdata.to_numpy()
        textdata = np.concatenate((np.zeros([85,2]), textdata, np.zeros([85,2])), axis=0)/600
        # fig, ax = plt.subplots(3,1, figsize=(10,6))
        # textdata_nan = textdata.copy()
        # textdata_nan[textdata_nan == 0] = np.nan
        # ax[0].plot(textdata_nan[:,0], textdata[:,1])
        # ax[1].plot(textdata[:,0])
        # ax[1].set_title("x-data")
        # ax[2].plot(textdata[:,1])
        # ax[2].set_title("y-data")
        # plt.show()
        print(np.max(textdata))
        print(np.min(textdata))
        textdict[textlabel] = textdata
        
# Pick number 4 to start with
targetx = textdict["4"][:,0]
targety = textdict["4"][:,1]
# plt.plot(targetx)
# plt.plot(targety)
# plt.show()
# assert()

inputs = audiodict["4"]

# plt.imshow(inputs)
# plt.colorbar()
# plt.show()
# As long as the audio        
nsensorneurons = len(inputs)

nalphamneurons = 200
ntrials = 50000
timesteps = len(inputs[1])
dt  = 1
trialtime = timesteps/dt

# 2 neural nets
amplitudes1 = np.random.uniform(amplitude_minimum , amplitude_maximum, nalphamneurons)
amplitudes1[:int(nalphamneurons/2)] *= -1
nn1 = Neurons(nsensorneurons, nalphamneurons, amplitudes1)
tau_rises1 = np.random.uniform(50, 100, nalphamneurons)
tau_decays1 = np.random.uniform(50, 100, nalphamneurons)
nn1.tau_rises = tau_rises1
nn1.tau_decays = tau_decays1

amplitudes2 = np.random.uniform(amplitude_minimum , amplitude_maximum, nalphamneurons)
amplitudes2[:int(nalphamneurons/2)] *= -1
nn2 = Neurons(nsensorneurons, nalphamneurons, amplitudes1)
tau_rises2 = np.random.uniform(50, 100, nalphamneurons)
tau_decays2 = np.random.uniform(50, 100, nalphamneurons)
nn2.tau_rises = tau_rises2
nn2.tau_decays = tau_decays2

losses = np.zeros(ntrials)
dt = 1
fig, ax = plt.subplots(3,1, figsize=(6,10))

neuronthresholds = 20

numbers = ["3", "4"]
for t in range(ntrials):
    if t%99 == 0:
        print(t)
    #Pick target
    number = np.random.choice(numbers)
    targetx = textdict[number][:,0]
    targety = textdict[number][:,1]
    inputs = audiodict[number]

    errorhist1 = []
    errorhist2 = []
    spikefilterhist1 = np.zeros([nalphamneurons, int(trialtime/dt)])
    spikefilterhist2 = np.zeros([nalphamneurons, int(trialtime/dt)])
    totalerr = 0
    error1 = 0
    error2 = 0

    spikefilter11 = np.zeros(nalphamneurons)
    spikefilter12 = np.zeros(nalphamneurons)

    spikefilter21 = np.zeros(nalphamneurons)
    spikefilter22 = np.zeros(nalphamneurons)

    spikehist = np.zeros([nsensorneurons, 970])
    outspikehist = np.zeros([nalphamneurons, 970])

    incharges = np.zeros(nsensorneurons)

    for idx, inp in enumerate(inputs.T):
        incharges += inp
        spikes = incharges >= neuronthresholds
        incharges[incharges >= neuronthresholds] = 0 

        spikehist[:,idx] = spikes.copy()
        # Run NN
        spikes1 = nn1.update_state(spikes, dt)
        spikes2 = nn2.update_state(spikes, dt)
        outspikehist[:,idx] = spikes2
        # Run muscles
        spikefilter11 = spikefilter11*np.exp(-dt/nn1.tau_rises) + (1-np.exp(-dt/nn1.tau_rises) ) * spikes1 * amplitudes1
        spikefilter12 = spikefilter12*np.exp(-dt/nn1.tau_decays) + (1-np.exp(-dt/nn1.tau_decays) ) * spikefilter11
        spikefilterhist1[:,idx] = spikefilter12.copy()


        # Run muscles
        spikefilter21 = spikefilter21*np.exp(-dt/nn2.tau_rises) + (1-np.exp(-dt/nn2.tau_rises) ) * spikes2 * amplitudes2
        spikefilter22 = spikefilter22*np.exp(-dt/nn2.tau_decays) + (1-np.exp(-dt/nn2.tau_decays) ) * spikefilter21
        spikefilterhist2[:,idx] = spikefilter22.copy()

        #Update without pen up
        if targetx[idx] != 0:
            errorx = targetx[idx] - np.sum(spikefilter12.copy())
        else:
            errorx = 0
        nn1.update_weight(errorx* amplitudes1/amplitude_maximum, lr, dt)
        errorhist1.append(errorx)

        if targety[idx] != 0:
            errory = targety[idx] - np.sum(spikefilter22.copy())
        else:
            errory = 0
        nn2.update_weight(errory* amplitudes2/amplitude_maximum, lr, dt)
        errorhist2.append(errory)

    if (t < 101 and t%10 == 0) or (t<1010 and (t%100 == 0 or t%100 == 1 or t % 100 == 3 or t % 100 == 4)) or (t %1000 == 0 or t % 1000 == 1 or t % 1000 == 2 or t % 1000 == 3 or t % 1000 == 4):
        print(t)
        # pass
        # plt.imshow(outspikehist)
        # plt.show()
        # assert()
        # plt.figure(1)
        # plt.imshow(spikehist)
        # plt.figure(2)
        # plt.imshow(outspikehist)
        # plt.figure(3)
        # plt.plot(spikefilterhist2.T)
        # plt.figure(4)
        # plt.plot(np.sum(spikefilterhist2, axis=0))
        # plt.show()
        # assert()
        # plt.plot(np.sum(spikefilterhist1, axis=1))
        # plt.show()
        # assert()

        ax[0].plot(np.sum(spikefilterhist1, axis=0), np.sum(spikefilterhist2, axis=0), label="output")
        ax[0].plot(targetx, targety, label="target")
        ax[0].invert_yaxis()
        ax[1].plot(np.sum(spikefilterhist1, axis=0), label="output")
        ax[1].plot(targetx, label="target")
        ax[1].set_title("x-data")
        ax[2].plot(targety, label="target")
        ax[2].plot(np.sum(spikefilterhist2, axis=0), label="output")
        ax[2].set_title("y-data")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.savefig(f"./plots/{t}.png")
        for axs in ax:
            axs.cla()

    # losses[t] = np.array(1/2*np.array(errorhist)**2).mean()
    nn1.reset()
    nn2.reset()


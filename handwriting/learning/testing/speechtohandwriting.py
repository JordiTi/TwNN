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
        self.threshold = 2  # Membrane threshold
        self.urest = 0  # rest potential
        self.Vs = np.ones(noutputs)*self.urest  # Membrane potentail
        self.Isyns = np.ones(noutputs)*self.urest  # Synaptic input current, evolves with exponential decay

        # Prresynaptic traces
        self.tau_pre_1 = 15
        self.tau_pre_2 = 15


        # Presynaptic membrane constants
        self.tau_syn = 15
        self.tau_mem = 15

        # eligibility trace time constants
        self.tau_elig_rise = 15
        self.tau_elig_decay = 15

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
        self.finalchunktime = 1

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

opts, args = getopt.getopt(sys.argv[1:], "-l:-m:-a:-i:-f:-t:")
for o, a in opts:
    if o == "-l":
        lr = float(a)
    if o == "-m":
        amplitude_minimum = float(a)
    if o == "-a":
        amplitude_maximum = float(a)
    if o == "-i":
        inputpercentage = float(a)
    if o == "-f":
        functiontofit = a
    if o == "-t":
        trial = int(a)

# Load audio data
melfolderpath = "../audio/audio_cut/mel_spectrograms"
melfolder = os.listdir(melfolderpath)
audiodict = {}
for file in melfolder:
    if file.endswith(".txt"):
        audiolabel = file.split("_")[0]
        # 256 * 970
        audiodata = np.loadtxt(melfolderpath + "/" + file)
        print(len(audiodata[0]))



        audiodict[audiolabel] = audiodata

# Load handwiring data
textfolderpath = "../text/text_normalized/"
textfolder = os.listdir(textfolderpath)
textdict = {}
for file in textfolder:
    if file.endswith(".csv"):
        textlabel = file.split(".")[0]
        
        # 2*800
        textdata = pd.read_csv(textfolderpath + "/" + file)

        # Append 135 on each side to make the data equally long
        #Convert to np
        textdata = textdata.to_numpy()
        print(textdata.shape)
        textdata = np.concatenate((np.zeros([85,2]), textdata, np.zeros([85,2])), axis=0)
        print(textdata.shape)
        fig, ax = plt.subplots(3,1, figsize=(10,6))
        textdata_nan = textdata.copy()
        textdata_nan[textdata_nan == 0] = np.nan
        ax[0].plot(textdata_nan[:,0], textdata[:,1])
        ax[1].plot(textdata[:,0])
        ax[1].set_title("x-data")
        ax[2].plot(textdata[:,1])
        ax[2].set_title("y-data")



# As long as the audio        
nsensorneurons = 256
nalphamneurons = 200

timesteps = 970


target_x = textdata[:,0]
target_y = textdata[:,1]



amplitudes = np.random.uniform(amplitude_minimum , amplitude_maximum, nalphamneurons)
amplitudes[:int(nalphamneurons/2)] *= -1

# Create neural network
outputs = Neurons(nsensorneurons, nalphamneurons, amplitudes)
inputs = np.random.choice([0, 1],[nsensorneurons, timesteps], p=[1-inputpercentage, inputpercentage])
np.savetxt(f"/scratch/p309238/archerfish/basisfunctions/{functiontofit}/lr={lr}_minamp={amplitude_minimum}_maxamp={amplitude_maximum}_inp={inputpercentage}/trial{trial}/inputs.txt", inputs)

tau_rises = np.random.uniform(20, 50, nalphamneurons)
tau_decays = np.random.uniform(20, 50, nalphamneurons)
outputs.tau_rises = tau_rises
outputs.tau_decays = tau_decays

ntrials = 50000
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
    if (t < 100 and t%10 == 0) or (t%1000 == 0) or t == ntrials-1:
        np.savetxt(f"/scratch/p309238/archerfish/basisfunctions/{functiontofit}/lr={lr}_minamp={amplitude_minimum}_maxamp={amplitude_maximum}_inp={inputpercentage}/trial{trial}/actuatorhist{t}.txt", spikefilterhist)
    losses[t] = np.array(1/2*np.array(errorhist)**2).mean()
    outputs.reset()

# Save hyperparameters and loss
np.savetxt(f"/scratch/p309238/archerfish/basisfunctions/{functiontofit}/lr={lr}_minamp={amplitude_minimum}_maxamp={amplitude_maximum}_inp={inputpercentage}/trial{trial}/losses.txt", losses)
np.savetxt(f"/scratch/p309238/archerfish/basisfunctions/{functiontofit}/lr={lr}_minamp={amplitude_minimum}_maxamp={amplitude_maximum}_inp={inputpercentage}/trial{trial}/tau_rises.txt", outputs.tau_rises)
np.savetxt(f"/scratch/p309238/archerfish/basisfunctions/{functiontofit}/lr={lr}_minamp={amplitude_minimum}_maxamp={amplitude_maximum}_inp={inputpercentage}/trial{trial}/tau_decays.txt", outputs.tau_decays)
np.savetxt(f"/scratch/p309238/archerfish/basisfunctions/{functiontofit}/lr={lr}_minamp={amplitude_minimum}_maxamp={amplitude_maximum}_inp={inputpercentage}/trial{trial}/amplitudes.txt", outputs.amplitudes)
np.savetxt(f"/scratch/p309238/archerfish/basisfunctions/{functiontofit}/lr={lr}_minamp={amplitude_minimum}_maxamp={amplitude_maximum}_inp={inputpercentage}/trial{trial}/weights.txt", outputs.weights)

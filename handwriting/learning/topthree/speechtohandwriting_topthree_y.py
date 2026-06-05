import numpy as np
import sys
import getopt
import os
import pandas as pd

"""Script for learning handwriting with 3 layers, only use one side, as it speeds up learning"""

class Neurons:
    def __init__(self, ninputs, noutputs, nalphamneurons, layertype):
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

        self.exp_pre_1 = np.exp(-dt/self.tau_pre_1)
        self.exp_pre_2 = np.exp(-dt/self.tau_pre_2)
        self.exp_syn = np.exp(-dt/self.tau_syn)
        self.exp_mem = np.exp(-dt/self.tau_mem)
        self.exp_elig_rise = np.exp(-dt/self.tau_elig_rise)
        self.exp_elig_decay = np.exp(-dt/self.tau_elig_decay)

        self.one_minus_exp_pre_2 = 1 - self.exp_pre_2
        self.one_minus_exp_mem = 1 - self.exp_mem
        self.one_minus_exp_elig_rise = 1 - self.exp_elig_rise
        self.one_minus_exp_elig_decay = 1 - self.exp_elig_decay

        self.axonaldelay = 0
        self.spike = 0
        self.ninputs = ninputs
        self.noutputs = noutputs

        self.actionpotential = 0
        self.Isynhist = []
        # self.vhist = []
        # self.tracehist = []
        self.vderivhist = []
        # self.lambdahist = []
        # self.lambdahist2 = []

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

        self.layertype = layertype

        # Hidden layer needs B matrix
        if self.layertype == "hidden":
            self.B = np.random.uniform(-1, 1, [nalphamneurons, self.noutputs] )

        # Computations for speedup


    def update_state(self, S, dt):
        # Input current streams in
        self.pre_trace1 = self.pre_trace1*self.exp_pre_1 + S
        self.pre_trace2 = self.pre_trace2*self.exp_pre_2 + self.one_minus_exp_pre_2*self.pre_trace1
        # self.tracehist.append(self.pre_trace2.copy())
        # self.Isynhist.append(self.Isyn.copy())
        # Update membrane potential
        # Membrane voltage increase
        self.Isyns = self.Isyns*self.exp_syn + np.dot(self.weights.T, S)
        self.Vs = self.Vs*self.exp_mem + self.Isyns*self.one_minus_exp_mem
        # self.vhist.append(self.Vs.copy())

        # Surrogate gradient
        self.surrogategradient = 1/((1+np.abs(self.beta*(self.Vs - self.threshold)))**2)
        # self.vderivhist.append(self.surrogategradient)

        # Outer product of presynaptic trace with postsynaptic partial derivative
        outer_hebbian = np.outer(self.pre_trace2, self.surrogategradient)
        outer_hebbian[outer_hebbian < 1e-7] = 0
        self.lambda_1 = self.lambda_1*self.exp_elig_rise + self.one_minus_exp_elig_rise*outer_hebbian.T
        self.lambda_2 = self.lambda_2*self.exp_elig_decay + self.one_minus_exp_elig_decay*self.lambda_1
        # self.lambdahist.append(self.lambda_2)
        # self.lambdahist2.append(self.lambda_2)

        # Voltage has crossed threshold. Now wait before transmitting and initialize refactory period
        spikes = self.Vs > self.threshold
        self.Vs[self.Vs > self.threshold] = 0

        return spikes

    def update_weight(self, error, lr, dt):
        if self.layertype == "hidden":
            delta = np.dot(error, self.B)
            dw = np.multiply(self.lambda_2.T, delta)

            self.chunktime += dt
            if self.chunktime >= self.finalchunktime:
                self.chunktime = 0

                self.weights += lr *  self.dw_running 
                self.weights = np.clip(self.weights, -1, 1)
                self.dw_running = np.zeros([self.ninputs, self.noutputs])
            self.dw_running += dw
        elif self.layertype == "output":

            dw = np.multiply(self.lambda_2.T , error)

            self.chunktime += dt
            if self.chunktime >= self.finalchunktime:
                self.chunktime = 0

                self.weights += lr *  self.dw_running
                self.weights = np.clip(self.weights, -1, 1)
                self.dw_running = np.zeros([self.ninputs, self.noutputs])
            self.dw_running += dw

        else: 
            print(f"layertype:{self.layertype} not recognized")
            assert()
            

    def reset(self):
        self.Vs = np.ones(self.noutputs)*self.urest
        self.Isyn = 0
        self.actionpotential = 0
        self.dwhist = []
        self.Isynhist = []
        # self.vhist = []
        # self.tracehist = []
        self.vderivhist = []
        # self.lambdahist = []
        # self.lambdahist2 = []
        self.previousv = 0
        self.vderiv = 0
        self.surrogategradient = 0
        self.lambda_1 = np.zeros(self.ninputs)
        self.lambda_2 = np.zeros(self.ninputs)

        self.pre_trace1 = np.zeros(self.ninputs, dtype=np.float32)
        self.pre_trace2 = np.zeros(self.ninputs, dtype=np.float32)


# Command line parsing 

opts, args = getopt.getopt(sys.argv[1:], "-l:-o:-m:-a:-t:-r:")
for o, a in opts:
    if o == "-l":
        lr_hidden = float(a)
    if o == "-o":
        lr_out = float(a)
    if o == "-m":
        amplitude_minimum = float(a)
    if o == "-a":
        amplitude_maximum = float(a)
    if o == "-t":
        totaldigits = int(a)
    if o == "-r":
        trial = int(a)

# Load audio data
melfolderpath = "/scratch/p309238/handwriting/data/audio/"
#melfolderpath = "../../audio/audio_cut/mel_spectrograms"
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
textfolderpath = "/scratch/p309238/handwriting/data/text/"
#textfolderpath = "../../text/text_normalized"
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



        textdict[textlabel] = textdata
# Pick number 4 to start with
target = textdict["4"][:,0]
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
nhiddenneurons = 250
ntrials = 100000
timesteps = len(inputs[1])
dt  = 1
trialtime = timesteps/dt

# 2 neural nets
amplitudes = np.random.uniform(amplitude_minimum , amplitude_maximum, nalphamneurons)
amplitudes[:int(nalphamneurons/2)] *= -1

l1 = Neurons(nsensorneurons, nhiddenneurons, nalphamneurons, layertype="hidden")
l2 = Neurons(nhiddenneurons, nalphamneurons, nalphamneurons, layertype="output")

tau_rises = np.random.uniform(20, 100, nalphamneurons)
tau_decays = np.random.uniform(20, 100, nalphamneurons)

exp_tau_rises = np.exp(-dt/tau_rises)
exp_tau_decays = np.exp(-dt/tau_decays)

one_minus_exp_rises = 1 - exp_tau_rises
one_minus_exp_decays = 1- exp_tau_decays

losses = np.zeros(ntrials)
dt = 1

neuronthresholds = 20

digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
numbers = digits[:totaldigits]

if not os.path.isdir(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/"):
    assert()
rng = np.random.default_rng(seed=42)

for t in range(ntrials):

    number = rng.choice(numbers)
    target = textdict[number][:,1]
    # targety = textdict[number][:,1]
    inputs = audiodict[number]

    errorhist = []
    spikefilterhist = np.zeros([nalphamneurons, int(trialtime/dt)])
    totalerr = 0
    error1 = 0
    error2 = 0

    spikefilter1 = np.zeros(nalphamneurons)
    spikefilter2 = np.zeros(nalphamneurons)


    spikehist = np.zeros([nsensorneurons, 970])
    outspikehist = np.zeros([nalphamneurons, 970])

    incharges = np.zeros(nsensorneurons)

    for idx, inp in enumerate(inputs.T):
        incharges += inp
        spikes = incharges >= neuronthresholds
        incharges[incharges >= neuronthresholds] = 0 
        spikehist[:,idx] = spikes.copy()
        # Run NN
        spikesl1 = l1.update_state(spikes, dt)
        spikesl2 = l2.update_state(spikesl1, dt)
        outspikehist[:,idx] = spikesl2
        # Run muscles

        spikefilter1 = spikefilter1*exp_tau_rises + (one_minus_exp_rises) * spikesl2 * amplitudes
        spikefilter2 = spikefilter2*exp_tau_decays + (one_minus_exp_decays) * spikefilter1

        spikefilterhist[:,idx] = spikefilter2.copy()
        #Update without pen up
        if target[idx] != 0:
            errorx = target[idx] - np.sum(spikefilter2.copy())
        else:
            errorx = 0
        l1.update_weight(errorx * amplitudes/amplitude_maximum, lr_hidden, dt)
        l2.update_weight(errorx * amplitudes/amplitude_maximum, lr_out, dt)
        errorhist.append(errorx)

    losses[t] = np.array(1/2*np.array(errorhist)**2).mean()
    l1.reset()
    l2.reset()

    if t > ntrials - 20:
        np.savetxt(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/spikefilterhisty_{t}.txt", spikefilterhist, fmt="%.5g")
        np.savetxt(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/targety_{t}.txt", target)

np.savetxt(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/lossy.txt", losses)
np.savetxt(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/amplitudesy.txt", amplitudes)
np.savetxt(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/tau_risesy.txt", tau_rises)
np.savetxt(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/tau_decaysy.txt", tau_decays)
np.savetxt(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/l1weightsy.txt", l1.weights)
np.savetxt(f"/scratch/p309238/handwriting/topthree/digits={totaldigits}/lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}_trial={trial}/l2weightsy.txt", l2.weights)

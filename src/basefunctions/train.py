import sys
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.layers import OutputLayer, SpikingLayerConfig

'''
Train TwNN on sine/gauss/block

Parameters:
-----------
name : name of function to train on
       choose from "sine, gaussian, or square"

length : total length of the signal in datapoints

width : 1/(2*var)
'''


if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data/basefunctions/"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train base function fitting with SuperSpike + DFA")
    parser.add_argument("--funcname", type=str, default="sine")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--n_outputs", type=int, default=50)
    parser.add_argument("--n_inputs", type=int, default=100)
    parser.add_argument("--twitch_amp_max", type=float, default=0.1)
    parser.add_argument("--twitch_amp_min", type=float, default=0.05)
    parser.add_argument("--tau_min", type=int, default=20)
    parser.add_argument("--tau_max", type=int, default=50)
    parser.add_argument("--ntrials", type=int, default=1000)
    parser.add_argument("--updatefrequency", type=int, default=50)
    return parser.parse_args()

def load_basefunction(basefunctionpath):
    """Load the earlier generated base function from txt"""
    data = np.loadtxt(basefunctionpath)
    x = data[:, 0]
    y = data[:, 1]

    return x, y

def findfullfilename(INPUT_DIR, basefunctionname):
    matching_files = [
        file for file in INPUT_DIR.iterdir()
        if file.is_file() and file.name.startswith(basefunctionname)
    ]

    if not matching_files:
        raise FileNotFoundError(f"No files starting with '{basefunctionname}' were found in the directory.")
    elif len(matching_files) == 1:
        print(f" File found: {str(matching_files[0]).split("/")[-1]}")
        return matching_files[0]
    elif len(matching_files) > 1:
        raise ValueError(f"Expected at most 1 file, but found {len(matching_files)} matching files")


def createtraindatadir(ROOT_DIR):
    TRAINDATA_DIR = ROOT_DIR / "data/basefunctions/traindata"
    TRAINDATA_DIR.mkdir(parents=True, exist_ok=True)

def run_trial(
        nalphamneurons, 
        l1, 
        inputs, 
        amplitudes, 
        triallength, 
        target, 
        lr,
        tau_rises,
        tau_decays
    ):

    errorhist = []

    spikefilter1 = np.zeros(nalphamneurons)
    spikefilter2 = np.zeros(nalphamneurons)
    spikefilterhist = np.zeros([nalphamneurons, int(triallength)])

    for idx, inp in enumerate(inputs.T):
        spikes = l1.update_state(inp, 1)


        spikefilter1 = spikefilter1*np.exp(-1/tau_rises) + (1-np.exp(-1/tau_rises) ) * spikes * amplitudes
        spikefilter2 = spikefilter2*np.exp(-1/tau_decays) + (1-np.exp(-1/tau_decays) ) * spikefilter1
        
        l1.update_weight((target[idx] - np.sum(spikefilter2))* amplitudes/0.1, lr, 1)
        errorhist.append(target[idx] - np.sum(spikefilter2))
        spikefilterhist[:,idx] = spikefilter2.copy()

    l1.reset()
    return np.array(1/2*np.array(errorhist)**2).mean(), spikefilterhist

def printtrainingstatus(iteration, mse):
    print(f"Iteration:{iteration}\t mse:{mse}")

def main():

    args = parse_args()

    OUTPUT_DIR = DATA_DIR / "trainresults"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_DIR = ROOT_DIR / "data/basefunctions/inputfiles" 

    basefunctionname = args.funcname
    ntrials = args.ntrials
    n_inputs = args.n_inputs
    n_outputs = args.n_outputs
    threshold = args.threshold
    amplitude_minimum = args.twitch_amp_min
    amplitude_maximum = args.twitch_amp_max
    lr = args.lr
    tau_min = args.tau_min
    tau_max = args.tau_max
    updatefrequency=args.updatefrequency

    basefunctionfilename = findfullfilename(INPUT_DIR, basefunctionname)
    _, basefunction = load_basefunction(basefunctionfilename)
    triallength = len(basefunction)

    cfg = SpikingLayerConfig
    cfg.threshold = threshold
    cfg.taumem = 30
    cfg.trest = 0
    cfg.tausyn = 15
    cfg.tauref = 50
    cfg.tau_rise = 20
    cfg.tau_decay = 15

    l1 = OutputLayer(n_inputs, n_outputs, config=cfg)
  

    amplitudes = np.random.uniform(amplitude_minimum, amplitude_maximum, n_outputs)
    amplitudes[: int(n_outputs / 2)] *= -1

    tau_rises = np.random.uniform(tau_min, tau_max, n_outputs)
    tau_decays = np.random.uniform(tau_min, tau_max, n_outputs)

    ntrials = args.ntrials

    losses = np.zeros(ntrials)

    inputs = np.random.choice([0, 1],[n_inputs, triallength], p=[0.99, 0.01])

    for iteration in range(ntrials):

        mse, twitchhistory = run_trial(n_outputs,
                         l1, 
                         inputs, 
                         amplitudes, 
                         triallength,
                         basefunction,
                         lr,
                         tau_rises,
                         tau_decays)
        
        losses[iteration] = mse
        if iteration % updatefrequency == 0:
            printtrainingstatus(iteration, mse)

    np.save(os.path.join(OUTPUT_DIR, "losses.npy"), losses)
    plt.plot(np.sum(twitchhistory, axis=0), label="TwitchOutput")
    plt.plot(basefunction, label="Basefunction")
    plt.show()

if __name__ == "__main__":
    main()

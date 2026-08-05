import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sys



ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SPECTROGRAM_DIR = ROOT_DIR / "data/handwriting/audio_spectrogram/"
TRAJECTORY_DIR = ROOT_DIR / "data/handwriting/digits_normalized/"
OUTPUT_DIR = ROOT_DIR / "data/handwriting/trainoutput/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.layers import HiddenLayer, OutputLayer
from src.shared.layers import OutputLayer, SpikingLayerConfig


def parse_args():

    parser = argparse.ArgumentParser(description="Train audio-to-handwriting SNN with SuperSpike + DFA")
    parser.add_argument("--n_hidden", type=int, default=250)
    parser.add_argument("--n_outputs", type=int, default=200)
    parser.add_argument("--lr_hidden", type=float, default=0.001)
    parser.add_argument("--lr_out", type=float, default=0.0001)
    parser.add_argument("--threshold_in", type=int, default=20)
    parser.add_argument("--twitch_amp_min", type=float, default=1)
    parser.add_argument("--twitch_amp_max", type=float, default=10)
    parser.add_argument("--tau_min", type=int, default=20)
    parser.add_argument("--tau_max", type=int, default=100)
    parser.add_argument("--ntrials", type=int, default=10000)
    parser.add_argument("--updatefrequency", type=int, default=1)
    parser.add_argument("--totaldigits", type=int, default=10)
    return parser.parse_args()


def load_audio_data(melfolderpath):
    """Load mel-spectrogram audio data, normalized. Returns {digit_label: (256, 970) array}."""
    audiodict = {}
    for file in os.listdir(melfolderpath):
        if file.endswith(".txt"):
            audiolabel = file.split("_")[0]
            audiodata = np.loadtxt(os.path.join(melfolderpath, file))
            audiodict[audiolabel] = audiodata / -100  # normalize
    return audiodict


def load_handwriting_data(textfolderpath):
    """Load handwriting pen-trajectory data, padded + normalized. Returns {digit_label: (970, 2) array}."""
    textdict = {}
    for file in os.listdir(textfolderpath):
        if file.endswith(".csv"):
            textlabel = file.split(".")[0].strip("digit").strip("_normalized")
            textdata = pd.read_csv(os.path.join(textfolderpath, file)).to_numpy()
            # Pad 85 on each side to make all sequences equally long (970 total), normalize to [-1, 1]
            textdata = np.concatenate(
                (np.zeros([85, 2]), textdata, np.zeros([85, 2])), axis=0
            ) / 600
            textdict[textlabel] = textdata
    return textdict


def make_output_dir(base_path, totaldigits, lr_hidden, lr_out, amplitude_minimum, amplitude_maximum):
    outdir = os.path.join(
        base_path,
        f"digits={totaldigits}",
        f"lrh={lr_hidden}_lro={lr_out}_ami={amplitude_minimum}_ama={amplitude_maximum}",
    )
    os.makedirs(outdir, exist_ok=True)
    return outdir


def run_trial(number, textdict, audiodict, l1, l2, amplitudes, amplitude_maximum,
              exp_tau_rises, exp_tau_decays, one_minus_exp_rises, one_minus_exp_decays,
              neuronthresholds, lr_hidden, lr_out, dt, nsensorneurons, nalphamneurons, triallength):
    target = textdict[number][:, 0]
    inputs = audiodict[number]

    errorhist = []
    spikefilter1 = np.zeros(nalphamneurons)
    spikefilter2 = np.zeros(nalphamneurons)
    incharges = np.zeros(nsensorneurons)
    spikefilterhist = np.zeros([nalphamneurons, int(triallength)])

    for idx, inp in enumerate(inputs.T):
        incharges += inp
        spikes = incharges >= neuronthresholds
        incharges[spikes] = 0

        # Run network
        spikesl1 = l1.update_state(spikes, dt)
        spikesl2 = l2.update_state(spikesl1, dt)

        # Run "muscle" output filter
        spikefilter1 = spikefilter1 * exp_tau_rises + one_minus_exp_rises * spikesl2 * amplitudes
        spikefilter2 = spikefilter2 * exp_tau_decays + one_minus_exp_decays * spikefilter1

        # Update, skipping timesteps where the pen is up (target == 0)
        if target[idx] != 0:
            errorx = target[idx] - np.sum(spikefilter2)
        else:
            errorx = 0

        scaled_error = errorx * amplitudes / amplitude_maximum
        l1.update_weight(scaled_error, lr_hidden, dt)
        l2.update_weight(scaled_error, lr_out, dt)
        errorhist.append(errorx)
        spikefilterhist[:,idx] = spikefilter2.copy()

    l1.reset()
    l2.reset()
    return np.mean(0.5 * np.array(errorhist) ** 2), spikefilterhist


def main():
    args = parse_args()

    ntrials = args.ntrials
    n_hidden = args.n_hidden
    n_outputs = args.n_outputs
    threshold = args.threshold_in
    twitch_amp_min = args.twitch_amp_min
    twitch_amp_max = args.twitch_amp_max
    lr_hidden = args.lr_hidden
    lr_out = args.lr_out
    tau_min = args.tau_min
    tau_max = args.tau_max
    updatefrequency=args.updatefrequency

    # Load audio and handwriting data
    audiodict = load_audio_data(SPECTROGRAM_DIR)
    textdict = load_handwriting_data(TRAJECTORY_DIR)

    # Check number of inputs 
    n_inputs = audiodict["0"].shape[0]
    triallength = audiodict["0"].shape[1]


    # Set config for shared spiking layers
    cfg = SpikingLayerConfig

    # Create neural network
    l1 = HiddenLayer(n_inputs, n_hidden, n_outputs, config=cfg)
    l2 = OutputLayer(n_hidden, n_outputs, config=cfg)

    amplitudes = np.random.uniform(twitch_amp_min, twitch_amp_max, n_outputs)
    amplitudes[: int(n_outputs / 2)] *= -1

    tau_rises = np.random.uniform(tau_min, tau_max, n_outputs)
    tau_decays = np.random.uniform(tau_min, tau_max, n_outputs)

    exp_tau_rises = np.exp(-1/tau_rises)
    exp_tau_decays = np.exp(-1/tau_decays)

    one_minus_exp_rises = 1 - exp_tau_rises
    one_minus_exp_decays = 1- exp_tau_decays

    digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    numbers = digits[: args.totaldigits]

    outdir = make_output_dir(
        OUTPUT_DIR, args.totaldigits, args.lr_hidden, args.lr_out,
        args.twitch_amp_min, args.twitch_amp_max,
    )

    losses = np.zeros(ntrials)

    for iteration in range(ntrials):

        if iteration % 100 == 0:
            print(f"Iteration: {iteration}")
        # Pick a random input
        number = np.random.choice(numbers)

        mse, twitchhistory = run_trial(number,
                                       textdict,
                                       audiodict,
                                       l1,
                                       l2,
                                       amplitudes,
                                       twitch_amp_max,
                                       exp_tau_decays,
                                       exp_tau_rises,
                                       one_minus_exp_rises,
                                       one_minus_exp_decays,
                                       threshold,
                                       lr_hidden,
                                       lr_out,
                                       1,
                                       n_inputs,
                                       n_outputs,
                                       triallength,
        )
        
        losses[iteration] = mse

        if iteration > ntrials-20:
            np.savetxt(os.path.join(outdir, f"twitches{iteration}.txt"), twitchhistory)
            np.savetxt(os.path.join(outdir, f"target{iteration}.txt"), textdict[number])
    np.savetxt(os.path.join(outdir, "losses.txt"), losses)

if __name__ == "__main__":
    main()

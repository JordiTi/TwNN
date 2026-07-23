import argparse
import os

import numpy as np
import pandas as pd

from shared.layers import HiddenLayer, OutputLayer

#TODO: need to set defaults
def parse_args():
    parser = argparse.ArgumentParser(description="Train audio-to-handwriting SNN with SuperSpike + DFA")
    parser.add_argument("--lr_hidden", type=float, default="0")
    parser.add_argument("--lr_out", type=float, default="0")
    parser.add_argument("--threshold", type=int, default="0")
    parser.add_argument("--amplitude_minimum", type=float, default=1)
    parser.add_argument("--amplitude_maximum", type=float, default=10)
    parser.add_argument("--tau_min", type=int, default=20)
    parser.add_argument("--tau_max", type=int, default=100)
    parser.add_argument("--ntrials", type=int, default=10000)
    parser.add_argument("--updatefrequency", type=int, default=1)
    parser.add_argument("--total-digits", type=int)
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
            textlabel = file.split(".")[0]
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
              neuronthresholds, lr_hidden, lr_out, dt, nsensorneurons, nalphamneurons):
    target = textdict[number][:, 0]
    inputs = audiodict[number]

    errorhist = []
    spikefilter1 = np.zeros(nalphamneurons)
    spikefilter2 = np.zeros(nalphamneurons)
    incharges = np.zeros(nsensorneurons)

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

    l1.reset()
    l2.reset()
    return np.mean(0.5 * np.array(errorhist) ** 2)


def main():
    args = parse_args()

    melfolderpath = "/scratch/p309238/handwriting/data/audio/"
    textfolderpath = "/scratch/p309238/handwriting/data/text/"
    output_base_path = "/scratch/p309238/handwriting/threelayers/"

    audiodict = load_audio_data(melfolderpath)
    textdict = load_handwriting_data(textfolderpath)

    nsensorneurons = len(audiodict["4"])
    nalphamneurons = 200
    nhiddenneurons = 250
    ntrials = 50000
    timesteps = len(audiodict["4"][1])
    dt = 1
    trialtime = timesteps / dt
    neuronthresholds = 20

    amplitudes = np.random.uniform(args.amplitude_minimum, args.amplitude_maximum, nalphamneurons)
    amplitudes[: int(nalphamneurons / 2)] *= -1

    l1 = HiddenLayer(nsensorneurons, nhiddenneurons, nalphamneurons)
    l2 = OutputLayer(nhiddenneurons, nalphamneurons)

    tau_rises = np.random.uniform(20, 100, nalphamneurons)
    tau_decays = np.random.uniform(20, 100, nalphamneurons)
    exp_tau_rises = np.exp(-dt / tau_rises)
    exp_tau_decays = np.exp(-dt / tau_decays)
    one_minus_exp_rises = 1 - exp_tau_rises
    one_minus_exp_decays = 1 - exp_tau_decays

    digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    numbers = digits[: args.totaldigits]

    outdir = make_output_dir(
        output_base_path, args.totaldigits, args.lr_hidden, args.lr_out,
        args.amplitude_minimum, args.amplitude_maximum,
    )

    losses = np.zeros(ntrials)
    for t in range(ntrials):
        number = np.random.choice(numbers)
        losses[t] = run_trial(
            number, textdict, audiodict, l1, l2, amplitudes, args.amplitude_maximum,
            exp_tau_rises, exp_tau_decays, one_minus_exp_rises, one_minus_exp_decays,
            neuronthresholds, args.lr_hidden, args.lr_out, dt, nsensorneurons, nalphamneurons,
        )

    np.save(os.path.join(outdir, "losses.npy"), losses)


if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "."
DIGIT_FOLDER = "digits=10"

digit_path = os.path.join(ROOT, DIGIT_FOLDER)

for settings in os.listdir(digit_path):

    settings_path = os.path.join(digit_path, settings)

    if not os.path.isdir(settings_path):
        continue

    print(f"Processing folder: {settings_path}")

    spike_files = [f for f in os.listdir(settings_path) if f.startswith("spikefilterhist_")]

    for spike_file in spike_files:

        idx = spike_file.replace("spikefilterhist_", "").replace(".txt", "")
        target_file = f"target_{idx}.txt"

        spike_path = os.path.join(settings_path, spike_file)
        target_path = os.path.join(settings_path, target_file)

        if not os.path.exists(target_path):
            continue

        spike = np.loadtxt(spike_path)
        target = np.loadtxt(target_path)

        summed = np.sum(spike, axis=0)

        plt.figure(figsize=(8,5))

        plt.plot(summed, label="sum(spikefilterhist)", linestyle="--")
        plt.plot(target, label="target")

        plt.title(f"{settings} — index {idx}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(settings_path, f"spike_vs_target_{idx}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

print("Finished plotting all spikefilterhist vs target pairs.")
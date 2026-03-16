import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "."


def avg_last_1000(loss):
    if len(loss) >= 1000:
        return np.mean(loss[-1000:])
    return np.mean(loss)


best_settings = {}

# -----------------------------------
# 1. Find best settings per digit
# -----------------------------------
for digit_folder in os.listdir(ROOT):

    if not digit_folder.startswith("digits"):
        continue

    digit_path = os.path.join(ROOT, digit_folder)

    best_loss = np.inf
    best_folder = None

    for settings in os.listdir(digit_path):

        settings_path = os.path.join(digit_path, settings)

        if not os.path.isdir(settings_path):
            continue

        loss_file = os.path.join(settings_path, "loss.txt")

        if not os.path.exists(loss_file):
            continue

        loss = np.loadtxt(loss_file)
        avg_loss = avg_last_1000(loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_folder = settings_path

    if best_folder:
        best_settings[digit_folder] = best_folder


# -----------------------------------
# 2. Plot spikefilterhist vs target
# -----------------------------------
for digit, folder in best_settings.items():

    print(f"Processing {digit} → {folder}")

    # find all spikefilterhist files
    spike_files = [f for f in os.listdir(folder) if f.startswith("spikefilterhist_")]

    plt.figure(figsize=(10,6))

    for spike_file in spike_files:

        idx = spike_file.replace("spikefilterhist_", "").replace(".txt", "")
        target_file = f"target_{idx}.txt"

        spike_path = os.path.join(folder, spike_file)
        target_path = os.path.join(folder, target_file)

        if not os.path.exists(target_path):
            continue

        spike = np.loadtxt(spike_path)
        target = np.loadtxt(target_path)

        summed = np.sum(spike, axis=0)

        plt.plot(summed, linestyle="--", label=f"spike {idx}")
        plt.plot(target, label=f"target {idx}")

    plt.title(f"{digit} — spikefilterhist sum vs target")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend(fontsize=7)
    plt.tight_layout()

    plt.savefig(f"{digit}_spike_vs_target.png", dpi=300)
    plt.close()

print("Plots saved.")
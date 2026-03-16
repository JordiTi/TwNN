import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "."
DIGIT_FOLDER = "data"

digit_path = os.path.join(ROOT, DIGIT_FOLDER)

for settings in os.listdir(digit_path):

    settings_path = os.path.join(digit_path, settings)

    if not os.path.isdir(settings_path):
        continue

    print(f"Processing folder: {settings_path}")

    spike_files = [f for f in os.listdir(settings_path) if f.startswith("spikefilterhist")]

    for spike_file in spike_files:
        if spike_file.startswith("spikefilterhistx_"):
            idx = spike_file.replace("spikefilterhistx_", "").replace(".txt", "")
            target_file = f"targetx_{idx}.txt"
        elif spike_file.startswith("spikefilterhisty_"):
            continue
        spike_path = os.path.join(settings_path, spike_file)
        target_path = os.path.join(settings_path, target_file)

        if not os.path.exists(target_path):
            continue

        spike = np.loadtxt(spike_path)
        target = np.loadtxt(target_path)

        summed = np.sum(spike, axis=0)

        plt.figure(figsize=(8,5))
        trialnumber = spike_file.split("_")[1].split(".")[0]
        xoutput = np.loadtxt(os.path.join(settings_path, f"spikefilterhistx_{trialnumber}.txt"))
        youtput = np.loadtxt(os.path.join(settings_path, f"spikefilterhisty_{trialnumber}.txt"))
        xtarget = np.loadtxt(os.path.join(settings_path, f"targetx_{trialnumber}.txt"))
        ytarget = np.loadtxt(os.path.join(settings_path, f"targety_{trialnumber}.txt"))

        xsum = np.sum(xoutput, axis=0)
        ysum = np.sum(youtput, axis=0)

        xsum[xsum==0] = np.nan
        ysum[xsum==0] = np.nan

        xtarget[xtarget==0] = np.nan
        ytarget[ytarget==0] = np.nan
    
        plt.plot(xsum, ysum)
        plt.plot(xtarget,ytarget)
        plt.gca().invert_yaxis()
        plt.savefig(f"{settings_path}/drawing_{trialnumber}.png", dpi=300)
        plt.close()




print("Finished plotting all spikefilterhist vs target pairs.")
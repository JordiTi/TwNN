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

    spike_files = [f for f in os.listdir(settings_path) if not f.startswith("target") and not f.endswith(".png")]

    for spike_file in spike_files:
        if spike_file.endswith("_x.txt"):
            number = spike_file.replace("_x.txt", "")
            output_filex = f"{number}_x.txt"
            output_filey = f"{number}_x.txt"
            target_filex = f"target_{number}_x.txt"
            target_filey = f"target_{number}_y.txt"
        elif spike_file.startswith("spikefilterhisty_"):
            continue

        plt.figure(figsize=(8,5))
        trialnumber = spike_file.split("_")[1].split(".")[0]
        xoutput = np.loadtxt(os.path.join(settings_path, f"{number}_x.txt"))
        youtput = np.loadtxt(os.path.join(settings_path, f"{number}_y.txt"))
        xtarget = np.loadtxt(os.path.join(settings_path, f"target_{number}_x.txt"))
        ytarget = np.loadtxt(os.path.join(settings_path, f"target_{number}_y.txt"))

        # PENUP PENDOWN
        xoutput[:85] = 0
        youtput[:85] = 0
        xoutput[-85:] = 0
        youtput[-85:] = 0
  
        xoutput[xoutput==0] = np.nan
        youtput[youtput==0] = np.nan

        xtarget[xtarget==0] = np.nan
        ytarget[ytarget==0] = np.nan

        minval = np.nanmin([np.nanmin(xtarget), np.nanmin(ytarget)])
        maxval = np.nanmax([np.nanmax(xtarget), np.nanmax(ytarget)])

        # Center values
        xtargetmean = np.nanmean(xtarget)
        ytargetmean = np.nanmean(ytarget)
  
        xoutput = xoutput - xtargetmean
        youtput = youtput - ytargetmean
        xtarget = xtarget - xtargetmean
        ytarget = ytarget - ytargetmean
    
        plt.plot(xoutput, youtput, label="output")
        plt.plot(xtarget,ytarget, label="target")

        plt.legend()
        plt.xlim(-0.8, 0.8)
        plt.ylim(-0.8, 0.8)
        plt.gca().invert_yaxis()
        plt.savefig(f"./data/{settings}/{number}.png", dpi=300)
        # plt.show()
        plt.close()




print("Finished plotting all spikefilterhist vs target pairs.")
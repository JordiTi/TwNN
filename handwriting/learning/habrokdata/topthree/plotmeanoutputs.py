import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

'''
Plots the mean output curves and the individual curves in alpha=0.3
'''
def moving_average(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')


ROOT = "."
OUTPUT_FILE = "average_last1000_loss.txt"

results = []

for digit_folder in sorted(os.listdir(ROOT)):

    if not digit_folder.startswith("data"):
        continue

    digit_path = os.path.join(ROOT, digit_folder)

    if not os.path.isdir(digit_path):
        continue

    digit = "10"


    # collect curves
    # unique settings
    allsettings = os.listdir(digit_path)
    uniquesettings = []
    for setting in allsettings:
        if setting.startswith("m"):
            continue
        uniquesettings.append(setting.rstrip("_trial12345="))
    uniquesettings = list(set(uniquesettings))
    digits = [0, 1, 2, 3 ,4 ,5 ,6 ,7 ,8, 9]
    for digit in digits:
        for setting in uniquesettings:
            xtarget = np.loadtxt(digit_path + "/" + setting + "_trial=1/" + f"target_{digit}_x.txt")
            ytarget = np.loadtxt(digit_path + "/" + setting + "_trial=1/" + f"target_{digit}_y.txt")
            xtarget[xtarget==0] = np.nan
            ytarget[ytarget==0] = np.nan
            xtargetmean = np.nanmean(xtarget)
            ytargetmean = np.nanmean(ytarget)
            xtarget = xtarget - xtargetmean
            ytarget = ytarget - ytargetmean

            xoutputs = []
            youtputs = []
            for folder in os.listdir(digit_path):
                if folder.startswith(setting):
                    xoutput = np.loadtxt(digit_path + "/" + folder + "/" + f"{digit}_x.txt")
                    youtput = np.loadtxt(digit_path + "/" + folder + "/" + f"{digit}_y.txt")

                    # PENUP PENDOWN
                    xoutput[:85] = 0
                    youtput[:85] = 0
                    xoutput[-85:] = 0
                    youtput[-85:] = 0
            
                    xoutput[xoutput==0] = np.nan
                    youtput[youtput==0] = np.nan



                    # Center values


                    xoutput = xoutput - xtargetmean
                    youtput = youtput - ytargetmean
                    xoutputs.append(xoutput)
                    youtputs.append(youtput)
            
            averagex = np.mean(np.array(xoutputs), axis=0)
            averagey = np.mean(np.array(youtputs), axis=0)

            minval = np.nanmin([np.nanmin(xtarget), np.nanmin(ytarget)])
            maxval = np.nanmax([np.nanmax(xtarget), np.nanmax(ytarget)])

            for i in range(len(xoutputs)):
                plt.plot(xoutputs[i], youtputs[i], alpha=0.3)
            plt.plot(averagex, averagey, linewidth=2, label="mean output")
            plt.plot(xtarget, ytarget, linewidth=2, linestyle=(0, (5,1)), label="target")
            plt.xlim(-0.8, 0.8)
            plt.ylim(-0.8, 0.8)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.savefig(f"./data/meanplots/{setting}_{digit}.pdf", dpi=300, format="pdf")
            plt.clf()
            # plt.show()
             
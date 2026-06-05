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

    for setting in uniquesettings:
        allerrors = []
        fig, axes = plt.subplots(2, 5, figsize=(4.8, 2.4))
        fig.subplots_adjust(wspace=0, hspace=0)
        for digit in digits:
            row = int(digit > 4)
            column = digit%5

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

                    allerrors.append(np.nansum(np.abs(xoutput -xtarget))+ np.nansum(np.abs(youtput-ytarget)))
            
            averagex = np.mean(np.array(xoutputs), axis=0)
            averagey = np.mean(np.array(youtputs), axis=0)

            minval = np.nanmin([np.nanmin(xtarget), np.nanmin(ytarget)])
            maxval = np.nanmax([np.nanmax(xtarget), np.nanmax(ytarget)])

            for i in range(len(xoutputs)):
                
                axes[row,column].plot(xoutputs[i], youtputs[i], alpha=0.3)
            axes[row,column].plot(averagex, averagey, linewidth=1, label="mean output")
            axes[row,column].plot(xtarget, ytarget, linewidth=1, linestyle=(0, (5,1)), label="target")
            axes[row,column].set_xlim(-0.42, 0.5)
            axes[row,column].set_ylim(-0.6, 0.6)
            axes[row,column].invert_yaxis()
            if row == 0 and column == 4:
                axes[row,column].legend(fontsize=8 ,loc='upper left', bbox_to_anchor=(1, 1.06))
            axes[row,column].set_xticks([])
            axes[row,column].set_yticks([])
            axes[row,column].tick_params(left=False, bottom=False)
            # for spine in axes[row,column].spines.values():
            #     spine.set_visible(False)
        print(setting)
        print(np.sum(np.array(allerrors))/len(uniquesettings)/len(xoutputs))
        plt.savefig(f"./data/meanplots/{setting}_wospine.pdf", dpi=300, format="pdf", bbox_inches='tight')
        plt.close()
            # plt.show()
            
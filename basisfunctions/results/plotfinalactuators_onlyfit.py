import os
import numpy as np
import matplotlib.pyplot as plt

root = "./"

functionfolders = [f for f in os.listdir(root) if os.path.isdir(f)]
fig, axes = plt.subplots(1, 3, figsize=(4.8,3), sharey=True)
bestfits = {}
bestfits["gaussian"] = [0.0001, 0.01]
bestfits["sine"] = [0.001, 0.01]
bestfits["square"] = [0.001, 0.01]
for i, functionfolder in enumerate(sorted(functionfolders)):

    target = np.loadtxt(os.path.join(functionfolder, functionfolder + ".txt"))

    settingsfolders = [
        s for s in os.listdir(functionfolder)
        if os.path.isdir(os.path.join(functionfolder, s))
    ]

    final_losses = {}

    for settingsfolder in settingsfolders:

        if float(settingsfolder.split("_")[1].split("=")[1]) == bestfits[functionfolder][0] and \
           float(settingsfolder.split("_")[2].split("=")[1]) == bestfits[functionfolder][1] and \
           float(settingsfolder.split("_")[0].split("=")[1]) == 0.1:
            print(settingsfolder)
            setting_path = os.path.join(functionfolder, settingsfolder)

            trialfolders = [
                t for t in os.listdir(setting_path)
                if os.path.isdir(os.path.join(setting_path, t))
            ]

            fits = []
            losses = []

            for trialfolder in trialfolders:

                trial_path = os.path.join(setting_path, trialfolder)

                # ----- Fit plot -----
                output = np.loadtxt(os.path.join(trial_path, "actuatorhist49999.txt"))
                output_sum = np.sum(output, axis=0)

                # plt.plot(output_sum, label="output")
                # plt.plot(target, label="target")
                # plt.legend()
                # plt.savefig(os.path.join(trial_path, "fit.png"))
                # plt.clf()

                # # ----- Loss plot -----
                loss = np.loadtxt(os.path.join(trial_path, "losses.txt"))

                # plt.plot(loss)
                # plt.yscale("log")
                # plt.savefig(os.path.join(trial_path, "losses.png"))
                # plt.clf()

                fits.append(output_sum)
                losses.append(loss)

            fits = np.array(fits)

            # ----- Average + stderr fit -----
            mean_fit = np.mean(fits, axis=0)
            stderr_fit = np.std(fits, ddof=1, axis=0) 

            if functionfolder == "gaussian":
                x = np.arange(-500, 500, 1)
            else:
                x = np.arange(len(mean_fit))

            axes[i].plot(x, mean_fit, label="mean fit", linewidth=1)
            axes[i].plot(x, target, label="target", linewidth=1)
            axes[i].fill_between(
                x,
                mean_fit - stderr_fit,
                mean_fit + stderr_fit,
                alpha=0.3
            )
            axes[i].set_title(f"{functionfolder.capitalize()}", fontsize=10)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

plt.subplots_adjust(wspace=0, hspace=0)

axes[2].legend(fontsize=8 )

plt.savefig("fit_mean.pdf", dpi=300, format="pdf")



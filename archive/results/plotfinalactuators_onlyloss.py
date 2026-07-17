import os
import numpy as np
import matplotlib.pyplot as plt

root = "./"

functionfolders = [f for f in os.listdir(root) if os.path.isdir(f)]

fig, axes = plt.subplots(1, 3, figsize=(4.8,1.9), sharey=True, sharex=True)

colors = ["#D81B60", "#1E88E5", "#FFC107"]
for i, functionfolder in enumerate(sorted(functionfolders)):

    target = np.loadtxt(os.path.join(functionfolder, functionfolder + ".txt"))

    settingsfolders = [
        s for s in os.listdir(functionfolder)
        if os.path.isdir(os.path.join(functionfolder, s))
    ]

    final_losses = {}

    for settingsfolder in settingsfolders:

        setting_path = os.path.join(functionfolder, settingsfolder)

        trialfolders = [
            t for t in os.listdir(setting_path)
            if os.path.isdir(os.path.join(setting_path, t))
        ]

        losses = []

        for trialfolder in trialfolders:

            trial_path = os.path.join(setting_path, trialfolder)


            # # ----- Loss plot -----
            loss = np.loadtxt(os.path.join(trial_path, "losses.txt"))

            losses.append(loss)

        losses = np.array(losses)

        # ----- Average + stderr loss -----
        mean_loss = np.mean(losses, axis=0)

        stderr_loss = np.std(losses, ddof=1, axis=0) / np.sqrt(len(losses))

        x = np.arange(len(mean_loss))

        # ----- mean of last 1000 loss values -----
        final_loss = np.mean(losses[:, -1000:])
        final_losses[settingsfolder] = final_loss

    # ----- Parse amplitude + lr from setting names -----
    amps = []
    lrs = []
    values = []

    for setting, loss in final_losses.items():

        parts = setting.split("_")

        lr = float(parts[0].split("=")[1])
        minamp = float(parts[1].split("=")[1])
        maxamp = float(parts[2].split("=")[1])

        amp_combo = (minamp, maxamp)   # tuple → hashable

        lrs.append(lr)
        amps.append(amp_combo)
        values.append(loss)

    lrs = np.array(lrs)
    values = np.array(values)

    unique_amp_combos = sorted(set(amps))

    # ----- Line plot: amplitude vs loss, lr as legend -----
    axes[i].grid(True, which="major", axis="y")
    axes[i].grid(True, which="minor", axis="y", alpha=0.8)
    axes[i].grid(True, which="major", axis="x")
    axes[i].set_axisbelow(True)
    markers=["D", "X", "s", "o"]
    for idx, amp_combo in enumerate(unique_amp_combos):

        mask = np.array([a == amp_combo for a in amps])

        lr_vals = lrs[mask]
        loss_vals = values[mask]

        sorted_indices = np.argsort(lr_vals)
        lr_vals_sorted = lr_vals[sorted_indices]
        loss_vals_sorted = loss_vals[sorted_indices]

        # # jitter for strip look
        # jitter = (np.random.rand(len(lr_vals)) - 0.5) * 0.001 * lr_vals

        # axes[i].scatter(lr_vals + jitter, loss_vals,
        #             alpha=0.7,
        #             label=r"$A_{min}=$" + f"{amp_combo[0]}" + "\n" r"$A_{max}=$" + f"{amp_combo[1]}",
        #             marker=markers[idx], s=20)
        axes[i].plot(lr_vals_sorted, loss_vals_sorted,label=r"$A_{min}=$" + f"{amp_combo[0]}" + "\n" r"$A_{max}=$" + f"{amp_combo[1]}", alpha=0.7)

    
    axes[i].set_xscale("log")
    axes[i].set_yscale("log")
    axes[i].set_xlabel("Learning rate", fontsize=8)
    axes[i].set_title(f"{functionfolder.capitalize()}", fontsize=10)
    axes[i].tick_params(axis='both', which='major', labelsize=8)
    axes[i].tick_params(axis='both', which='minor', labelsize=8)

axes[0].set_ylabel("Final loss", fontsize=8)


plt.legend(fontsize=8, markerscale=0.75, handlelength=0.5, 
           loc="center left", bbox_to_anchor=(1.02, 0.54), borderpad=0.2)
plt.subplots_adjust(wspace=0.2, hspace=0, bottom=0.22, right=0.78)
plt.savefig("basis_errors_onlylines_alpha.pdf", dpi=300, format="pdf")



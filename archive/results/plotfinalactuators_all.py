import os
import numpy as np
import matplotlib.pyplot as plt

root = "./"

functionfolders = [f for f in os.listdir(root) if os.path.isdir(f)]

fig, axes = plt.subplots(1, 3, figsize=(15,10))

for functionfolder in functionfolders:

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
        losses = np.array(losses)

        # ----- Average + stderr fit -----
        mean_fit = np.mean(fits, axis=0)
        stderr_fit = np.std(fits, ddof=1, axis=0) / np.sqrt(len(fits))

        x = np.arange(len(mean_fit))

        # plt.plot(x, mean_fit, label="mean fit")
        # plt.plot(x, target, label="target")
        # plt.fill_between(
        #     x,
        #     mean_fit - stderr_fit,
        #     mean_fit + stderr_fit,
        #     alpha=0.3
        # )
        # plt.legend()
        # plt.savefig(os.path.join(setting_path, "fit_mean.png"))
        # plt.clf()

        # ----- Average + stderr loss -----
        mean_loss = np.mean(losses, axis=0)

        stderr_loss = np.std(losses, ddof=1, axis=0) / np.sqrt(len(losses))

        x = np.arange(len(mean_loss))
        # plt.plot(x, mean_loss)
        # plt.fill_between(
        #     x,
        #     mean_loss - stderr_loss,
        #     mean_loss + stderr_loss,
        #     alpha=0.3, color="red"
        # )
        # plt.yscale("log")
        # plt.savefig(os.path.join(setting_path, "loss_mean.png"))
        # plt.clf()

        # ----- mean of last 1000 loss values -----
        final_loss = np.mean(losses[:, -1000:])
        final_losses[settingsfolder] = final_loss

    # ----- Parse amplitude + lr from setting names -----
    # expected format example: amp0.5_lr0.001
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

    markers=["D", "X", "s", "o"]
    for idx, amp_combo in enumerate(unique_amp_combos):

        mask = np.array([a == amp_combo for a in amps])

        lr_vals = lrs[mask]
        loss_vals = values[mask]

        # jitter for strip look
        jitter = (np.random.rand(len(lr_vals)) - 0.5) * 0.001 * lr_vals

        plt.scatter(lr_vals + jitter, loss_vals,
                    alpha=0.7,
                    label=f"minamp={amp_combo[0]} maxamp={amp_combo[1]}",
                    marker=markers[idx])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Final loss")
    plt.legend()
    plt.savefig(os.path.join(functionfolder, "lr_vs_amp_strip.png"))
    plt.clf()


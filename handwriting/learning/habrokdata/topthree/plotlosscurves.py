import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

'''
Plots loss curves over time
'''
def moving_average(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')


ROOT = "."
OUTPUT_FILE = "average_last1000_loss.txt"

results = []

for digit_folder in sorted(os.listdir(ROOT)):

    if not digit_folder.startswith("digits=10"):
        continue

    digit_path = os.path.join(ROOT, digit_folder)

    if not os.path.isdir(digit_path):
        continue

    digit = digit_folder.replace("digits", "")

    plt.figure(figsize=(8,5))

    curves_by_setting = defaultdict(list)

    # collect curves
    for folder in sorted(os.listdir(digit_path)):
        print(folder)
        folder_path = os.path.join(digit_path, folder)

        if not os.path.isdir(folder_path):
            continue
        loss_filex = os.path.join(folder_path, "lossx.txt")
        loss_filey = os.path.join(folder_path, "lossx.txt")

        if not os.path.exists(loss_filex):
            continue

        lossx = np.loadtxt(loss_filex)
        lossy = np.loadtxt(loss_filey)
        lossx = moving_average(lossx, w=100)
        lossy = moving_average(lossy, w=100)
        loss = (lossx + lossy)/2

        # remove trial component
        setting = "_".join(
            part for part in folder.split("_") if not part.startswith("trial=")
        )

        curves_by_setting[setting].append(loss)

    # compute mean curve per setting
    colors=["green", "orange","blue"]
    cnt = 0
    for setting, curves in curves_by_setting.items():
        print(setting)
        # make all curves same length
        min_len = min(len(c) for c in curves)
        aligned = np.array([c[:min_len] for c in curves])
        x = np.arange(0, len(aligned[0]), 1)
        mean_curve = np.mean(aligned, axis=0)
        stderr_curve = np.std(aligned, axis=0, ddof=1)/np.sqrt(len(aligned))

        plt.fill_between(x, mean_curve, mean_curve - stderr_curve, alpha=0.3, color=colors[cnt])
        plt.fill_between(x, mean_curve, mean_curve + stderr_curve, alpha=0.3, color=colors[cnt])
        plt.plot(x, mean_curve, label=setting, color=colors[cnt])
        plt.yscale("log")

        last1000 = mean_curve[-1000:] if len(mean_curve) >= 1000 else mean_curve
        avg_loss = np.mean(last1000)
        print(avg_loss)
        results.append((digit, setting, avg_loss))
        cnt += 1

    plt.title(f"Average loss curves for digit {digit}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.grid(True, "both", "both")
    plt.savefig(f"loss_digit_{digit}.png", dpi=300)
    plt.close()


# save averages
with open(OUTPUT_FILE, "w") as f:
    for digit, setting, avg in results:
        f.write(f"digits{digit}/{setting} = {avg}\n")

print("Saved averages to:", OUTPUT_FILE)
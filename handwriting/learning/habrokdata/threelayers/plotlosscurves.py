import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')


ROOT = "."   # change if needed
OUTPUT_FILE = "average_last1000_loss.txt"

results = []

for digit_folder in sorted(os.listdir(ROOT)):
    if not digit_folder.startswith("digits"):
        continue

    digit_path = os.path.join(ROOT, digit_folder)

    if not os.path.isdir(digit_path):
        continue

    digit = digit_folder.replace("digits", "")

    plt.figure(figsize=(8,5))

    for settings in sorted(os.listdir(digit_path)):

        settings_path = os.path.join(digit_path, settings)

        if not os.path.isdir(settings_path):
            continue

        loss_file = os.path.join(settings_path, "loss.txt")

        if not os.path.exists(loss_file):
            continue

        loss = np.loadtxt(loss_file)
        loss = moving_average(loss,w=100)

        plt.plot(loss, label=settings)
        plt.yscale("log")

        last1000 = loss[-1000:] if len(loss) >= 1000 else loss
        avg_loss = np.mean(last1000)

        results.append((digit, settings, avg_loss))

    plt.title(f"Loss curves for digit {digit}")
    plt.xlabel("Trial")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"loss_digit_{digit}.png", dpi=300)
    plt.close()

# Save averages
with open(OUTPUT_FILE, "w") as f:
    for digit, settings, avg in results:
        f.write(f"digits{digit}/{settings} = {avg}\n")

print("Saved averages to:", OUTPUT_FILE)
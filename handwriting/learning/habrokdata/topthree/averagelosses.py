import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "average_last1000_loss.txt"

digits = []
losses = []

with open(INPUT_FILE) as f:
    for line in f:
        left, val = line.split(" = ")
        avg_loss = float(val.strip())

        digit = int(left.split("/")[0].replace("digits=", ""))

        digits.append(digit)
        losses.append(avg_loss)

digits = np.array(digits)
losses = np.array(losses)

plt.figure(figsize=(8,5))

unique_digits = sorted(set(digits))

for d in unique_digits:
    y = losses[digits == d]
    x = np.ones(len(y)) * (d-1)
    jitter = (np.random.rand(len(y)) - 0.5) * 0.2
    plt.scatter(x + jitter, y)

plt.yscale("log")
plt.xlabel("Number of digits")
plt.ylabel("Average loss (last 1000 trials)")
plt.title("Average loss per settings folder")
plt.xticks(range(len(unique_digits)), [str(d) for d in unique_digits])

plt.tight_layout()
plt.savefig("stripplot_avg_loss.png", dpi=300)
plt.show()
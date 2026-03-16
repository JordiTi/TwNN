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

plt.figure(figsize=(4.8, 2.5))

unique_digits = sorted(set(digits))

for d in unique_digits:
    y = losses[digits == d]
    x = np.ones(len(y)) * (d-1)
    jitter = (np.random.rand(len(y)) - 0.5) * 0.2
    plt.scatter(x + jitter, y, s=10)

plt.yscale("log")
plt.xlabel("Number of digits", fontsize=8)
plt.ylabel("Loss", fontsize=8)

plt.xticks(range(len(unique_digits)), [str(d) for d in unique_digits], fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, which="major", axis="y", alpha=0.35)
plt.grid(True, which="minor", axis="y", alpha=0.15)
plt.grid(True, which="major", axis="x", alpha=0.35)
plt.tight_layout()
plt.savefig("loss_per_digit.pdf", dpi=300, format="pdf")
plt.show()
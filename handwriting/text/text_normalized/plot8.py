import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load trajectory CSV
df = pd.read_csv("8.csv")  # replace with your file

# Load the image of the handwritten digit
img = Image.open("8.png")  # replace with the corresponding digit image

# Create figure and axes
fig, axes = plt.subplots(2, 2, figsize=(1.2, 2))

# --- Trajectory (x vs y) ---
axes[0,0].plot(df['x'], df['y'], 'b-', linewidth=2)
axes[0,0].set_aspect('equal')
axes[0,0].invert_yaxis()  # typical for image coordinates
axes[0,0].axis('off')      # remove all ticks & labels

# --- X stroke over time ---
axes[1,0].plot(df['x'], 'r-', linewidth=2)
axes[1,0].axis('off')

# --- Y stroke over time ---
axes[0,1].plot(df['y'], 'g-', linewidth=2)
axes[0,1].invert_yaxis()
axes[0,1].axis('off')

# --- Digit image ---
axes[1,1].imshow(img, cmap='gray')
axes[1,1].axis('off')

# Adjust spacing to remove whitespace
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout(pad=0)
axes[1,1].remove()
plt.savefig("handwritten_8.pdf", dpi=300, format="pdf")
plt.show()
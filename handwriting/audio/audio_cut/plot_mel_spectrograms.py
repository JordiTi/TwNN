import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Load all mel spectrograms
spectrogram_dir = Path('mel_spectrograms')
spectrogram_files = sorted(glob.glob(str(spectrogram_dir / '*.txt')))

# First pass: load all and find global min/max
spectrograms = {}
all_mels = []

for spec_file in spectrogram_files:
    if spec_file.endswith(".txt"):
        spec_name = Path(spec_file).stem
        S_db = np.loadtxt(spec_file)
        spectrograms[spec_name] = S_db
        all_mels.append(S_db)

# Find global min and max
global_min = min(db.min() for db in all_mels)
global_max = max(db.max() for db in all_mels)

# Create a figure with subplots (2 rows x 5 columns for digits 0-9)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, (spec_name, S_db) in enumerate(sorted(spectrograms.items())):
    ax = axes[idx]
    
    # Display the spectrogram with consistent scale
    img = ax.imshow(S_db, aspect='auto', origin='lower', cmap='viridis',
                    vmin=global_min, vmax=global_max, interpolation='nearest')
    ax.set_title(f'Digit: {spec_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Mel Frequency Bin')

# Add a colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(img, cax=cbar_ax, label='Power (dB)')

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.suptitle(f'Mel Spectrograms (Global Range: [{global_min:.1f}, {global_max:.1f}] dB)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig('mel_spectrograms_plot.pdf', dpi=150, bbox_inches='tight', format="pdf")
print("Saved plot to: mel_spectrograms_plot.png")
plt.show()

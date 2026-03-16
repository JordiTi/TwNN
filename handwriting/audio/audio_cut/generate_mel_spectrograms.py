import librosa
import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter


# Create output directory if it doesn't exist
output_dir = Path('mel_spectrograms')
output_dir.mkdir(exist_ok=True)

# Get all audio files
audio_files = sorted(glob.glob('*.wav'))

# First pass: compute all spectrograms and identify maximum power value
# We'll store raw mel spectrograms first, then convert to dB in a second pass once
# the true global power max is known.
raw_spectrograms = {}
spectrograms = {}
all_mels = []
# global reference power, to be determined across all files
global_power_max = -np.inf

print("Audio info:")
# compute raw spectrograms and track power maxima
hop_length=16
melcoeffs = 256
sr_global = 0
for audio_file in audio_files:
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)
    sr_global=sr

    print(f"============={audio_file}=============")
    print(f"Sample rate {sr}")
    print(f"data length: {len(y)}")
    print(f"duration: {len(y)/sr}s")
    # Compute mel spectrogram with increased time resolution

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=melcoeffs, fmax=8000, hop_length=hop_length)

    print(f"Mel coefficients: {melcoeffs}")
    print(f"Hop length: {hop_length}")
    print(f"Mel spectrogram shape:{S.shape}")

    raw_spectrograms[audio_file] = (S, sr)
    # track the largest power value seen
    power_max = S.max()
    if power_max > global_power_max:
        global_power_max = power_max

print("=============================")

# second pass: convert each spectrogram to dB using the global reference
for audio_file, (S, sr) in raw_spectrograms.items():
    S_db = librosa.power_to_db(S, ref=global_power_max)
    spectrograms[audio_file] = S_db
    all_mels.append(S_db)

# Find global min and max
global_min = min(db.min() for db in all_mels)
global_max = max(db.max() for db in all_mels)

print(f"Global min: {global_min:.2f} dB, Global max: {global_max:.2f} dB")

# Save spectrograms with consistent range
for audio_file, S_db in spectrograms.items():
    filename_stem = Path(audio_file).stem
    
    # Save as numpy file
    np.savetxt(output_dir / f'{filename_stem}.txt', S_db)

    # Also save a visualization image with consistent scale
    fig, ax = plt.subplots(figsize=(3.6, 2))
    img = librosa.display.specshow(S_db, 
                                    x_axis='time', y_axis='mel', ax=ax,
                                    vmin=global_min, vmax=global_max, fmax=8000, sr=sr_global, hop_length=hop_length)
    # ax.set_title(f'Mel Spectrogram: {filename_stem}')
    # ticks = np.linspace(0, 0.32, 6)
    # ax.set_xticks(ticks)
    # ax.set_xlim(0, 0.32)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.xlabel('t(s)', fontsize=10)
    plt.ylabel('f(Hz)', fontsize=10)
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=8) 
    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_stem}.pdf', dpi=300, bbox_inches='tight', format="pdf")
    plt.close(fig)
    print(f"Saved: {filename_stem}")

print("Done! Spectrograms saved in 'mel_spectrograms' directory")


# Audio file info:
# Audio files are 0.
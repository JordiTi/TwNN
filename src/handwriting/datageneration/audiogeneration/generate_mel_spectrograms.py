import librosa
import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
import sys

"""
Creates spectrograms from the cut data
"""

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

CUT_DIR = ROOT_DIR / "data/handwriting/audio_cut"
OUTPUT_DIR = ROOT_DIR / "data/handwriting/audio_spectrogram/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# Audio file info:
# Audio files are 0.

def raw_spectrograms(audio_files, hop_length, mel_coeffs):

        spectrograms_raw = {}
        spectrograms = {}

        # global reference power, to be determined across all files
        global_power_max = -np.inf

        print("Audio info:")
        # compute raw spectrograms and track power maxima
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

            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_coeffs, fmax=8000, hop_length=hop_length)

            print(f"Mel coefficients: {mel_coeffs}")
            print(f"Hop length: {hop_length}")
            print(f"Mel spectrogram shape:{S.shape}")

            spectrograms_raw[audio_file] = (S, sr)
            # track the largest power value seen
            power_max = S.max()
            if power_max > global_power_max:
                global_power_max = power_max
        return spectrograms_raw, global_power_max, sr_global

def spectrograms_to_db(spectrograms, global_power_max):

        all_mels = []

        # second pass: convert each spectrogram to dB using the global reference
        for audio_file, (S, sr) in spectrograms.items():
            S_db = librosa.power_to_db(S, ref=global_power_max)
            spectrograms[audio_file] = S_db
            all_mels.append(S_db)

        # Find global min and max
        global_min = min(db.min() for db in all_mels)
        global_max = max(db.max() for db in all_mels)

        print(f"Global min: {global_min:.2f} dB, Global max: {global_max:.2f} dB")

        return(spectrograms, all_mels, global_min, global_max)


def savespectrograms(spectrograms_db, all_mels, global_min, global_max, sr_global, hop_length, maximages=1):
    # Save spectrograms with consistent range
    for audio_file, S_db in spectrograms_db.items():
        filename_stem = Path(audio_file).stem
        
        # Save as numpy file
        np.savetxt(str(OUTPUT_DIR / f'{filename_stem}.txt'), S_db)

        # Also save a visualization image with consistent scale
        fig, ax = plt.subplots(figsize=(3.5, 1.7))
        img = librosa.display.specshow(S_db, 
                                        x_axis='time', y_axis='mel', ax=ax,
                                        vmin=global_min, vmax=global_max, fmax=8000, sr=sr_global, hop_length=hop_length)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        plt.xlabel('t(s)', fontsize=10)
        plt.ylabel('f(Hz)', fontsize=10)
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.tick_params(labelsize=8) 
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f'{filename_stem}.pdf'), dpi=300, bbox_inches="tight", pad_inches=0.05, format="pdf")
        plt.close(fig)
        print(f"Saved: {filename_stem}")

    print("Done! Spectrograms saved in 'mel_spectrograms' directory")
     

def main():

    hop_length=16
    mel_coeffs = 256

    # Get all audio files
    audio_files = CUT_DIR.glob('*.wav')

    spectrograms_raw, global_power_max, sr_global = raw_spectrograms(audio_files, hop_length, mel_coeffs)
    spectrograms_db, all_mels, global_min, global_max = spectrograms_to_db(spectrograms_raw, global_power_max)

    savespectrograms(spectrograms_db, all_mels, global_min, global_max, sr_global, hop_length)


if __name__ == "__main__":
    main()
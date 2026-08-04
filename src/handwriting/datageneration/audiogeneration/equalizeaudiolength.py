import os
import wave
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

WAV_DIR = ROOT_DIR / "data/handwriting/audio_raw"
OUTPUT_DIR = ROOT_DIR / "data/handwriting/audio_cut/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def shortestfilelength(wav_files):
    # Find the shortest file length
    min_frames = float('inf')
    for wav_file in wav_files:
        with wave.open(str(wav_file), 'rb') as f:
            frames = f.getnframes()
            min_frames = min(min_frames, frames)
    return min_frames

def main():

    # Get all .wav files in current directory
    wav_files = sorted(WAV_DIR.glob('*.wav'))

    min_frames = shortestfilelength(wav_files)


    # Cut all files to the shortest length
    for wav_file in wav_files:
        with wave.open(str(wav_file), 'rb') as f:
            params = f.getparams()
            audio_data = f.readframes(min_frames)
        
        # Save as [digit]_cut.wav
        output_name = str(OUTPUT_DIR / str(wav_file.stem + '_cut.wav'))
        with wave.open(output_name, 'wb') as f:
            f.setparams(params)
            f.writeframes(audio_data)
        
        print(f"Saved {output_name}")


if __name__ == "__main__":
    main()
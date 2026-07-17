import os
import wave
from pathlib import Path

# Get all .wav files in current directory
wav_files = sorted(Path('.').glob('*.wav'))

# Find the shortest file length
min_frames = float('inf')
for wav_file in wav_files:
    with wave.open(str(wav_file), 'rb') as f:
        frames = f.getnframes()
        min_frames = min(min_frames, frames)

# Cut all files to the shortest length
for wav_file in wav_files:
    with wave.open(str(wav_file), 'rb') as f:
        params = f.getparams()
        audio_data = f.readframes(min_frames)
    
    # Save as [digit]_cut.wav
    output_name = wav_file.stem + '_cut.wav'
    with wave.open(output_name, 'wb') as f:
        f.setparams(params)
        f.writeframes(audio_data)
    
    print(f"Saved {output_name}")
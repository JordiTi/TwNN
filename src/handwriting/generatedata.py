import os
from pathlib import Path
import argparse
import sys
import subprocess

"""
Generates handwriting recognition data.
Handles both the handwriting trajectory that you have to make yourself,
as well as raw audio data into audio spectrograms. Ten digits from one speaker. 
"""

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

HANDWRITING_DIR = ROOT_DIR / "src/handwriting/datageneration/"

def parse_args():
    parser = argparse.ArgumentParser(description="Train generate audio and image data for training")
    parser.add_argument("--datatype", type=str, default="digits", choices=["audio", "digits"])
    parser.add_argument("--audiooperation", type=str, default="all", choices=["equalizelength", "createspectrogram"])
    parser.add_argument("--digitoperation", type=str, default="all", choices=["writedigit", "upsample", "normalize", "plot"])
    return parser.parse_known_args()

# Run script with arguments
def run_script(script_name: str, forward_args: list = []):

    script_path = HANDWRITING_DIR / script_name
    print(script_path)
    # Combine the python execution command, the script path, and the extra flags
    command = [sys.executable, str(script_path)] + forward_args
    
    print(f"\n Running {script_name} with arguments: {forward_args if forward_args else 'None'}")
    subprocess.run(command, check=True)

def main():

    known_args, extra_args = parse_args()

    if known_args.datatype == "audio":
        if known_args.audiooperation in ["all", "equalizelength"]:
            script_name = "audiogeneration/equalizeaudiolength.py"
            run_script(script_name, extra_args)
        elif known_args.audiooperation in ["all", "createspectrogram"]:
            script_name = "audiogeneration/generate_mel_spectrograms.py"
            run_script(script_name, extra_args)
        
    elif known_args.datatype == "digits":
        if known_args.digitoperation in ["all", "writedigit"]:
            script_name = "handwritinggeneration/recordwriting.py"
            run_script(script_name, extra_args)
        elif known_args.digitoperation in ["all", "upsample"]:
            script_name = "handwritinggeneration/upsampletrajectory.py"
            run_script(script_name, extra_args)
        elif known_args.digitoperation in ["all", "normalize"]:
            script_name = "handwritinggeneration/normalizetrajectory.py"
            run_script(script_name, extra_args)
        elif known_args.digitoperation in ["all", "plot"]:
            script_name = "handwritinggeneration/plotdigits.py"
            run_script(script_name, extra_args)



if __name__ == "__main__":
    main()
"""
Run the handwriting pipeline

===============PLEASE NOTE THAT SILENCE AT THE START AND END OF THE RAW AUDIO FILES IS TRIMMED USING AUDACITY===================
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASEFUNCTION_DIR = ROOT / "src" / "handwriting"


# Run script with arguments
def run_script(script_name: str, forward_args: list = []):

    script_path = BASEFUNCTION_DIR / script_name

    # Combine the python execution command, the script path, and the extra flags
    command = [sys.executable, str(script_path)] + forward_args
    
    print(f"\n Running {script_name} with arguments: {forward_args if forward_args else 'None'}")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(prog="handwriting", description="Run handwriting pipeline")
    parser.add_file_arg = parser.add_argument(
        "--step", 
        choices=["data", 
                 "train", 
                 "evaluate", 
                 "all"
                 ], 
        default="all",
        help="Specify which pipeline step to run"
    )

    known_args, extra_args = parser.parse_known_args()

    if known_args.step in ["data", "all"]:
        run_script("generatedata.py", extra_args)
        
    if known_args.step in ["train", "all"]:
        run_script("train.py", extra_args)
        
    if known_args.step in ["evaluate", "all"]:
        run_script("evaluate.py", extra_args)

if __name__ == "__main__":
    main()
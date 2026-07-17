"""
Run the base function pipeline
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Define paths
ROOT = Path(__file__).resolve().parent
BASEFUNCTION_DIR = ROOT / "src" / "basefunctions"

# Run script with arguments
def run_script(script_name: str, forward_args: list):

    script_path = BASEFUNCTION_DIR / script_name

    # Combine the python execution command, the script path, and the extra flags
    command = [sys.executable, str(script_path)] + forward_args
    
    print(f"\n Running {script_name} with arguments: {forward_args if forward_args else 'None'}")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(prog="basefunction", description="Run base function pipeline")
    parser.add_file_arg = parser.add_argument(
        "--step", 
        choices=["data", 
                 "train", 
                 "evaluate", 
                 "all",
                 "plotbasefunctions"], 
        default="all",
        help="Specify which pipeline step to run"
    )

    known_args, extra_args = parser.parse_known_args()

    if known_args.step in ["data", "all"]:
        run_script("generatebasefunctions.py", extra_args)
        
    if known_args.step in ["train", "all"]:
        run_script("train.py", extra_args)
        
    if known_args.step in ["evaluate", "all"]:
        run_script("evaluate.py", extra_args)
    
    if known_args.step == "plotbasefunctions":
        run_script("visualize_data.py")

if __name__ == "__main__":
    main()
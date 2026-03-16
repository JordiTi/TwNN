#!/bin/bash

#SBATCH --job-name=pongjob
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400M

module purge
module load Python/3.10.8-GCCcore-12.2.0

echo activating venv
source /home3/p309238/venvs/archer_env/bin/activate

python speechtohandwriting_threelayers.py -l ${1} -o ${2} -m ${3} -a ${4} -t ${5}
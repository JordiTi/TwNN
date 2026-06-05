#!/bin/bash

#SBATCH --job-name=pongjob
#SBATCH --time=40:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500M

module purge
module load Python/3.10.8-GCCcore-12.2.0

echo activating venv
source /home3/p309238/venvs/archer_env/bin/activate
echo r: ${6}
python speechtohandwriting_topthree_x.py -l ${1} -o ${2} -m ${3} -a ${4} -t ${5} -r ${6}
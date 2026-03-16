#!/bin/bash

#SBATCH --job-name=51001000
#SBATCH --time=8:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300M

module purge
module load Python/3.10.8-GCCcore-12.2.0

echo activating venv
source /scratch/p309238/pongvenv/bin/activate

python fba_vanilla_reworked.py -l ${1} -z ${2} -t ${3} -m ${4} -d ${5}

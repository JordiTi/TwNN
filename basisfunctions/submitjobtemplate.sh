#!/bin/bash

#SBATCH --job-name=pongjob
#SBATCH --time=3:00:00
#SBATCH --begin=now+17hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300M

module purge
module load Python/3.10.8-GCCcore-12.2.0

echo activating venv
source /home3/p309238/venvs/archer_env/bin/activate
python fitfunctions_twolayers.py -l ${1} -m ${2} -a ${3} -i ${4} -f ${5} -t ${6}
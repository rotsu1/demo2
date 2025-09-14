#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

#SBATCH --job-name=part3.3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

#SBATCH --output=part3.3.out
#SBATCH --error=part3.3.err

source /home/Student/s4800316/.bashrc

conda activate p3

python part3.3.py

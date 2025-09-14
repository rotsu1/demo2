#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

#SBATCH --job-name=p4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --output=p4.out
#SBATCH --error=p4.err

source /home/Student/s4800316/.bashrc

conda activate p3

python p4.py

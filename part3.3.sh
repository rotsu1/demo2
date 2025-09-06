#!/bin/bash
#SBATCH --partition=ai
#SBATCH --gres=gpu:1

#SBATCH --job-name=part1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00

#SBATCH --output=part3.3.out
#SBATCH --error=part3.3.err

source /home/Student/s4800316/.bashrc

conda activate conda-pytorch

python part3.3.py
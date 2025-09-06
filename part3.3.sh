#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

#SBATCH --job-name=part1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

#SBATCH --output=part3.3.out
#SBATCH --error=part3.3.err

source /home/Student/s4800316/.bashrc

conda activate conda-pytorch

python part3.3.py
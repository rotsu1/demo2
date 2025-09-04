#!/bin/bash
#SBATCH --partition=ai
#SBATCH --gres=gpu:1

#SBATCH --job-name=part1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00

#SBATCH --output=part1.out
#SBATCH --error=part1.err

module load anaconda3/

source /home/${USER}/.bashrc

conda activate conda-pytorch

python part1.py
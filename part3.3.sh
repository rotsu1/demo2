#!/bin/bash
#SBATCH --partition=ai
#SBATCH --gres=gpu:1

#SBATCH --job-name=part1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00

#SBATCH --output=part3.3.out
#SBATCH --error=part3.3.err

module load anaconda3/

source /home/${USER}/.bashrc

conda activate conda-pytorch

python part3.3.py
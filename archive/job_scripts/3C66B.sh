#!/usr/bin/env bash
#SBATCH --job-name=3C66B_det
#SBATCH --partition=pi_mingarelli
#SBATCH --time=3-
#SBATCH --output=logs/3C66B_det.txt
#SBATCH --error=logs/error/3C66B_det.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G

module load miniconda
conda activate targeted

export PYTHONPATH=$(pwd):$PATH
python scripts/recreate_3C66B.py

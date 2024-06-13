#!/usr/bin/env bash
#SBATCH --job-name=targeted
#SBATCH --time=10:00:00
#SBATCH --partition=scavenge
#SBATCH --output=logs/simple.txt
#SBATCH --error=logs/error/simple.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G

module load miniconda
conda activate # name of your conda environment

python scripts/simple_targeted_search.py
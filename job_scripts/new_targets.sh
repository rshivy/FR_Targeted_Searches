#!/usr/bin/env bash
#SBATCH --job-name=targeted001
#SBATCH --time=10:00:00
#SBATCH --partition=scavenge
#SBATCH --output=logs/targeted/001.txt
#SBATCH --error=logs/error/targeted/001.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G

module load miniconda
module load OpenMPI
conda activate targeted # name of your conda environment

export PYTHONPATH=$(pwd):$PATH
python scripts/simple_targeted_search.py -t 001

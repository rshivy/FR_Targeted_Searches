#!/usr/bin/env bash
#SBATCH --job-name=mock-ul
#SBATCH --time=7-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --output=logs/mock_ul.txt
#SBATCH --error=logs/error/mock_ul.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PYTHONPATH
srun -n 8 python scripts/mock_search.py -m upper-limit

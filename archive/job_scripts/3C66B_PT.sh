#!/usr/bin/env bash
#SBATCH --job-name=3C66B_det
#SBATCH --partition=pi_mingarelli
#SBATCH --time=2-
#SBATCH --output=logs/3C66B_det.txt
#SBATCH --error=logs/error/3C66B_det.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5G

module load miniconda OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PATH
mpirun -n 4 python scripts/recreate_3C66B.py

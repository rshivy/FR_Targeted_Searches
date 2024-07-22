#!/usr/bin/env bash
#SBATCH --job-name=t001
#SBATCH --time=3-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --output=logs/target_001_det.txt
#SBATCH --error=logs/error/target_001_det.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PYTHONPATH
mpirun -n 4 python scripts/full_targeted.py

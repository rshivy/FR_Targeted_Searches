#!/usr/bin/env bash
#SBATCH --job-name=all-sky
#SBATCH --time=7-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --output=logs/all-sky.txt
#SBATCH --error=logs/error/all-sky.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PATH
srun -n 8 python scripts/all_sky_ul.py

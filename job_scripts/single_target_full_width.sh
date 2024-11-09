#!/usr/bin/env bash
#SBATCH --job-name=full-width
#SBATCH --time=3-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --output=logs/single_target_full_width.txt
#SBATCH --error=logs/error/single_target_full_width.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PATH
srun -n 8 python scripts/full_targeted.py -t 1 -m upper-limit -f full -o fullwidth

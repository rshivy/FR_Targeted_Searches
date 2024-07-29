#!/usr/bin/env bash
#SBATCH --job-name=t001
#SBATCH --time=3-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --output=logs/target_001_det_narrowfgw.txt
#SBATCH --error=logs/error/target_001_det_narrowfgw.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/full_targeted.py -t 001

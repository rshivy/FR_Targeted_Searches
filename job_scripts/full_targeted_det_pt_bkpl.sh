#!/usr/bin/env bash
#SBATCH --job-name=t1bkpl
#SBATCH --time=3-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --output=logs/t1bkpl_det_narrowfgw_pt_v3.txt
#SBATCH --error=logs/error/t1bkpl_det_narrowfgw_pt_v3.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PYTHONPATH
srun -n 8 python scripts/full_targeted.py -o bkpl -t 001 -m detection -f narrow -b

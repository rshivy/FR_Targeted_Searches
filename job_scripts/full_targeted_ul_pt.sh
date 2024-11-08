#!/usr/bin/env bash
#SBATCH --job-name=t001pt-ul
#SBATCH --time=3-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --output=logs/target_001_ul_narrowfgw_pt_v2.txt
#SBATCH --error=logs/error/target_001_ul_narrowfgw_pt_v2.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PYTHONPATH
srun -n 8 python scripts/full_targeted.py -o pt_v2 -t 001 -m upper-limit -f narrow

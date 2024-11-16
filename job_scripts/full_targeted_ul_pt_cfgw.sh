#!/usr/bin/env bash
#SBATCH --job-name=201-ul-cfgw
#SBATCH --time=5-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --output=logs/target_201_ul_cfgw_pt_v3.txt
#SBATCH --error=logs/error/target_201_ul_cfgw_pt_v3.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PYTHONPATH
srun -n 8 python scripts/full_targeted.py -o pt_v3 -t 201 -m upper-limit -f constant

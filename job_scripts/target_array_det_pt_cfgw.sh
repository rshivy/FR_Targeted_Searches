#!/usr/bin/env bash
#SBATCH --job-name=v3-d-c
#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenge
#SBATCH --array=1,3,5,7,11,13,15,17,19,21
#SBATCH --output=logs/target_%3a_det_cfgw_pt_v3.txt
#SBATCH --error=logs/error/target_%3a_det_cfgw_pt_v3.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --requeue

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PATH
srun -n 8 python scripts/full_targeted.py -t $SLURM_ARRAY_TASK_ID -m detection -f constant -o cfgw_pt_v3

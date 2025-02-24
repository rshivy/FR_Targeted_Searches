#!/usr/bin/env bash
#SBATCH --job-name=v4-u-c
#SBATCH --time=7-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --array=1-10
#SBATCH --output=logs/target_%3a_ul_cfgw_pt_v4.txt
#SBATCH --error=logs/error/target_%3a_ul_cfgw_pt_v4.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PATH
srun -n 8 python scripts/full_targeted.py -t $SLURM_ARRAY_TASK_ID -m upper-limit -f constant -o pt_v4

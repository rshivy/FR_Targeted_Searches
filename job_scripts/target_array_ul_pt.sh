#!/usr/bin/env bash
#SBATCH --job-name=t1-21-ul-pt
#SBATCH --time=3-00:00:00
#SBATCH --partition=pi_mingarelli
#SBATCH --array=1,3,5,7,11,13,15,17,19,21
#SBATCH --output=logs/target_%3a_ul_narrowfgw_pt_v2.txt
#SBATCH --error=logs/error/target_%3a_ul_narrowfgw_pt_v2.txt
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G

module load miniconda
module load OpenMPI
conda activate targeted

export PYTHONPATH=$(pwd):$PATH
srun -n 8 python scripts/full_targeted.py -t $SLURM_ARRAY_TASK_ID -m upper-limit -f narrow -o pt_v2
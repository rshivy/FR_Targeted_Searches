#!/usr/bin/env bash
#SBATCH --job-name=tarr
#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenge
#SBATCH --array=12-36
#SBATCH --output=logs/targeted/%a.txt
#SBATCH --error=logs/error/targeted/%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

module load miniconda
module load OpenMPI
conda activate targeted # name of your conda environment

export PYTHONPATH=$(pwd):$PATH
python scripts/simple_targeted_search.py -t $SLURM_ARRAY_TASK_ID

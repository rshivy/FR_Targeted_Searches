#!/usr/bin/env bash
#SBATCH --job-name=targets1-24
#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenge
#SBATCH --array=1-24
#SBATCH --output=logs/targeted/%a.txt
#SBATCH --error=logs/error/targeted/%a.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --mail-type=END
#SBATCH --mail-user=forrest.hutchison@yale.edu

module load miniconda
module load OpenMPI
conda activate targeted # name of your conda environment

export PYTHONPATH=$(pwd):$PATH
mpirun -n 4 python scripts/simple_targeted_search.py -t $SLURM_ARRAY_TASK_ID

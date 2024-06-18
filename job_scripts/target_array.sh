#!/usr/bin/env bash
<<<<<<< HEAD
#SBATCH --job-name=targeted001
=======
#SBATCH --job-name=tarr
>>>>>>> ba323d8 (job array support)
#SBATCH --time=10:00:00
#SBATCH --partition=scavenge
#SBATCH --array=2-11
#SBATCH --output=logs/targeted/%a.txt
#SBATCH --error=logs/error/targeted/%a.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G

module load miniconda
module load OpenMPI
conda activate targeted # name of your conda environment

export PYTHONPATH=$(pwd):$PATH
python scripts/simple_targeted_search.py -t $SLURM_ARRAY_TASK_ID

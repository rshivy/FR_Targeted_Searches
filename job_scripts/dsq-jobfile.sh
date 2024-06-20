#!/bin/bash
#SBATCH --output logs/targeted/%a.txt
#SBATCH --array 0-111
#SBATCH --job-name targeted
#SBATCH --time 1- --partition scavenge --error logs/error/targeted/%a.txt --nodes 1 --ntasks 4 --cpus-per-task 1 --mem-per-cpu 10G --mail-type END --mail-user=forrest.hutchison@yale.edu

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/frh7/FR_Targeted_Searches/job_scripts/joblist.txt --status-dir /vast/palmer/home.grace/frh7/FR_Targeted_Searches/logs/targeted


#!/bin/bash
#SBATCH --output logs/targeted-rerun/%a.txt
#SBATCH --array=96,97,98,100,101,102,103,104,105,106,107,108,109,110,111
#SBATCH --job-name rerun
#SBATCH --time 1- --partition scavenge --error logs/error/targeted-rerun/%a.txt --nodes 1 --ntasks 1 --cpus-per-task 1 --mem-per-cpu 10G

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/frh7/FR_Targeted_Searches/job_scripts/rerun_jobs.txt --status-dir /vast/palmer/home.grace/frh7/FR_Targeted_Searches/logs/targeted-rerun


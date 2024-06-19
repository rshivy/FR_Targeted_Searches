#!/usr/bin/env bash

module purge
module load python

python job_scripts/dsqwriter.py

module load dSQ

dsq --job-file job_scripts/joblist.txt \
--batch-file job_scripts/dsq-jobfile.sh \
--status-dir logs/targeted/ \
--job-name targeted \
--time 1- \
--partition scavenge \
--output logs/targeted/%a.txt \
--error logs/error/targeted/%a.txt \
--ntasks 4 \
--cpus-per-task 1 \
--mem-per-cpu 10G \
--mail-type END \
--mail-user=forrest.hutchison@yale.edu

echo 'submit with sbatch job_scripts/dsq-jobfile.sh'
echo Done
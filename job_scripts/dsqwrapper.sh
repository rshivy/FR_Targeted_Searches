#!/usr/bin/env bash

high=$1
if [ -z "$1" ]
then
  high=111
fi


module purge

python3 job_scripts/dsqwriter.py $high

module load dSQ

dsq --job-file job_scripts/joblist.txt \
--batch-file job_scripts/dsq-jobfile.sh \
--status-dir logs/targeted/ \
--job-name targeted \
--array 1-$1 \
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
#!/usr/bin/env bash

tasks=$1 # First argument is number of tasks
if [ -z "$1" ]
then # If first argument is not passed then set to one
  tasks=1
fi
high=$2 # Second argument is number of targets to include
if [ -z "$2" ]
then # If second argument is not passed then set to 111 (all of them)
  high=111
fi


module purge

python3 job_scripts/dsqwriter.py $high $tasks

module load dSQ

dsq --job-file job_scripts/joblist.txt \
--batch-file job_scripts/dsq-jobfile.sh \
--status-dir logs/targeted/ \
--job-name targeted \
--time 3- \
--partition pi_mingarelli \
--output logs/targeted/%a.txt \
--error logs/error/targeted/%a.txt \
--nodes 1 \
--ntasks $tasks \
--cpus-per-task 1 \
--mem-per-cpu 10G \
--mail-type END \
--mail-user=forrest.hutchison@yale.edu

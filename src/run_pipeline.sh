#!/bin/bash

set -x

# 1. Submit the training job and extract its Slurm Job ID
# (Assuming your first script is named 'training.slurm')
rm -r log.training.out
TRAINING_MSG=$(sbatch training.slurm)
echo "$TRAINING_MSG"

# Extract just the numeric digits of the Job ID from the string "Submitted batch job 123456"
TRAINING_JOB_ID=$(echo "$TRAINING_MSG" | awk '{print $4}')

# 2. Submit the forecast job with a strict dependency
# 'afterok' ensures forecast ONLY runs if training exits with a exit code of 0 (successful completion)
rm -r log.forecast.out
echo "Submitting forecast job to queue dependent on Training Job ID: $TRAINING_JOB_ID..."
sbatch --dependency=afterok:$TRAINING_JOB_ID forecast.slurm


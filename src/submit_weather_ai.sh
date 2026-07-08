#!/bin/bash
#SBATCH --job-name=h3_weather_ai
#SBATCH --partition=gpu               # Your target cluster GPU queue name
#SBATCH --nodes=2                     # Match the num_nodes parameter in your Trainer script
#SBATCH --ntasks-per-node=4           # Number of tasks per node (must match the number of physical GPUs per node)
#SBATCH --gres=gpu:4                  # Allocate 4 GPUs per physical cluster node
#SBATCH --cpus-per-task=8             # Allocate CPU threads to support datamodule num_workers processing
#SBATCH --memory=128G                 # System RAM allocation pool per node
#SBATCH --time=12:00:00               # Max execution time limit clock window
#SBATCH --output=logs/h3_ai_%j.out

# Load your cluster's CUDA and network communication modules
module load cuda/12.1
module load ucx
module load openmpi

# Activate your custom conda environment
source activate anemoi

# Set environment variables to optimize distributed scaling performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0              # Set to 0 to enable InfiniBand inter-node communication
export PYTHONUNBUFFERED=1

# Retrieve the master node's network IP address to coordinate the worker pool
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500              # Port communication channel

echo "Launching training job. Master node address is: $MASTER_ADDR"

# Execute using srun to initialize process threads across the allocated cluster nodes
srun python -u run_distributed_training.py


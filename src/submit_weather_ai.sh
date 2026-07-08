#!/bin/bash
#SBATCH --job-name=h3_weather_ai
#SBATCH --account=epic
#SBATCH --chdir=/scratch4/NAGAPE/epic/Wei.Huang/src/h3/src
#SBATCH --export=NONE
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --output=log.training.out
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=01:00:00

set -x

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

# Set environment variables to optimize distributed scaling performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0              # Set to 0 to enable InfiniBand inter-node communication
export PYTHONUNBUFFERED=1

# Retrieve the master node's network IP address to coordinate the worker pool
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500              # Port communication channel

# Step 1: Create the Configuration File (config.yaml)

# Step 2: Create the Topology Generator Script (build_graph.py)
#python build_graph.py

# Step 3: Create the Complete PyTorch Training Pipeline (train_distributed.py)
 python train_distributed.py

#echo "Launching training job. Master node address is: $MASTER_ADDR"

# Execute using srun to initialize process threads across the allocated cluster nodes
#srun python -u ../src/run_distributed_training.py


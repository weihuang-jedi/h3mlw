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

#pip install litlogger

# Ensure output streams log in real-time without print buffering lags
export PYTHONUNBUFFERED=1

echo "Launching Single-GPU training run on node: $SLURMD_NODENAME"

# Step 1: Create the Configuration File (config.yaml)

# Step 2: Create the Topology Generator Script (build_graph.py)
#python build_graph.py

#rm -f ../data/checkpoints/*
#rm -rf ../lightning_logs/*

# Step 3: Create the Complete PyTorch Training Pipeline (train_distributed.py)
python train_single_gpu.py

# nvidia-smi -l 1

#echo "Launching training job. Master node address is: $MASTER_ADDR"

# Execute using srun to initialize process threads across the allocated cluster nodes
#srun python -u ../src/run_distributed_training.py

# Step 4:

python rollout_forecast.py
#python evaluate_acc.py
#python plot_loss_function.py
#python ../utils/interpolate2latlon.py -i h3_autoregressive_forecast.nc -o autoregressive_forecast.nc -t ../data/air.sfc.2000.nc -w 4
#python animate_forecast.py
#python ../utils/plot_regular_grid.py -i autoregressive_forecast.nc -s


#!/bin/bash
#SBATCH --job-name=forecast
#SBATCH --account=epic
#SBATCH --export=NONE
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --output=log.forecast.out
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=00:15:00

set -x

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

#pip install litlogger


cd /scratch4/NAGAPE/epic/Wei.Huang/src/h3/src/training

python run_forecast.py --config config_inference.yaml


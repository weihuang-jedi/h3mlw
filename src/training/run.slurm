#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=epic
#SBATCH --export=NONE
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --output=log.training.out
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=04:00:00

set -x

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

#pip install litlogger

# Ensure output streams log in real-time without print buffering lags
export PYTHONUNBUFFERED=1

cd /scratch4/NAGAPE/epic/Wei.Huang/src/h3/src/training

python training_on_icosahedral_grid.py -c config.yaml


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

# Ensure output streams log in real-time without print buffering lags
export PYTHONUNBUFFERED=1

cd /scratch4/NAGAPE/epic/Wei.Huang/src/h3/src

rm -f h3_autoregressive_forecast.nc autoregressive_forecast.nc animate_forecast.gif
python rollout_forecast.py

#python ../utils/interpolate2latlon.py -i h3_autoregressive_forecast.nc -o autoregressive_forecast.nc -t ../data/air.sfc.2000.nc -w 4

#python ../utils/animate_regular_grid.py -i autoregressive_forecast.nc -o animate_forecast.gif

#python evaluate_acc.py -t ../data/global_h3_res2_air_sfc_2000.nc -p h3_autoregressive_forecast.nc
#python plot_loss_function.py
#python ../utils/plot_regular_grid.py -i autoregressive_forecast.nc -s

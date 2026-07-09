#!/bin/bash

set -x

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

#totalyears=24
totalyears=2
startyear=1976

webdir=https://downloads.psl.noaa.gov/Datasets/20thC_ReanV2c/gaussian/monolevel

SRC_DIR=/scratch4/NAGAPE/epic/Wei.Huang/src/h3
DATADIR=${SRC_DIR}/data
UTILDIR=${SRC_DIR}/utils

cd ${DATADIR}

n=0
while [ "${n}" -le "${totalyears}" ]
do
   curyear=$(( startyear + n ))
   flnm=air.sfc.${curyear}.nc
   if [ ! -f ${flnm} ]
   then
      wget ${webdir}/${flnm}
      rm -f interpolate_all_timesteps.py
   fi
   OUTPUT_FLNM="global_h3_res2_air_sfc_${curyear}.nc"
   if [ ! -f ${OUTPUT_FLNM} ]
   then
      python ${UTILDIR}/interpolate2h3.py \
	      -i ${flnm} -o ${OUTPUT_FLNM}
   fi

   echo "python ${UTILDIR}/plot_regular_grid.py -i ${DATADIR}/${flnm} -o ${DATADIR}/air.sfc.${curyear}.png -s"

   n=$(( n + 1 ))
done

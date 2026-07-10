#!/bin/bash

set -x

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

totalyears=24
#totalyears=2
startyear=1976

webdir=https://downloads.psl.noaa.gov/Datasets/20thC_ReanV2c/gaussian/monolevel

SRC_DIR=/scratch4/NAGAPE/epic/Wei.Huang/src/h3
DATADIR=${SRC_DIR}/data
UTILDIR=${SRC_DIR}/utils

cd ${UTILDIR}

n=0
while [ "${n}" -le "${totalyears}" ]
do
   curyear=$(( startyear + n ))
   flnm=air.sfc.${curyear}.nc
   if [ ! -f ${DATADIR}/${flnm} ]
   then
      wget ${webdir}/${flnm} ${DATADIR}/${flnm}
   fi
   OUTPUT_FLNM="google-graphcast-grid/global_icosahedral_m4_air_${curyear}.nc"
   # rm -f ${OUTPUT_FLNM}
   if [ ! -f ${OUTPUT_FLNM} ]
   then
      python interpolate2icosahedral.py \
        --input ${DATADIR}/${flnm} \
        --mesh google-graphcast-grid/global_icosahedral_mesh_m4.nc \
        --output ${OUTPUT_FLNM}
   fi

   n=$(( n + 1 ))
done

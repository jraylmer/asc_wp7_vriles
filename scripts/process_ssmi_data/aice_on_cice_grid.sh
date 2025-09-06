#!/bin/bash
#
# Interpolates SSM/I raw daily data onto CICE grid. First runs the auxiliary
# python script _prepare_raw_for_interp.py, which combines the raw daily
# data files into a yearly file with preset netCDF metadata and also determines
# the location of the CICE land mask file and output directory from the python
# configuration info (i.e., as set in module src/io/config.py). Interpolation
# is then carried out using CDO.
#
# --------------------------------------------------------------------------- #

CFG=default  # configuration (ini) file(s) [use quotes on command line if > 1]
YRS=1979     # start year
YRE=1979     # end year
DATASET=nt   # dataset (nt or bt)

# Command-line flags to change from default values above:
while getopts c:s:e:d: flag
do
    case "${flag}" in
        c) CFG=${OPTARG};;
        s) YRS=${OPTARG};;
        e) YRE=${OPTARG};;
        d) DATASET=${OPTARG};;
    esac
done

# Validate command line options:
if (( YRE < YRS ))
then
    YRE=${YRS}
fi

if [ "${DATASET}" == "nt" ]
then
    PREFIX="NSIDC-0051_nasateam_v2_aice_cice_grid_"  # filename prefix
elif [ "${DATASET}" == "bt" ]
then
    PREFIX="NSIDC-0079_bootstrap_v4_aice_cice_grid_"  #filename prefix
else
    echo "Invalid flag -d, choose 'nt' or 'bt'"
    exit 1
fi

# Start calculations: iterate over years:
for y in $(seq ${YRS} ${YRE})
do
    # Merge time and prepare daily data for one year via Python script.
    # This also generates a pole-hole mask for this year (can vary in time
    # for some years where sensor changes), to be interpolated separately
    # and then applied to the interpolated sea ice concentration data.
    #
    # This script also prints, at the end, the location of the land mask and
    # output directory which it gets from the python configuration(s) ${CFG}
    #
    PYOUT=($(python ./scripts/process_ssmi_data/_prepare_raw_for_interp.py \
        -c ${CFG} -d ${DATASET} -y ${y}                                    \
	-o ./_tmp_${DATASET}_${y}.nc -p ./_tmp_${DATASET}_${y}_pmask.nc    \
	-a "methods" "remapped using CDO remapdis,8"))

    LMSKFILE=${PYOUT[-2]}
    OUTDIR=${PYOUT[-1]}

    if [ ! -f "${LMSKFILE}" ]
    then
	exit 1  # something must have gone wrong in python step, so stop
    fi

    mkdir -p ${OUTDIR}

    # Interpolate pole-hole mask to CICE grid using CDO
    # -> use the land mask file for the target grid
    #
    cdo -O remapdis,${LMSKFILE},8 ./_tmp_${DATASET}_${y}_pmask.nc \
        ./_tmp_${DATASET}_${y}_pmask_remapped.nc

    # Interpolate sea ice concentration data to CICE grid using CDO
    # -> use the land mask file for the target grid
    # -> apply CICE land mask and pole-hole mask after interpolation
    # -> set all missing values to NaN afterwards
    #
    cdo -O --no_history -z zip_2 -setmisstoc,nan -div -div -remapdis,${LMSKFILE},8     \
        ./_tmp_${DATASET}_${y}.nc ${LMSKFILE} ./_tmp_${DATASET}_${y}_pmask_remapped.nc \
	${OUTDIR}/${PREFIX}${y}.nc

    if [ -f "${OUTDIR}/${PREFIX}${y}.nc" ]
    then
        echo "Saved: ${OUTDIR}/${PREFIX}${y}.nc"
        rm -f ./_tmp_${DATASET}_${y}*.nc  # intermediate files, no longer needed
    else
	exit 1  # something must have gone wrong with CDO step so stop
    fi

done

exit 0


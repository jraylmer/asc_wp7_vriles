#!/bin/bash

# Interpolate raw atmospheric forcing/reanalysis data onto CICE grid.
#
# Note this script and the auxiliary 'cleanup' scripts are partly hardcoded
# to work with JRA-55-do only; it is mainly included in the repository as a
# reference and 'for the record' of the method originally used.
#
# The paths to the raw data and interpolation grid file must be provided by
# the flags -i and -g respectively, i.e., the bash script is not tied to
# python cfg/*.ini files for these. However, the companion python cleanup
# script (which is called during this bash script) is, so if the config is
# set via the environment variable ASC_WP7_VRILES_CFG or via the local flag
# -c to this script, then it can get metadata from there. For example, the
# default output data directory (if not set here via flags) is taken from
# cfg.data_path['atmo_forc'].
#
# The script also calculates daily and monthly averages of the forcing fields.
# If the flags -d (for daily output directory) and -m (for monthly output
# directory) are not provided, the default is to get them from the config.
# --------------------------------------------------------------------------- #

# Variable                    Description                                 Flag
# --------                    -----------                                 ----
inDir=""                      # Raw data location (then ${varname}/*.nc)  -i
outDir=""                     # Output directory (forcing files)          -o
gridFile=""                   # Grid file for CDO interpolation           -g
yr0=1979                      # Start of year range to process            -s
yr1=2023                      # End   of year range to process            -e
outDirMon=""                  # Output directory for monthly means        -m
outDirDay=""                  # Output directory for daily means          -d
cfg=${ASC_WP7_VRILES_CONFIG}  # Configuration(s)                          -c
showHelp="false"              # Show help message on flags and exit       -h

while getopts i:o:g:a:b:s:e:m:d:c:h flag
do
    case "${flag}" in
        i) inDir=${OPTARG};;
        o) outDir=${OPTARG};;
        g) gridFile=${OPTARG};;
        s) yr0=${OPTARG};;
        e) yr1=${OPTARG};;
        m) outDirMon=${OPTARG};;
        d) outDirDay=${OPTARG};;
        c) cfg=${OPTARG};;
        h) showHelp="true";;
    esac
done


if [[ "${showHelp}" == "true" ]]
then
    echo ""
    echo "scripts/process_atmo_data/interp_to_cice_grid.sh"
    echo ""
    echo "    Bash script to interpolate raw atmospheric reanalysis (JRA-55-do) data onto CICE grid"
    echo ""
    echo "Required flags:"
    echo "    -i <directory>        Directory of raw data (with names of variables as subdirectories)"
    echo "    -g <path>             Path to grid file for CDO interpolation"
    echo ""
    echo "Optional flags:"
    echo "    -c '<cfg1> [cfg2]'    Repository config(s) (or use environment variable ASC_WP7_VRILES_CONFIG)"
    echo "    -o <directory>        Output directory (default in config)"
    echo "    -s <int>              Start year of range to process (default 1979)"
    echo "    -e <int>              End year of range to process (default 2023)"
    echo "    -m <directory>        Output directory for monthly means (default in config)"
    echo "    -d <directory>        Output directory for daily means (default in config)"
    echo "    -h                    Show this message and exit"
    echo ""
    exit 0
fi


if [[ "${inDir}" == "" ]]
then
    echo "Input directory required (-i /path/to/data)"
    exit 1
elif [[ "${gridFile}" == "" ]]
then
    echo "Grid file for interpolation required (-g /path/to/grid/file.txt)"
    exit 1
fi

years=($(seq ${yr0} ${yr1}))


# All required non-wind variables: variable names are different for input and output
# files (there is assumed consistency of this info in the cleanup python script):
#
varIn=("tas" "huss" "psl" "rlds" "rsds" "prra" "prsn")
varOut=("t2" "q2" "psl" "qlw" "qsw" "precip" "snow")

# Similar for winds (these need to be separate due to extra rotation step
# indicated by different arguments passed to cleanup script):
#
uIn="uas"
uOut="u10"
vIn="vas"
vOut="v10"


for y in ${years[@]}
do
    for ((i=0; i < ${#varIn[@]}; i++))
    do
        # Interpolate and remove leap days:
        cdo -O -del29feb -remapdis,${gridFile}   \
            ${inDir}/${varIn[i]}/*${y}*${y}*.nc  \
            ./${varIn[i]}_${y}_remapped.nc

        # Cleanup the interpolated files. This script also prints, at the end,
        # the output directories for the forcing and daily/monthly averages
        # from the python configuration files. In each case, it is either the
        # same as the flag -o, -d, or -m value, or got from config if the
        # respective flag not passed to the present script. In the latter case,
        # quotes around the various outDir variables are required as defaults
        # are '' (empty strings):
        pyout=($(python -W ignore ./scripts/process_atmo_data/_clean_interp.py          \
            -v ${varIn[i]} -y ${y} -i ./${varIn[i]}_${y}_remapped.nc --config ${cfg}    \
            -o "${outDir}" --out-dir-day "${outDirDay}" --out-dir-mon "${outDirMon}"))

        rm ./${varIn[i]}_${y}_remapped.nc  # delete intermediate file
	
        # These are the same for each variable/year so this is repeated
        # unnecessarily each loop iteration, but easier for script structure):
        outDir=${pyout[-3]}
        outDirDay=${pyout[-2]}
        outDirMon=${pyout[-1]}

        mkdir -p ${outDir} ${outDirDay} ${outDirMon}

        # Compute daily and monthly means (daily to be compressed, below):
        cdo --no_history daymean ${outDir}/${varOut[i]}_y${y}.nc  \
            ${outDirDay}/${varOut[i]}_d_y${y}_uncomp.nc

        cdo --no_history monmean ${outDir}/${varOut[i]}_y${y}.nc  \
            ${outDirMon}/${varOut[i]}_m_y${y}.nc

        # Compress the daily outputs (not much point for monthly):
        ncks -4 -L 5 ${outDirDay}/${varOut[i]}_d_y${y}_uncomp.nc  \
                     ${outDirDay}/${varOut[i]}_d_y${y}.nc
        rm           ${outDirDay}/${varOut[i]}_d_y${y}_uncomp.nc
    done

    # Wind done separately as both components needed for rotation to CICE grid
    # orientation. Interpolate each component, then pass both to cleanup script:
    cdo -O -del29feb -remapdis,${gridFile}  \
        ${inDir}/${uIn}/*${y}*${y}*.nc      \
        ./${uOut}_${y}_remapped.nc

    cdo -O -del29feb -remapdis,${gridFile}  \
        ${inDir}/${vIn}/*${y}*${y}*.nc      \
        ./${vOut}_${y}_remapped.nc

    # Cleanup and rotate components:
    python -W ignore ./scripts/process_atmo_data/_clean_interp.py  \
        --no-print-dirs --config ${cfg} -y ${y} -v ${uIn} ${vIn}   \
        -i ./${uOut}_${y}_remapped.nc ./${vOut}_${y}_remapped.nc   \
        -o "${outDir}" --out-dir-day "${outDirDay}" --out-dir-mon "${outDirMon}"

    rm ./${uOut}_${y}_remapped.nc ./${vOut}_${y}_remapped.nc

    # Compute daily and monthly means and compress the former:
    for v in ${uOut} ${vOut}
    do
        cdo --no_history daymean ${outDir}/${v}_y${y}.nc  \
            ${outDirDay}/${v}_d_y${y}_uncomp.nc

        cdo --no_history monmean ${outDir}/${v}_y${y}.nc  \
            ${outDirMon}/${v}_m_y${y}.nc

        ncks -4 -L 5 ${outDirDay}/${v}_d_y${y}_uncomp.nc  \
                     ${outDirDay}/${v}_d_y${y}.nc
        rm           ${outDirDay}/${v}_d_y${y}_uncomp.nc
    done

done

exit 0


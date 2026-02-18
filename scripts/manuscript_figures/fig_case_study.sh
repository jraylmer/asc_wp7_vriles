#!/bin/bash

# Generate figure for VRILE case studies. This uses the generic script for case
# studies, vrile_case_study.py, and so this bash script is just a shortcut that
# passes the correct command-line options (see below) to that python script for
# each case study. This script should still be run from the the top-level
# repository directory, as with the other python scripts. Usage:
#
#     $ bash ./scripts/manuscript_figures/fig_case_study.sh [-c <cfgs>] [-s] [-v <v>]
#
# where
#
#     -c <cfgs> is a configuration file to use (sets data paths, etc.).
#               For multiple configuration files, enclose in quotes, e.g.:
#
#                   -c "cfg_1 cfg_2"
#
#     -s        indicates to save the figure (flag only; no arguments)
#
#     -v <v>    Sets which case:  1 = Sep 2018, 2 = Jul 2004, 3 = Aug 2022
#                                S3 = Jul 2004 thermo terms (Fig. S3)
#
# --------------------------------------------------------------------------- #

# Default options (-v, -c, -s respectively):
vri="1"
cfg=${ASC_WP7_VRILES_CONFIG}
sav=""

while getopts c:v:s flag
do
    case "${flag}" in
        c) cfg="${OPTARG}";;
	v) vri=${OPTARG};;
	s) sav="--savefig";;
    esac
done

if [[ "${vri}" == "1" ]]
then
    # Case study 1: 22-30 September 2018
    # ----------------------------------
    python -W ignore ./scripts/manuscript_figures/_case_study.py      \
        --region 5 --rank 5 --savefig-name fig2                       \
        --add-tracks 11261 --rm-auto-tracks 11640 11722               \
	--panels-iel 1 1 1 1 0 0 --qv-days-offset 2 ${sav} -c ${cfg}  \
	--which-ssmi-dataset nt

elif [[ "${vri}" == "2" ]]
then
    # Case study 2: 6-15 July 2004
    # ----------------------------
    python -W ignore ./scripts/manuscript_figures/_case_study.py  \
        --region 4 --rank 38 --savefig-name fig3                  \
        --panels-pcm daidt dvidtd dvidtt ssmi div_strair qlw      \
        --qv-plot-mean --qv-adjust-factor 0.5 ${sav} -c ${cfg}

elif [[ "${vri}" == "S3" ]]
then
    # Case study 2 Supplemental material Fig. S3 (additional thermo. terms)
    # ---------------------------------------------------------------------

    # Pass in this title for the metadata (usually it is generated
    # automatically in the python script):
    title="Thermodynamic variables during case study: Kara sector,"
    title="${title} VRILE ID 40, rank 38 of 79, 06-15 Jul 2004"

    python -W ignore ./scripts/manuscript_figures/_case_study.py              \
        --region 4 --rank 38 --savefig-name figS3 --savefig-title "${title}"  \
        --panels-pcm meltb meltt meltl t2 qsw qnet                            \
	--panels-qv "-" "-" "-" "-" "-" "-"                                   \
	--panels-iel 1 1 1 1 1 1 ${sav} -c ${cfg}

elif [[ "${vri}" == "3" ]]
then
    # Case study 3: 14-29 August 2022
    # -------------------------------
    python -W ignore ./scripts/manuscript_figures/_case_study.py     \
        --dt-start 2022 8 14 --dt-end 2022 8 29 --savefig-name fig4  \
        --panels-pcm daidt dvidtd dvidtt ssmi div_strair qlw         \
        --add-tracks 9628 9198 --qv-days-offset 0 ${sav} -c ${cfg}

else
    echo "Error: VRILE case study -v 1  --> 22-30 Sep 2018"
    echo "                        -v 2  --> 06-15 Jul 2004"
    echo "                        -v 3  --> 14-29 Aug 2022"
    echo "                        -v S2 --> Supplemental figure for -v 2"
    exit 1
fi

exit 0

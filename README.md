# asc_wp7_vriles
**Data analysis code for the Arctic Summertime Cyclones (ASC) project, work package 7 on very rapid sea ice loss events (VRILEs).**

This work involved running a developmental version of the CICE sea ice model forced by atmospheric reanalysis. This repository contains data processing and analysis code accompanying a manuscript currently under consideration. It does not contain any code for the model itself but what was used is already described in the literature.[^b22]

See the [project website](https://research.reading.ac.uk/arctic-summertime-cyclones/ "Arctic Summertime Cyclones project homepage") for more information about this project.


## Dependencies
The python code should be OS-independent but there are some Bash scripts and it has only been tested on a UNIX system.

### Non-standard python packages (version tested with):
* [matplotlib](https://matplotlib.org/stable/ "matplotlib homepage") (3.8.1)
* [netCDF4](https://pypi.org/project/netCDF4/ "PyPI: netCDF4") (1.6.5)
* [numpy](https://numpy.org/doc/stable/index.html "NumPy: homepage") (1.26.0)
* [tabulate](https://pypi.org/project/tabulate/ "PyPI: tabulate") (0.9.0)

### External, custom python code
* [posterproxy](https://github.com/jraylmer/posterproxy "GitHub: posterproxy v1.0.0") (version 1.0.0; only needed for the figure generation scripts that involve map plots)

## Setup
1. Clone the repository
2. Setup a local 'configuration':
    1. In directory `cfgs`, create one more more copies of `cfg_default.ini` with a name of your choice but it must start with `cfg_` and end with `.ini` (e.g., `cfg_local.ini`)
    2. This is a configuration file parsed by the standard python library `configparser`. There is more information about how it works in the comment header of `cfg_default.ini`. Edit the various paths to the data on your system.*
4. Optional: set environment variable `ASC_WP7_VRILES_CONFIG` to the name of your configuration (the part between `cfg_` and `.ini`; e.g., `export ASC_WP7_VRILES_CONFIG='local'`). If this is _not_ set, it is necessary to pass the flag `-c local` (in this example) to any python scripts. Setting the environment variable enables this to be bypassed and the correct configuration file is read internally (see `src/io/config.py`)

*_Note: data that goes with this data analysis code is not currently available but will be made available in due course, upon which time a user should be able to reproduce everything starting with the CICE model outputs_

## Order of scripts
There are number of data processing scripts and the following is a record of the order in which scripts were applied, for future reference. Assuming the raw model output, atmospheric reanalysis raw data (used to generate forcing), and passive microwave SSM/I data are available, the analysis proceeds as follows (each path is relative to the `scripts` directory):
1. Prepare atmospheric forcing data
    1. `process_atmo_data/interp_to_cice_grid.sh` &#x2192; generates atmospheric forcing from raw reanalysis data
    2. `process_atmo_data/daily_monthly_averages.py` &#x2192; outputs needed for 1.iii, 3.iv
    3. `process_atmo_data/detrended_2d.py` &#x2192; outputs needed for 3.iv

3. Prepare CICE diagnostics and calculate/classify CICE VRILEs
   1. `generate_region_masks.py`
   2. `process_cice_data/area_extent_volume.py` &#x2192; outputs needed for 3.iv
   3. `process_cice_data/div_u_curl_strair.py` &#x2192; outputs needed for 3.iii, 3.iv
   4. `process_cice_data/detrended_2d.py` &#x2192; outputs needed for 3.iv
   5. `vriles/vriles_cice.py` &#x2192; outputs modified in 4.iii, 5.ii

5. Prepare SSMI data on the CICE grid and caluate SSMI VRILEs
   1. `process_ssmi_data/aice_on_cice_grid.sh` &#x2192; outputs needed for 4.ii, 4.iii
   2. `process_ssmi_data/siextent_on_cice_grid.py` &#x2192; outputs needed for 4.iii, 5.ii
   3. `vriles/vriles_ssmi.py` &#x2192; also updates output of 3.iv

6. Prepare cyclone track data and update VRILE data with matches to tracks
   1. `tracks/filter_tracks.py` &#x2192; outputs needed for 5.ii
   2. `tracks/match_tracks_to_vriles.py` &#x2192; updates outputs of 3.iv, 4.iii

7. Generation of manuscript figures (`scripts/manuscript_figures`)


[^b22]: Bateson, A. W., D. L. Feltham, D. Schr&#246;der, Y. Wang, B. Hwang, J. K. Ridley, and Y. Aksenov, 2022: Sea ice floe size: its impact on pan-Arctic and local ice mass and required model complexity, _Cryosphere_, **16**, 2565&#8211;2593, doi:[10.5194/tc-16-2565-2022](https://doi.org/10.5194/tc-16-2565-2022)

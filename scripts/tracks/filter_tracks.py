"""Runs the vorticity track filtering code and saves data in a more readiliy-
usable format for the filtered tracks only. This script can also be used to
calculate a percentile-based vorticity threshold to be used for the filtering
in the first place (the value of which is hard-coded in the tracks.py module,
which was originally calculated using this script with flag
--get-vor-threshold).
"""

import numpy as np

from src import script_tools
from src.io import cache, config as cfg
from src.data import tracks


def main():

    prsr = script_tools.argument_parser(
        usage="Filter tracks based on vorticity threshold")

    script_tools.add_track_filter_cmd_args(prsr)
    prsr.add_argument("-y", "--year-range", type=int, nargs=2,
                      default=[1979, 2023])

    # Add extra options so that this script can also be used to calculate the
    # percentile-based threshold for filtering, before being run again with
    # that threshold to get the filtered tracks (the threshold is now pre-
    # stored in module tracks.py after being calculated using this script)
    #
    prsr.add_argument("--get-vor-threshold", action="store_true")
    prsr.add_argument("--vor-threshold-percentile", type=float, default=95.)
    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    if cmd.get_vor_threshold:
        cmd.track_lat_min = 70.
        cmd.track_vor_min = 0.
        print("Overriding track filter options to:")
        print(f"    --track-lat-min {cmd.track_lat_min}")
        print(f"    --track-vor-min {cmd.track_vor_min}")
        print(f"Determining {cmd.vor_threshold_percentile:.1f} percentile "
              + "along-track max. vorticity")

    filter_kw = script_tools.get_track_filter_options(cmd, header=False,
                                                      footer=True)

    years = np.arange(cmd.year_range[0], cmd.year_range[1] + 1, 1)

    filtered_tracks = tracks.get_filtered_tracks_multiyear(years, **filter_kw)

    # 'filtered_tracks' returned above is actually eight returned values in
    # a tuple, corresponding to titles listed below. Save these 'headers' with
    # the actual data below (no need to get individual return values here):
    headers = ["TRACKS_PER_YEAR", "TRACK_IDS", "DATETIMES", "LONGITUDES",
               "LATITUDES", "VORTICITIES", "VORTICITY_MAXIMA", "SECTOR_FLAG"]

    if cmd.get_vor_threshold:

        vor_thr = np.percentile(filtered_tracks[6], cmd.vor_threshold_percentile)
        txt_save = f"{cmd.vor_threshold_percentile:.1f} percentile vorticity = "
        txt_save += f"{vor_thr:.6E} s-1"

        print(txt_save)
        cache.write_txt(txt_save, "vorticity_threshold.txt",
                        directory=cfg.data_path["tables"])
        # Don't save tracks if getting threshold

    else:
        cache.save([headers] + list(filtered_tracks),
                   f"tracks_filtered_{years[0]}-{years[-1]}.pkl")


if __name__ == "__main__":
    main()


"""Identify all vorticity tracks entering the Arctic, defined as north of 70 N,
regardless of intensity, and save the data and a summary table, for one year.
"""

from tabulate import tabulate

from src import script_tools
from src.io import cache, config as cfg
from src.data import tracks


def main():

    prsr = script_tools.argument_parser(
        usage="Tabulate unfiltered tracks for one year")

    script_tools.add_track_filter_cmd_args(prsr)
    prsr.add_argument("-y", "--year", type=int, default=2022)

    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    # Keep original vorticity threshold for table to note if it is exceeded
    # (i.e., is in the filtered set or not):
    vor_threshold = cmd.track_vor_min * 1.e-5

    filter_kw = script_tools.get_track_filter_options(cmd, header=False,
                                                      footer=True)

    cmd.track_lat_min = 70.
    cmd.track_vor_min = 0.

    filter_kw["vor_min"] = cmd.track_vor_min
    filter_kw["lat_min"] = cmd.track_lat_min

    print("Now setting track filter options (for sake of loading all tracks):")
    print(f"    --track_lat_min {cmd.track_lat_min}")
    print(f"    --track_vor_min {cmd.track_vor_min}\n")

    # Use all region info defined in config module, but not those set by
    # config files, which may be a subset, e.g., for VRILEs. This includes
    # pan-Arctic so is a list of 'regions' rather than 'sectors':
    region_labels   = cfg.all_reg_long_names
    region_lon_bnds = cfg.all_reg_sector_lon_bnds
    n_regions       = len(region_labels)

    # Get all tracks for specified year:
    all_tracks = tracks.get_filtered_tracks(cmd.year, sort_by_date=True,
                                            sector_lon_bnds=region_lon_bnds,
                                            **filter_kw)

    # 'all_tracks' returned above is actually seven returned values
    # containined a tuple, each element corresponding to:
    headers = ["TRACK_IDS", "DATETIMES", "LONGITUDES", "LATITUDES",
               "VORTICITIES",  "VORTICITY_MAXIMA", "SECTOR_FLAG"]

    # Save the actual data:
    cache.save([headers] + list(all_tracks), f"tracks_all_{cmd.year}.pkl")

    # Construct tabulated version for saving to text file:
    n_tracks = len(all_tracks[0])

    # The following information will be tabulated for each track:
    tab_headers = ["START", "END", "TRACK ID", "MAX. VORT.\n(1e-5/s)", "> THR?",
                   " ".join([f"{i}" for i in range(n_regions)])]

    # Argument for tabulate function:
    floatfmt = ["", "", "", ".2f", "", ""]

    tab_rows   = []
    n_filtered = 0  # below, count number of tracks that would be filtered

    for j in range(n_tracks):
        row_j = [all_tracks[1][j][0].strftime("%d-%b"),
                 all_tracks[1][j][-1].strftime("%d-%b"),
                 all_tracks[0][j],
                 all_tracks[5][j] * 1.e5,
                 u"\u2713" if all_tracks[5][j] > vor_threshold else " ",
                 " ".join([u"\u2713" if x.any() else "-" for x in all_tracks[6][j]])]

        if all_tracks[5][j] >= vor_threshold:
            n_filtered += 1

        tab_rows.append(row_j)

    # Tabulate and save:
    txt =  f"Total number of tracks in {cmd.year:04}     : {n_tracks:>4}\n"
    txt += f"of which exceed vorticity threshold: {n_filtered:>4} "
    txt += f"({100.*n_filtered/n_tracks:.2f}%)\n\n"
    txt += f"Vorticity threshold = {1.e5*vor_threshold:.5f} \u00d7 1e-5/s\n\n"
    txt += "Region/sector indices:\n"
    txt += "\n".join([f"  {i}: {region_labels[i]}" for i in range(n_regions)])
    txt += "\n\n" + tabulate(tab_rows, headers=tab_headers, floatfmt=floatfmt)
    txt += "\n"

    cache.write_txt(txt, f"tracks_all_{cmd.year:04}.txt",
                    directory=cfg.data_path["tables"])


if __name__ == "__main__":
    main()


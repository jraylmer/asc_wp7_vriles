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

    cmd.track_lat_min = 70.
    cmd.track_vor_min = 0.

    filter_kw = script_tools.get_track_filter_options(cmd, header=False,
                                                      footer=True)

    print("Overriding track filter options to:")
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
    tab_headers = ["START", "END", "TRACK ID",
                   " ".join([f"{i}" for i in range(n_regions)])]

    tab_rows = [[all_tracks[1][j][0].strftime("%d-%b"),
                 all_tracks[1][j][-1].strftime("%d-%b"),
                 all_tracks[0][j],
                 " ".join([u"\u2713" if x.any() else "-" for x in all_tracks[6][j]])]
                for j in range(n_tracks)]

    # Tabulate and save:
    txt = f"Total number of {cmd.year} tracks: {n_tracks}\n\n"
    txt += "Region/sector indices:\n"
    txt += "\n".join([f"  {i}: {region_labels[i]}" for i in range(n_regions)])
    txt += "\n\n" + tabulate(tab_rows, headers=tab_headers) + "\n"

    cache.write_txt(txt, f"tracks_all_{cmd.year:04}.txt",
                    directory=cfg.data_path["tables"])


if __name__ == "__main__":
    main()


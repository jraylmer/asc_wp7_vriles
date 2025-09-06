"""Run the VRILE-vorticity track matching code and caches the data directly as
returned from function match_tracks_to_vriles() from the track_diagnostics.py
module.
"""

from src import script_tools
from src.io import cache, config as cfg
from src.data import tracks
from src.diagnostics import track_diagnostics


def main():

    prsr = script_tools.argument_parser(
        usage="Find occurences of tracks matching VRILEs")

    script_tools.add_vrile_cmd_args(prsr, ssmi=True)
    script_tools.add_track_filter_cmd_args(prsr)
    script_tools.add_track_vrile_matching_cmd_args(prsr)

    cmd = prsr.parse_args()

    cfg.set_config(*cmd.config)

    filter_kw = script_tools.get_track_filter_options(cmd, header=False,
                                                      footer=False)

    script_tools.get_track_vrile_matching_options(cmd, header=False,
                                                  footer=True)

    allowed_sectors = tracks.allowed_sectors(n_nei=cmd.track_n_sector_neighbours)

    vrds = "" if cmd.match_unjoined_vriles else "_joined"
    yrng = f"{cmd.year_range[0]:04}-{cmd.year_range[1]:04}"

    # Load cached VRILE and filtered track data (run relevant scripts first):
    if cmd.match_ssmi:
        vriles_fname = f"vriles_ssmi-{cmd.ssmi_dataset}_{yrng}.pkl"
    else:
        vriles_fname = f"vriles_cice_{yrng}.pkl"

    vriles = cache.load(vriles_fname)

    track_data = cache.load(f"tracks_filtered_{yrng}.pkl")

    # Extract required track data (datetimes and sector for each coordinate)
    # Assumes headers set as in the script generating the filtered track data:
    track_dts = track_data[1+track_data[0].index("DATETIMES")]
    track_sec = track_data[1+track_data[0].index("SECTOR_FLAG")]

    # Run the track/VRILE matching code (this returns the track-array indices
    # for each VRILE and vice-versa, respectively):
    v_tr_indices, tr_v_indices = track_diagnostics.match_tracks_to_vriles(
        [vr[f"date_bnds_vriles{vrds}"] for vr in vriles], track_dts, track_sec,
        allowed_sectors=allowed_sectors,
        track_max_lead_days=cmd.match_n_day_lag[0],
        track_max_lag_days=cmd.match_n_day_lag[1])

    cache_match_name = (f"tracks_matches_to_vriles_"
                        + (("ssmi-" + cmd.ssmi_dataset) if cmd.match_ssmi else "cice")
                        + f"_{yrng}.pkl")

    cache.save([v_tr_indices, tr_v_indices], cache_match_name)


if __name__ == "__main__":
    main()


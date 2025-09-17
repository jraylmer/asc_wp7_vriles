"""Run the VRILE-vorticity track matching code and caches the data directly as
returned from function match_tracks_to_vriles() from the track_diagnostics.py
module.
"""

from pathlib import Path

from src import script_tools
from src.io import cache, config as cfg, sumtabtxt
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

    # This is needed for updating/writing the summary table only as this script
    # assumes the saved VRILEs have been calculated using correct options:
    id_vriles_kw, join_vriles_kw, dt_min, dt_max = \
        script_tools.get_id_vriles_options(cmd, footer=False)

    filter_kw = script_tools.get_track_filter_options(cmd, header=False,
                                                      footer=False)

    script_tools.get_track_vrile_matching_options(cmd, header=False,
                                                  footer=True)

    allowed_sectors = tracks.allowed_sectors(n_nei=cmd.track_n_sector_neighbours)

    vrds = "" if cmd.match_unjoined_vriles else "_joined"
    yrng = f"{dt_min.year:04}-{dt_max.year:04}"

    # Load cached VRILE and filtered track data (run relevant scripts first):
    # Also determine new header for updating track data at the end:
    if cmd.match_ssmi:
        vriles_fname = f"vriles_ssmi-{cmd.ssmi_dataset}_{yrng}.pkl"
        track_new_header = f"vriles_ssmi-{cmd.ssmi_dataset}_indices"
    else:
        vriles_fname = f"vriles_cice_{yrng}.pkl"
        track_new_header = "vriles_cice_indices"

    tracks_fname = f"tracks_filtered_{yrng}.pkl"

    vrile_data = cache.load(vriles_fname)
    track_data = cache.load(tracks_fname)

    # Extract required track data (IDs, datetimes and sector for each coordinate)
    # Assumes headers set as in the script generating the filtered track data:
    track_ids = track_data[1+track_data[0].index("TRACK_IDS")]
    track_dts = track_data[1+track_data[0].index("DATETIMES")]
    track_sec = track_data[1+track_data[0].index("SECTOR_FLAG")]

    # Run the track/VRILE matching code (this returns the track-array indices
    # for each VRILE and vice-versa, respectively):
    v_tr_indices, tr_v_indices = track_diagnostics.match_tracks_to_vriles(
        [vr[f"date_bnds_vriles{vrds}"] for vr in vrile_data], track_dts,
        track_sec, allowed_sectors=allowed_sectors,
        track_max_lead_days=cmd.match_n_day_lag[0],
        track_max_lag_days=cmd.match_n_day_lag[1])

    # Add new data to the VRILE results dictionaries:
    for r in range(len(vrile_data)):
        # Length nv list of array (n_trk_match,) of int, where nv is the
        # number of VRILEs for this region (results dictionary) and n_trk_match
        # is the number of matching track indices (different per VRILE).
        # The values are indices of the full, filtered track array:
        vrile_data[r]["track_indices"] = v_tr_indices[r]

        # Mainly for output tables, helpful to also have the actual track IDs
        # (similar format as above, but list of list of int):
        vrile_data[r]["track_ids"] = [[int(track_ids[y]) for y in x]
                                      for x in v_tr_indices[r]]

    # The new data (tr_v_indices; returned above) is a length n_track list
    # of length n_v_match list of length 2 list of region and VRILE indices,
    # where n_v_match is the number of matching VRILEs to a given track
    # (different per track), such that:
    #
    #     tr_v_indices[k] = [ [r1, v1], [r2, v2], ...]
    #
    # gives the region and VRILE array indices [r, v] of each VRILE matching
    # track index k.
    #
    # 'track_data' is a list of various data, first element being a list of
    # headers describing the remaining elements. Either add a new header and
    # append the new data to the end or overwrite if the 'new header' exists:
    #
    if track_new_header in track_data[0]:
        # Previous data exists; overwrite
        track_data[1+track_data[0].index(track_new_header)] = tr_v_indices
    else:
        # New data; append to the end:
        track_data[0].append(track_new_header)
        track_data.append(tr_v_indices)

    # Overwrite saved data, now updated:
    cache.save(vrile_data, vriles_fname)
    cache.save(track_data, tracks_fname)

    # Save VRILE summary tables (possibly overwriting earlier-saved tables,
    # but now VRILE dicationaries will have the track indicies so these are
    # included in the table).
    additional_metadata = {}

    if cmd.match_ssmi:
        if cmd.ssmi_dataset == "nt":
            additional_metadata["SSM/I dataset"] = "NASA Team"
            save_subdir = f"vriles_ssmi-nt_{dt_min.year}-{dt_max.year}"
        else:
            additional_metadata["SSM/I dataset"] = "Bootstrap"
            save_subdir = f"vriles_ssmi-bt_{dt_min.year}-{dt_max.year}"
    else:
        save_subdir = f"vriles_cice_{dt_min.year}-{dt_max.year}"

    additional_metadata["Time range"]  = (f"{dt_min.strftime('%d %b %Y')} to "
                                          + f"{dt_max.strftime('%d %b %Y')}")
    additional_metadata["Description"] = cfg.title

    sumtabtxt.save_tables(vrile_data, id_vriles_kw,
            which=[True, True, False, False],
            vresults_labels=cfg.reg_labels_short,
            additional_metadata=additional_metadata, verbose=True,
            save_dir=Path(cfg.data_path["tables"], save_subdir))


if __name__ == "__main__":
    main()


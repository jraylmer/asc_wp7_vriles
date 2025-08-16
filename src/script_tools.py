"""Provide tools for preparing command-line argument parsing using argparse
with some functions to print options to the console when running scripts.
"""

from argparse import ArgumentParser
import calendar
from datetime import datetime as dt

from .data import tracks


def _print_header(title, header="-"*64):
    """Print header for command-line argument print."""
    print(f"\n{header}\n{title}\n{header}")


def _print_option(opt_name, opt_val, opt_gap=30):
    """Print an option opt_name and its value opt_val."""
    print(opt_name + " "*max(0, (opt_gap - len(opt_name))) + f": {opt_val}")


def _print_footer():
    """Print footer for command-line argument print."""
    print("")


def argument_parser(**kw):
    """Return an argparse.ArgumentParser() instance with configuration
    argument (-c --config [<cfg_1> [<cfg_2> ...]]) option already added.
    
    Keyword arguments are passed to argparse.ArgumentParser().
    
    """

    prsr = ArgumentParser(**kw)

    # The configuration is generally always needed:
    prsr.add_argument("-c", "--config", type=str, nargs="*",
                      default=["default"], help="Configuration file(s)")
    
    return prsr


def add_vrile_cmd_args(prsr, obs=False):
    """Add command line arguments/options for VRILE identification to an
    argparse.ArgumentParser() instance. Optional parameter 'obs', bool,
    determines whether to add observations-specific options (i.e., if
    script is being called to run VRILE identification on SSM/I data).
    """

    prsr.add_argument("--metric", type=str, default="sea_ice_extent",
                      choices=["sea_ice_extent", "sea_ice_area"])

    prsr.add_argument("-y", "--year-range", type=int, nargs=2,
                      default=(1979, 2023))

    prsr.add_argument("--n-days"          , type=int, default=5)
    prsr.add_argument("--n-moving-average", type=int, default=31)

    prsr.add_argument("--threshold", type=float, nargs=2, default=(0., 5.))
    prsr.add_argument("--threshold-type", type=str, default="percent",
                      choices=["percent", "value"])

    prsr.add_argument("--months-allowed", type=int, nargs="*",
                      default=[5, 6, 7, 8, 9])

    prsr.add_argument("--data-min", type=float, default=None)
    prsr.add_argument("--data-max", type=float, default=None)

    prsr.add_argument("--criteria-order", type=str, nargs=3,
                      default=("data_range", "months", "threshold"),
                      choices=("data_range", "months", "threshold"))

    prsr.add_argument("--join-vriles-max-gap", type=int, default=1)

    if obs:
        prsr.add_argument("--obs", type=str, default="nt",
                          choices=["nt", "bt", "amsr"])
        prsr.add_argument("--obs-filter-n-days", type=int, default=5,
            help="Low pass filter applied to OBS only")


def add_vrile_classification_cmd_args(prsr):
    """Add command-line arguments/options for classifying VRILEs to an
    argparse.ArgumentParser() instance.
    """

    prsr.add_argument("--class-no-detrend", action="store_true",
                      help="Use raw, not-detrended, tendency diagnostics")

    prsr.add_argument("--class-metric", type=str, default="volume",
                      choices=["volume", "concentration"],
                      help="Use volume (default) or concentration tendencies")


def add_track_filter_cmd_args(prsr):
    """Add command-line arguments/options for vorticity track filtering to an
    argparse.ArgumentParser() instance.
    """

    prsr.add_argument("--track-n-check", type=int,
                      default=tracks.default_track["filter_n_check"],
                      help="Consecutive points to apply filter criteria")

    prsr.add_argument("--track-lat-min", type=float,
                      default=tracks.default_track["filter_lat_min"],
                      help="Minimum latitude threshold")

    prsr.add_argument("--track-vor-min", type=float,
                      default=tracks.default_track["filter_vor_min"],
                      help="Minimum vorticity threshold in 10^-5 s^-1")

    prsr.add_argument("--track-n-sector-neighbours", type=int, default=1,
                      choices=[0, 1],
                      help="Number of nearest sector neighbours for track to "
                           + "count as 'in' sector")


def add_track_vrile_matching_cmd_args(prsr):
    """Add command-line arguments/options for track/VRILE matching to an
    argparse.ArgumentParser() instance.
    """
    prsr.add_argument("--match-n-day-lag", type=int, nargs=2, default=(0, 0))
    prsr.add_argument("--match-unjoined-vriles", action="store_true")
    prsr.add_argument("--match-obs", action="store_true")


def get_id_vriles_options(cmd, header=True, footer=True):
    """Construct dictionaries of keyword arguments to pass to function
    src.diagnostics.vriles.identify() from command-line arguments and
    print them to the console.


    Parameters
    ----------
    cmd : Namespace of command-line arguments
        From argparse.ArgumentParser().parse_args()


    Optional parameters
    -------------------
    header, footer : bool, default = True
        Print the header and/or footer respectively.


    Returns
    -------
    id_vriles_kw : dict
        Keyword arguments to pass to src.diagnostics.vriles.identify().

    join_vriles_kw : dict
        Keyword arguments to pass to src.diagnostics.vriles.identify().

    dt_min, dt_max : datetime.datetime
        Start and end date to apply VRILE identification/criteria.

    """

    id_vriles_kw   = {"nt_delta"      : cmd.n_days,
                      "nt_delta_units": "indices",
                      "threshold"     : cmd.threshold,
                      "threshold_type": cmd.threshold_type,
                      "months_allowed": cmd.months_allowed,
                      "data_min"      : cmd.data_min,
                      "data_max"      : cmd.data_max,
                      "criteria_order": cmd.criteria_order}

    join_vriles_kw = {"max_gap"       : cmd.join_vriles_max_gap}

    dt_min = dt(cmd.year_range[0] ,  1,  1, 12)
    dt_max = dt(cmd.year_range[-1], 12, 31, 12)

    if header:
        _print_header("VRILE ID options")

    if "obs" in [x[0] for x in cmd._get_kwargs()]:
        _print_option("Observations", cmd.obs)

    _print_option("Time range",       f"{dt_min.strftime('%d %b %Y')}"
                                + f" to {dt_max.strftime('%d %b %Y')}")

    _print_option("Mov. avg. filter width", f"{cmd.n_moving_average} days")

    if "obs" in [x[0] for x in cmd._get_kwargs()]:
        _print_option("Obs. additional filter", f"{cmd.obs_filter_n_days} days")

    _print_option("Timescale", f"{cmd.n_days} day changes")
    _print_option("Metric", cmd.metric)

    _print_option("Threshold", f"{cmd.threshold}"
                  + (" %" if cmd.threshold_type == "percent" else " * 10^6 km^2"))

    _print_option("Months allowed for VRILEs",
        ("all" if len(cmd.monthsallowed) >= 12
               else ", ".join([calendar.month_abbr[j] for j in cmd.monthsallowed])))

    _print_option("Range allowed for VRILEs",
                    ("-inf" if cmd.data_min is None else cmd.data_min) + " to "
                  + ("+inf" if cmd.data_max is None else cmd.data_max) + " * 10^6 km^2")

    _print_option("Criteria order", " > ".join(cmd.criteria_order))

    _print_option("Joined VRILEs max. gap", f"{cmd.join_vriles_max_gap}")

    if "class_metric" in [x[0] for x in cmd._get_kwargs()]:
        _print_option("Classification metric", f"{cmd.class_metric}, "
                      + ("Raw (not detrended)" if cmd.class_no_detrend
                                               else "detrended"))

    if footer:
        _print_footer()

    return id_vriles_kw, join_vriles_kw, dt_min, dt_max


def get_track_filter_options(cmd, header=False, footer=False):
    """Construct dictionaries of keyword arguments to pass to function
    src.data.tracks filtering functions from command-line arguments and print
    them to the console.


    Parameters
    ----------
    cmd : Namespace of command-line arguments
        From argparse.ArgumentParser().parse_args()


    Optional parameters
    -------------------
    header, footer : bool, default = False
        Print the header and/or footer.


    Returns
    -------
    filter_tracks__kw : dict
        Keyword arguments to pass to src.data.tracks.get_filtered_tracks().

   """

    filter_tracks_kw = {"vor_min": cmd.track_vor_min,
                        "lat_min": cmd.track_lat_min,
                        "n_check": cmd.track_n_check}

    if header:
        _print_header("Track filtering options")

    _print_option("Minimum vorticity", f"{cmd.track_vor_min:.2f} * 10^-5 s^-1")
    _print_option("Minimum latitude", f"{cmd.track_lat_min:.0f} degrees_north")
    _print_option("Criteria min. consec. coords.", f"{cmd.track_n_check}")

    if footer:
        _print_footer()

    return filter_tracks_kw


def get_track_vrile_matching_options(cmd, header=True, footer=True):
    """Print options for matching VRILEs to tracks from command-line arguments
    to the console.


    Parameters
    ----------
    cmd : Namespace of command-line arguments
        From argparse.ArgumentParser().parse_args()


    Optional parameters
    -------------------
    header, footer : bool, default = True
        Print the header and/or footer.

    """

    if header:
        _print_header("Track/VRILE matching options")

    _print_option("Matching", ("Obs." if cmd.match_obs else "CICE") + " VRILEs")

    _print_option("Track allowed lead time", f"{cmd.match_n_day_lag[0]} day"
                  + ("s" if cmd.match_n_day_lag[0] != 1 else ""))

    _print_option("VRILE allowed lead time", f"{cmd.match_n_day_lag[1]} day"
                  + ("s" if cmd.matchndaylag[1] != 1 else ""))

    _print_option("Use joined VRILEs", "No" if cmd.match_unjoined_vriles else "Yes")

    if footer:
        _print_footer()


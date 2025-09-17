"""Provides functionality to save SUMmary TABle TeXT files, primarily for lists
of VRILEs, their matches between regions, and between different datasets.
"""

import calendar
from datetime import datetime as dt
import numpy as np
from pathlib import Path

from tabulate import tabulate

from . import config as cfg


def _generated_txt(realtime_fmt="%H:%M UTC %d %b %Y"):
    """String for adding datetime stamp to summary table metadata."""
    return f"Generated {dt.utcnow().strftime(realtime_fmt)} by {cfg.author}"


def _detrend_type_txt(x):
    """String for adding detrend method to summary table metadata."""
    return "Seasonal trend" if "seasonal" in x else "Trend"


def _months_allowed_txt(x):
    """String for adding month range used to summary table metadata."""
    return ("all" if len(x) == 12
                  else ", ".join([calendar.month_abbr[y] for y in x]))


def _criteria_order_txt(x):
    """String for adding method criteria order to summary table metadata."""
    return " > ".join(x)


def _txt_file_header(txtfile, id_vriles_kw, detrend_type="seasonal_periodic",
                     n_ma=5, title="VRILEs", header_length=80,
                     additional_metadata={}):
    """Add common header information for table text files. Uses the keyword
    arguments passed to the function src.diagnostics.vriles.identify() to
    generate description of the data generation method.

    Parameters
    ----------
    txtfile : File object
        The opened text file being written to.

    id_vriles_kw : dict
        The keyword arguments passed to the function
        src.diagnostics.vriles.identify(). 
    
    detrend_type : str
        String used to identify the detrending method used on the sea ice
        extent data prior to identifying VRILEs.

    n_ma : int
        Size in days of the moving average filter used in the detrending.

    
    Optional parameters
    -------------------
    title : str, default = 'VRILEs'
        Title of the dataset (appears at the very top of the file).

    header_length : int, default = 80
        Character width of the output file.

    additional_metadata : dict {key: str}, default = {}
        Additional metadata values to include in the header information.

    """

    txtfile.write(f"\n{'-'*header_length}\n{title}\n{'-'*header_length}\n\n")

    txtfile.write("Data processing:\n")
    txtfile.write(f"    {_detrend_type_txt(detrend_type)} "
                  + f"defined on {n_ma} day moving average"
                  + "\n\n")

    txtfile.write("Identification criteria:\n")

    txtfile.write("    n days         = ")
    txtfile.write(f"{id_vriles_kw['nt_delta']}" + "\n")

    txtfile.write("    threshold      = ")

    if id_vriles_kw["threshold_type"] == "percent":
        txtfile.write(f"{id_vriles_kw['threshold'][0]:.0f}-")
        txtfile.write(f"{id_vriles_kw['threshold'][1]:.0f} percentiles\n")
    else:
        txtfile.write(f"{id_vriles_kw['threshold'][0]:.2f} to ")
        txtfile.write(f"{id_vriles_kw['threshold'][1]:.2f} *10^6 km^2 (fixed)\n")

    txtfile.write("    months         = " +
                  _months_allowed_txt(id_vriles_kw["months_allowed"]) + "\n")

    txtfile.write("    dSIE range     = ")

    if id_vriles_kw["data_min"] is None:
        txtfile.write("-inf to ")
    else:
        txtfile.write(f"{id_vriles_kw[data_min]:.2f} to ")

    if id_vriles_kw["data_max"] is None:
        txtfile.write("+inf\n")
    else:
        txtfile.write(f"{id_vriles_kw[data_max]:.2f}\n")

    txtfile.write("    criteria order = "
                  + _criteria_order_txt(id_vriles_kw["criteria_order"]) + "\n")

    if additional_metadata:
        txtfile.write("\nAdditional metadata:\n")
        am_keys = list(additional_metadata.keys())
        maxlen = max([len(k) for k in am_keys])
        for k in am_keys:
            txtfile.write(f"    {k:<{maxlen}}: {additional_metadata[k]}\n")


def _txt_file_footer(txtfile):
    """Write footer into text file."""
    txtfile.write("\n")


def _fmt_date(x):
    """Common formatter for VRILE date data (input: datetime.datetime)."""
    return x.strftime("%Y-%m-%d")


def _fmt_int(x):
    """Common formatter for VRILE integer data (input: integer)."""
    return f"{x}"


def _fmt_float(x):
    """Common formatter for VRILE float data (input: float)."""
    return f"{x:.2f}"


def _fmt_list(x):
    """Common formatter for VRILE list/iterable data (input: iterable)."""
    return " ".join([str(j) for j in x])


def _fmt_none(x):
    """Common formatter for arbitrary data type (input: anything)."""
    return f"{repr(x)}"


# Dicitonary of dictionaries of vrile results key properties for table outputs
# =========================================================================== #
# Each (well, most) column in the table outputs get data from something in the
# VRILE results dictionar(ies) passed to the functions generating them. Define
# various table properties ('props') for the VRILE results keys ('vrk') to be
# accessed as required in the table generating functions:
#
#     header  : table column header (defaults to key itself)
#     fmt_func: function to format values into a string for the table
#               (defaults to _fmt_none)
#     fmt_tab : formatter used by tabulate function (defaults to '')
#     scale   : factor by which to multiply data before formatting
#               (if not present, do not scale)
#
# Defaults set below if not defined here. Multiple headers can be defined in a
# list for multi-dimensional vresults data (e.g., datetime bounds which are 2D
# arrays: first header 'START' corresponds to index 0 and second header 'END'
# to index 1 for the second axis of that data).
#
_vrk_props = {
    "vriles_joined_rates_rank": {"header"  : "RANK",
                                 "fmt_func": _fmt_int,
                                  "fmt_tab" : "03d"
                                },
    "date_bnds_vriles_joined" : {"header"  : ["START", "END"],
                                 "fmt_func": _fmt_date
                                },
    "vriles_joined_n_days"    : {"header"  : "N DAYS",
                                 "fmt_func": _fmt_int,
                                 "fmt_tab" : "03d"
                                },
    "vriles_joined"           : {"header"  : u"\u0394" + u"SIE\n(10\u2076km\u00b2)",
                                 "fmt_func": _fmt_float,
                                 "fmt_tab" : ".2f"
                                },
    "vriles_joined_rates"     : {"header"  : u"dSIE/dt\n(10\u00b3km\u00b2/day)",
                                 "fmt_func": _fmt_float,
                                 "fmt_tab" : ".2f",
                                 "scale"   : 1.e3
                                },
    "vriles_joined_class"     : {"header"  : "CLASS",
                                 "fmt_func": _fmt_float,
                                 "fmt_tab" : ".2f"},
    "track_ids"               : {"header"  : "TRACK IDS\nASSOCIATED WITH VRILE",
                                 "fmt_func": _fmt_list}
}

# Add default values (except 'scale') and entry for number of headings:
for k in _vrk_props.keys():
    for subkey, default_value in zip(["header", "fmt_func", "fmt_tab"],
                                     [k       , _fmt_none , ""       ]):
        if subkey not in _vrk_props[k].keys():
            _vrk_props[k][subkey] = default_value

    if type(_vrk_props[k]["header"]) in [list]:
        _vrk_props[k]["n_headers"] = len(_vrk_props[k]["header"])
    else:
        _vrk_props[k]["n_headers"] = 1

# --------------------------------------------------------------------------- #


def save_vrile_table(vresults, filename, id_vriles_kw, additional_metadata={},
                     sort_by_rank=False):
    """Summarise VRILE data in a table and save to a text file.

    Parameters
    ----------
    vresults : dict
        A VRILE 'results dictionary' as returned by function identify() in
        module src.diagnostics.vriles. This is assumed to contain both the data
        and metadata required for the summary table.

    filename : str or pathlib.Path
        The file to save. Overwrites if it already exists, and assumes that the
        directory exists (otherwise, this fails).

    id_vriles_kw : dict
        The keyword arguments passed to function identify() in module
        src.diagnostics.vriles, assumed to contain all the options used to
        generate vresults.


    Optional parameters
    -------------------
    additional_metadata : dict {key: str}
        Any other metadata key-pairs to write in the header for the record.

    sort_by_rank : bool, default = False
        Whether to list VRILEs in the table in order of rank (if True),
        otherwise in the order they appear in the data itself (if False;
        usually this should mean data order).
    """

    # Specify the required keys of vresults and their order (columns), and the
    # order of VRILE indices (rows). Note that required keys may not be in
    # vresults (check and if so discard them, afterwards):
    if sort_by_rank:

        vrk_want = ["vriles_joined_rates_rank", "date_bnds_vriles_joined",
                    "vriles_joined_n_days", "vriles_joined_class",
                    "vriles_joined", "vriles_joined_rates", "track_ids"]

        j_vriles = np.argsort(vresults["vriles_joined_rates_rank"])

    else:

        vrk_want = ["date_bnds_vriles_joined", "vriles_joined_n_days",
                    "vriles_joined_rates_rank", "vriles_joined_class",
                    "vriles_joined", "vriles_joined_rates", "track_ids"]

        j_vriles = np.argsort(vresults["date_bnds_vriles_joined"][:,0])

    # Headers and tabulate argument 'floatfmt' for table output:
    headers  = []
    floatfmt = []
    vrk_use  = []  # equal to or subset of vrk_want (discard unavailable keys)
    for k in vrk_want:
        if k in vresults.keys():
            if _vrk_props[k]["n_headers"] > 1:
                for i in range(_vrk_props[k]["n_headers"]):
                    headers.append(_vrk_props[k]["header"][i])
                    floatfmt.append(_vrk_props[k]["fmt_tab"])
            else:
                headers.append(_vrk_props[k]["header"])
                floatfmt.append(_vrk_props[k]["fmt_tab"])
            vrk_use.append(k)

    # List of rows for table output:
    rows = []

    # Loop over joined VRILEs in order of indices specified in j_vriles:
    for j in j_vriles:
        row_j = []  # set up row for joined VRILE with array index j

        # Loop over required results dictionary keys:
        for k in vrk_use:
            if _vrk_props[k]["n_headers"] > 1:
                # Multiple 'sub'-headings / 2D array for key k
                for i in range(_vrk_props[k]["n_headers"]):
                    val = vresults[k][j,i]
                    if "scale" in _vrk_props[k].keys():
                        val *= _vrk_props[k]["scale"]

                    row_j.append(_vrk_props[k]["fmt_func"](val))

            else:
                # No 'sub'-headings / 1D array for key k
                val = vresults[k][j]
                if "scale" in _vrk_props[k].keys():
                    val *= _vrk_props[k]["scale"]

                row_j.append(_vrk_props[k]["fmt_func"](val))

        rows.append(row_j)  # append to main list of rows

    table = tabulate(rows, headers=headers, floatfmt=floatfmt)

    with open(filename, "w") as txtfile:
        
        _txt_file_header(txtfile, id_vriles_kw,
                         detrend_type=vresults["detrend_type"],
                         n_ma=vresults["moving_average_filter_n_days"],
                         title=vresults["title"],
                         additional_metadata=additional_metadata)

        txtfile.write(f"\nTable lists the {vresults['n_joined_vriles']} non-"
                      + "overlapping, joined events (hence "
                      + "the different "
                      + _vrk_props["vriles_joined_n_days"]["header"]
                      + ").\n")

        if sort_by_rank:
            txtfile.write("\nSorted in descending order of average rates, "
                          + "dSIE/dt (magnitudes).\n")
        else:
            txtfile.write("\nRanks are based on the magnitude of the average "
                          + "rates, dSIE/dt.\n")

        txtfile.write("\n" + _generated_txt() + "\n\n\n" + table + "\n")

        _txt_file_footer(txtfile)


def save_vrile_matches_to_txt(vresults_list, filename, id_vriles_kw,
                              region_labels=None, additional_metadata={},
                              sort_by_rank=False):
    """Save a table listing, for each pan-Arctic VRILE, which regions exhibit a
    VRILE at the same time (overlapping datetime bounds for each joined VRILE).

    TODO: generalise this for matching other regions to any given region?


    Parameters
    ----------
    vresults_list : length nV list of dict
        List of VRILE 'results dictionaries' as returned by function identify()
        in module src.diagnostics.vriles.

    filename : str or pathlib.Path
        The file to save. Overwrites if it already exists, and assumes that the
        directory exists (otherwise, this fails).

    id_vriles_kw : dict
        The keyword arguments passed to function identify() in module
        src.diagnostics.vriles, assumed to contain all the options used to
        generate vresults.


    Optional parameters
    -------------------
    region_labels : length nV list of str or None
        Region labels corresponding to each VRILE results dictionary in
        vresults_list. If None, get the default list set in the configuration
        (src.io.config), specifically, reg_labels_long.

    additional_metadata : dict {key: str}
        Any other metadata key-pairs to write in the header for the record.

    sort_by_rank : bool, default = False
        Whether to list VRILEs in the table in order of rank (if True),
        otherwise in the order they appear in the data itself (if False;
        usually this should mean data order).
    """

    if region_labels is None:
        region_labels = cfg.reg_labels_long

    matches = []

    for k in range(vresults_list[0]["n_joined_vriles"]):

        matches_k = []

        # Date bounds of current VRILE considered:
        db_k = vresults_list[0]["date_bnds_vriles_joined"][k,:]

        for r in range(1, len(vresults_list)):

            overlaps = np.array([(db_k[0] <= x) & (db_k[1] >= x)
                for x in vresults_list[r]["date_bnds_vriles_joined"][:,0]]).astype(bool)

            if any(overlaps):
                matches_k.append(f"{r}")

        matches.append(matches_k)

    # Table will essentially be the standard table for pan Arctic (region index 0)
    # with an extra column at the end listing the regions it 'matches' to:
    if sort_by_rank:

        vrk_want = ["vriles_joined_rates_rank", "vriles_joined_class",
                    "date_bnds_vriles_joined"]

        j_vriles = np.argsort(vresults_list[0]["vriles_joined_rates_rank"])

    else:

        vrk_want = ["date_bnds_vriles_joined", "vriles_joined_rates_rank",
                    "vriles_joined_class"]

        j_vriles = np.argsort(vresults_list[0]["date_bnds_vriles_joined"][:,0])

    # Headers and tabulate argument 'floatfmt' for table output:
    headers  = []
    floatfmt = []
    vrk_use  = []  # equal to or subset of vrk_want (discard unavailable keys)
    for k in vrk_want:
        if k in vresults_list[0].keys():
            if _vrk_props[k]["n_headers"] > 1:
                for i in range(_vrk_props[k]["n_headers"]):
                    headers.append(_vrk_props[k]["header"][i])
                    floatfmt.append(_vrk_props[k]["fmt_tab"])
            else:
                headers.append(_vrk_props[k]["header"])
                floatfmt.append(_vrk_props[k]["fmt_tab"])
            vrk_use.append(k)

    # Now append the non-vresults-based headers and floatfmt
    # (i.e., the region matches information):
    headers.append("Region match")
    floatfmt.append("")

    # List of rows for table output:
    rows = []

    # Loop over joined pan-Arctic VRILEs in order of indices j_vriles:
    for j in j_vriles:
        row_j = []  # set up row for joined pan-Arctic VRILE with array index j

        # Loop over required results dictionary keys:
        for k in vrk_use:
            if _vrk_props[k]["n_headers"] > 1:
                # Multiple headings / 2D array for key k
                for i in range(_vrk_props[k]["n_headers"]):
                    val = vresults_list[0][k][j,i]
                    if "scale" in _vrk_props[k].keys():
                        val *= _vrk_props[k]["scale"]

                    row_j.append(_vrk_props[k]["fmt_func"](val))

            else:
                # No headings / 1D array for key k
                val = vresults_list[0][k][j]
                if "scale" in _vrk_props[k].keys():
                    val *= _vrk_props[k]["scale"]

                row_j.append(_vrk_props[k]["fmt_func"](val))

        # Now append the non-vresults-based data to row_j
        # (i.e., the region matches for this pan-Arctic VRILE):
        row_j.append(", ".join(matches[j]) if len(matches[j]) > 0 else "")

        # Finally, append row to main list of all rows:
        rows.append(row_j)

    table = tabulate(rows, headers=headers, floatfmt=floatfmt)

    with open(filename, "w") as txtfile:

        _txt_file_header(txtfile, id_vriles_kw,
                         detrend_type=vresults_list[0]["detrend_type"],
                         n_ma=vresults_list[0]["moving_average_filter_n_days"],
                         title="Pan-Arctic VRILEs matched to regions",
                         additional_metadata=additional_metadata)

        txtfile.write(f"\nTable lists the {vresults_list[0]['n_joined_vriles']}"
                      + " non-overlapping, joined events for the pan-Arctic.\n")

        if sort_by_rank:
            txtfile.write("\nSorted in descending order of average rates, "
                          + "dSIE/dt (magnitudes).\n")
        else:
            txtfile.write("\nRanks are based on the magnitude of the average "
                          + "rates, dSIE/dt.\n")

        txtfile.write("\nPan-Arctic VRILES matched to regions by checking for"
                      + " any overlap of time periods:\n\n")

        for r in range(1, len(vresults_list)):
            txtfile.write(f"{r}: {region_labels[r-1]}" + "\n")

        txtfile.write("\n" + _generated_txt() + "\n\n\n" + table + "\n")

        _txt_file_footer(txtfile)


def save_tables(vresults_list, id_vriles_kw, filenames=None,
                filename_matches=None, save_dir=None, which=[True]*4,
                vresults_labels=None, additional_metadata={}, verbose=True):
    """Wrapper function taking a list of VRILE 'results dictionaries' and
    writing summary tables to text files.


    Parameters
    ----------
    vresults_list : length nV list of dict
        List of VRILE 'results dictionaries' as returned by function identify()
        in module src.diagnostics.vriles.

    id_vriles_kw : dict
        The keyword arguments passed to function identify() in module
        src.diagnostics.vriles, assumed to contain all the options used to
        generate vresults.


    Optional parameters
    -------------------
    filenames : length nV list of str, or None
        The file names to save into directory save_dir. Overwrites files if
        they already exist. If None, uses parameter 'vresults_labels' (see
        below; this defaults to the configuration region short names).

    filename_matches : str or None
        The file name to save into directory save_dir for the matches between
        regions, if which[2] or which[3] (see below). If None (default), a
        default name is given: 'pan-Arctic_to_regional_matches.txt', as this
        part is currently hard-coded to find matches between region 0
        (pan Arctic) and the remaining regions only.

    save_dir : str or pathlib.Path, or None
        The directory to save all files to (note: sub-directories are created
        automatically for summary tables sorted by rank and by date). If
        None, gets from config.

    which : length 4 list or tuple of bool, default = [True]*4
        Which tables to save, where index j = 0, 1, 2, 3 corresponds to:

            0 : regional VRILE summaries, sorted by date
            1 : regional VRILE summaries, sorted by rank
            2 : pan-Arctic VRILEs sorted by date with regional VRILE matches
            3 : pan-Arctic VRILEs sorted by rank with regional VRILE matches

    vresults_labels : length nV list of str, or None
        The labels for each VRILE results dictionary. If None (default), uses
        the default region (short) names set in configuration.

    additional_metadata : dict {key: str}
        Any other metadata key-pairs to write in the header for the record.

    verbose : bool, default = True
        Print progress and saved filenames (on success) to console.

    """

    if vresults_labels is None:
        vresults_labels = cfg.reg_labels_long

    if save_dir is None:
        save_dir = cfg.data_path["tables"]

    if filenames is None:
        filenames = vresults_labels

    # Set file name for the matches. Currently hard-coded just for region 0
    # (pan Arctic) matching to remaining regions, so default reflects that:
    if which[2] or which[3]:
        if filename_matches is None:
            filename_matches = "pan-Arctic_to_regional_matches.txt"

        if not filename_matches.endswith(".txt"):
            filename_matches += ".txt"

    # Make directories:
    Path(save_dir, "sorted_by_date").mkdir(parents=True, exist_ok=True)
    Path(save_dir, "sorted_by_rank").mkdir(parents=True, exist_ok=True)

    kw = {"id_vriles_kw": id_vriles_kw,
          "additional_metadata": additional_metadata}

    if which[0]:
        # Save individual VRILE results tables ranked by date:
        for k in range(len(vresults_list)):

            # Full file path:
            fk = Path(save_dir, "sorted_by_date", f"{filenames[k]}"
                      + ("" if filenames[k].endswith(".txt") else ".txt"))

            save_vrile_table(vresults_list[k], fk, sort_by_rank=False, **kw)
            if verbose:
                print(f"Saved: {str(fk)}")

    if which[1]:
        # Save individual VRILE results tables ranked by value:
        for k in range(len(vresults_list)):

            # Full file path:
            fk = Path(save_dir, "sorted_by_rank", f"{filenames[k]}"
                      + ("" if filenames[k].endswith(".txt") else ".txt"))

            save_vrile_table(vresults_list[k], fk, sort_by_rank=True, **kw)

            if verbose:
                print(f"Saved: {str(fk)}")

    if which[2]:
        # Save pan-Arctic to regional VRILE matches ranked by date:
        fk = Path(save_dir, "sorted_by_date", filename_matches)
        save_vrile_matches_to_txt(vresults_list, fk, sort_by_rank=False,
                                  region_labels=vresults_labels, **kw)
        if verbose:
            print(f"Saved: {str(fk)}")

    if which[3]:
        # Save pan-Arctic to regional VRILE matches ranked by value:
        fk = Path(save_dir, "sorted_by_rank", filename_matches)
        save_vrile_matches_to_txt(vresults_list, fk, sort_by_rank=True,
                                  region_labels=vresults_labels, **kw)
        if verbose:
            print(f"Saved: {str(fk)}")


def save_vrile_set_intersection(v1, v2, filename, id_vriles_kw,
                                v1_label="cice", v2_label="obs",
                                v1_title=None, v2_title=None,
                                region_name="Arctic", additional_metadata={},
                                sort_by_rank=False, verbose=True):
    """Save a sub-set of VRILEs from a VRILE 'results dictionary' v1 that
    overlap in timespane with one or more VRILEs in a second 'results
    directionary' v2. Also updates the metadata of v1 and v2 with the indices
    of such VRILEs that 'match' in this way between the two sets, and the total
    number of such matches. Does this for the 'joined' VRILEs only.


    Parameters
    ----------
    v1, v2 : dict
        The VRILE 'results dictionaries' as returned by function identify() in
        module src.diagnostics.vriles. These are assumed to contain both the
        data and metadata required for the summary table.

    filename : str or pathlib.Path
        The file to save. Overwrites if it already exists, and assumes that the
        directory exists (otherwise, this fails).

    id_vriles_kw : dict
        The keyword arguments passed to the function
        src.diagnostics.vriles.identify().


    Optional Parameters
    -------------------
    v1_label, v2_label : str, default is 'cice' and 'obs, respectively
        Short label each for v1 and v2 to be used in new keys added to v1 and
        v2 (v1 gets 'indices_joined_vriles_matched_to{v2_label}' and
        'n_joined_vriles_matched_to{v2_label}', and vice-versa for v2).

    v1_title, v2_title : str or None
        Labels for v1 and v2 used in metadata and headings of output text
        file. If None (default), v1_label and v2_label are used instead.

    region_name : str, default = 'Arctic'
        Description/label of the region corresponding to v1 and v2
        (typically set this to a region name, e.g., Barents Sea, etc.)

    additional_metadata : dict {key: str}, default = {}
        Additional metadata values to include in the header information.

    sort_by_rank : bool, default = False
        If True, the VRILEs in the table are sorted in the order of the rank
        of the joined v1 VRILEs. Otherwise (default), they are in date order.

    verbose : bool, default = True
        Print progress and saved filenames (on success) to console.


    Returns
    -------
    v1, v2
        Updated with new metadata keys 'indices_joined_vriles_matched_to_{x}'
        and 'n_joined_vriles_matched_to{x}' where x is v2_label in v1 and
        x is v1_label in v2. These contain, respectively, the indices of
        the joined VRILEs that are matched to the other VRILE set (as a boolean
        array) and the number of such matches (integer). Note the latter is the
        same for both.

    """

    if v1_title is None:
        v1_title = v1_label

    if v2_title is None:
        v2_title = v2_label

    # Create a list of length-2 lists [change to an array of shape (n_matches, 2)
    # at the end] for the indices of VRILEs which match, where first index of
    # second axis corresponds to v1 and second index to v2:
    matches = []

    for k in range(v1["n_joined_vriles"]):

        # Date bounds of current VRILE considered:
        db_k = v1["date_bnds_vriles_joined"][k,:]

        overlap = np.array([(db_k[0] <= x) & (db_k[1] >= x)
                            for x in v2["date_bnds_vriles_joined"][:,0]]).astype(bool)

        if any(overlap):
            matches.append([k, np.argmax(overlap)])

    matches = np.array(matches)
    n_matched = len(matches)

    # Update metadata:
    if n_matched > 0:

        v1[f"indices_joined_vriles_matched_to_{v2_label}"] \
            = np.array([j in matches[:,0]
                        for j in range(v1["n_joined_vriles"])]).astype(bool)

        v2[f"indices_joined_vriles_matched_to_{v1_label}"] \
            = np.array([j in matches[:,1]
                        for j in range(v2["n_joined_vriles"])]).astype(bool)

    else:
        v1[f"indices_joined_vriles_matched_to_{v2_label}"] \
            = np.array([]).astype(bool)
        v2[f"indices_joined_vriles_matched_to_{v1_label}"] \
            = np.array([]).astype(bool)

    v1[f"n_joined_vriles_matched_to_{v2_label}"] = n_matched
    v2[f"n_joined_vriles_matched_to_{v1_label}"] = n_matched

    # Construct table for text file write. Order of rows depends whether
    # sorting by rank or by date, but headers are the same in each case.
    #
    # For rows, specify indices of matches array to select from. By default,
    # this is just the order they are identified in (i.e., date order of v1).
    # Otherwise, sort by ranks of matched VRILEs in v1.
    #
    row_indices = np.arange(n_matched).astype(int)
    if sort_by_rank and n_matched > 1:
        row_indices = row_indices[
            np.argsort(v1["vriles_joined_rates_rank"][matches[:,0]])]

    vrk_want = ["date_bnds_vriles_joined", "vriles_joined_rates_rank",
                "vriles_joined_class", "vriles_joined_rates"]

    # Headers and tabulate argument 'floatfmt' for table output:
    headers    = []
    floatfmt   = []
    vrk_use_v1 = []  # equal to or subset of vrk_want, possibly different to v2
    vrk_use_v2 = []  # equal to or subset of vrk_want, possibly different to v1

    # For of v1 and v2, loop over each required key k, check if it exists in v*,
    # and if so, append the appropriate header(s) to headers list and k to the
    # vrk_use_* list:
    for vrk_use, v_title, v_dict in zip([vrk_use_v1, vrk_use_v2],
                                        [v1_title  , v2_title  ],
                                        [v1        , v2        ]):
        for k in vrk_want:
            if k in v_dict.keys():
                if _vrk_props[k]["n_headers"] > 1:
                    for i in range(_vrk_props[k]["n_headers"]):
                        headers.append(f"{v_title}\n{_vrk_props[k]['header'][i]}")
                        floatfmt.append(_vrk_props[k]["fmt_tab"])
                else:
                    headers.append(f"{v_title}\n{_vrk_props[k]['header']}")
                    floatfmt.append(_vrk_props[k]["fmt_tab"])

                vrk_use.append(k)

    rows = []

    # Loop over matched VRILEs in order of indices set above (row_indices):
    for j in row_indices:

        row_j = []  # set up row for next matched VRILE

        # Indices of actual VRILE data in v1 and v2, respectively:
        jv1 = matches[j,0]
        jv2 = matches[j,1]

        # Loop over required results dictionary keys for v1:
        for vrk_use, v, jv in zip([vrk_use_v1, vrk_use_v2], [v1, v2], [jv1, jv2]):
            for k in vrk_use:
                if _vrk_props[k]["n_headers"] > 1:
                    # Multiple headings / 2D array for key k
                    for i in range(_vrk_props[k]["n_headers"]):
                        val = v[k][jv,i]
                        if "scale" in _vrk_props[k].keys():
                            val *= _vrk_props[k]["scale"]

                        row_j.append(_vrk_props[k]["fmt_func"](val))

                else:
                    # One heading / 1D array for key k
                    val = v[k][jv]
                    if "scale" in _vrk_props[k].keys():
                        val *= _vrk_props[k]["scale"]

                    row_j.append(_vrk_props[k]["fmt_func"](val))

        rows.append(row_j)

    table = tabulate(rows, headers=headers, floatfmt=floatfmt)

    title = f"{region_name} VRILEs matched between {v1_title} and {v2_title}"

    with open(filename, "w") as txtfile:

        # Assume that v1 and v2 have the same 'detrend_type'
        # and 'moving_average_filter_n_days':
        _txt_file_header(txtfile, id_vriles_kw,
                         detrend_type=v1["detrend_type"],
                         n_ma=v1["moving_average_filter_n_days"],
                         title=title, header_length=max(80, len(title)),
                         additional_metadata=additional_metadata)

        txtfile.write("\n\n")

        txtfile.write(f"Table lists the {n_matched} non-overlapping, joined "
                      + f"events for the {region_name},\ndate-matched between "
                      + f"{v1_title} and {v2_title}. Here, 'date-matched' "
                      + "means any\noverlap of date bounds between each pair "
                      + "of events.\n")

        if sort_by_rank:
            txtfile.write("\nSorted in descending order of average rates, "
                          + f"dSIE/dt (magnitudes) for {v1_title} VRILEs.\n")
        else:
            txtfile.write("\nRanks are based on the magnitude of the average "
                          + f"rates, dSIE/dt, for {v1_title} VRILEs.\n")

        txtfile.write("\n" + _generated_txt() + "\n\n\n" + table + "\n")

        _txt_file_footer(txtfile)

    if verbose:
        print(f"Saved: {str(filename)}")

    return v1, v2


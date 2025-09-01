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
                     n_ma=5, title="VRILEs", header_length=52,
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

    header_length : int, default = 52
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

    txtfile.write("    N_days         = ")
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
        for amk in additional_metadata.keys():
            txtfile.write(f"    {amk}: {additional_metadata[amk]}" + "\n")


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

    if sort_by_rank:

        headers = ["RANK", "START", "END", "N_DAYS", "DELTA_SIE\n(10^6 km^2)",
                   "DELTA_SIE/N_DAYS\n(10^3 km^2/day)"]

        sort = np.argsort(vresults["vriles_joined_rates_rank"])

        rows = [[_fmt_int(vresults["vriles_joined_rates_rank"][j]),
                 _fmt_date(vresults["date_bnds_vriles_joined"][j,0]),
                 _fmt_date(vresults["date_bnds_vriles_joined"][j,1]),
                 _fmt_int(vresults["vriles_joined_n_days"][j]),
                 _fmt_float(vresults["vriles_joined"][j]),
                 _fmt_float(1.E3*vresults["vriles_joined_rates"][j])]
                for j in sort]

        table = tabulate(rows, headers=headers,
                         floatfmt=("03d", "", "", "03d", ".2f", ".1f"))

    else:

        headers = ["START", "END", "N_DAYS", "RANK", "DELTA_SIE\n(10^6 km^2)",
                   "DELTA_SIE/N_DAYS\n(10^3 km^2/day)"]

        rows = [[_fmt_date(vresults["date_bnds_vriles_joined"][j,0]),
                 _fmt_date(vresults["date_bnds_vriles_joined"][j,1]),
                 _fmt_int(vresults["vriles_joined_n_days"][j]),
                 _fmt_int(vresults["vriles_joined_rates_rank"][j]),
                 _fmt_float(vresults["vriles_joined"][j]),
                 _fmt_float(1.E3*vresults["vriles_joined_rates"][j])]
                for j in range(vresults["n_joined_vriles"])]

        table = tabulate(rows, headers=headers,
                         floatfmt=("", "", "03d", "03d", ".2f", ".1f"))

    with open(filename, "w") as txtfile:
        
        _txt_file_header(txtfile, id_vriles_kw,
                         detrend_type=vresults["detrend_type"],
                         n_ma=vresults["moving_average_filter_n_days"],
                         title=vresults["title"],
                         additional_metadata=additional_metadata)

        txtfile.write("\nTable lists non-overlapping, joined events (hence\n"
                      + "the different N_days), of which there are a total "
                      + ("%i." % vresults['n_joined_vriles']) + "\n")

        if sort_by_rank:
            txtfile.write("\nSorted in descending order of average rate of "
                          + "dSIE\n(magnitude; last column).\n")
        else:
            txtfile.write("\nRanks are based on the average rate of dSIE.\n")

        txtfile.write("\n" + _generated_txt() + "\n\n" + table + "\n")

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

    if sort_by_rank:

        headers = ["RANK", "START", "END", "REGIONS"]

        sort = np.argsort(vresults_list[0]["vriles_joined_rates_rank"])

        rows = [[_fmt_int(vresults_list[0]["vriles_joined_rates_rank"][j]),
                 _fmt_date(vresults_list[0]["date_bnds_vriles_joined"][j,0]),
                 _fmt_date(vresults_list[0]["date_bnds_vriles_joined"][j,1]),
                 ", ".join(matches[j]) if len(matches[j]) > 0 else "None"]
                for j in sort]

        table = tabulate(rows, headers=headers, floatfmt=("03d", "", "", ""))

    else:

        headers = ["START", "END", "RANK", "REGIONS"]

        rows = [[_fmt_date(vresults_list[0]["date_bnds_vriles_joined"][j,0]),
                 _fmt_date(vresults_list[0]["date_bnds_vriles_joined"][j,1]),
                 _fmt_int(vresults_list[0]["vriles_joined_rates_rank"][j]),
                 ", ".join(matches[j]) if len(matches[j]) > 0 else "None"]
                for j in range(vresults_list[0]["n_joined_vriles"])]

        table = tabulate(rows, headers=headers, floatfmt=("", "", "03d", ""))

    with open(filename, "w") as txtfile:

        _txt_file_header(txtfile, id_vriles_kw,
                         detrend_type=vresults_list[0]["detrend_type"],
                         n_ma=vresults_list[0]["moving_average_filter_n_days"],
                         title="Pan-Arctic VRILEs matched to regions",
                         additional_metadata=additional_metadata)

        txtfile.write("\nTable lists non-overlapping, joined events for\n"
                      + "the pan-Arctic, of which there are a total "
                      + f"{vresults_list[0]['n_joined_vriles']}.\n")

        if sort_by_rank:
            txtfile.write("\nSorted in descending order of average rate\n"
                          + "of dSIE (magnitude).\n")
        else:
            txtfile.write("\nRanks are based on the average rate of dSIE.\n")

        txtfile.write("\nPan-Arctic VRILES matched to regions by\nchecking for"
                      + " any overlap of time periods:\n\n")

        for r in range(1, len(vresults_list)):
            txtfile.write(f"{r}: {region_labels[r-1]}" + "\n")

        txtfile.write(f"\n\n{_generated_txt()}\n\n\n{table}\n")

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

    # Construct table for text file write:
    row_indices = np.arange(n_matched).astype(int)
    if sort_by_rank and n_matched > 1:
        row_indices = row_indices[
            np.argsort(v1["vriles_joined_rates_rank"][matches[:,0]])]

    headers = [f"{v1_title} RANK", f"{v2_title} RANK",
               f"{v1_title} DELTA_SIE/N_DAYS\n(10^3 km^2 day^-1)",
               f"{v2_title} DELTA_SIE/N_DAYS\n(10^3 km^2 day^-1)",
               f"{v1_title} START", f"{v1_title} END",
               f"{v2_title} START", f"{v2_title} END"]

    rows = [[_fmt_int(v1["vriles_joined_rates_rank"][matches[j,0]]),
             _fmt_int(v2["vriles_joined_rates_rank"][matches[j,1]]),
             _fmt_float(1.E3*v1["vriles_joined_rates"][matches[j,0]]),
             _fmt_float(1.E3*v2["vriles_joined_rates"][matches[j,1]]),
             _fmt_date(v1["date_bnds_vriles_joined"][matches[j,0],0]),
             _fmt_date(v1["date_bnds_vriles_joined"][matches[j,0],1]),
             _fmt_date(v2["date_bnds_vriles_joined"][matches[j,1],0]),
             _fmt_date(v2["date_bnds_vriles_joined"][matches[j,1],1])]
            for j in row_indices]

    table = tabulate(rows, headers=headers, floatfmt=("", "", "", ""))

    title = f"{region_name} VRILEs matched between {v1_title} and {v2_title}"

    with open(filename, "w") as txtfile:

        # Assume that v1 and v2 have the same 'detrend_type'
        # and 'moving_average_filter_n_days':
        _txt_file_header(txtfile, id_vriles_kw,
                         detrend_type=v1["detrend_type"],
                         n_ma=v1["moving_average_filter_n_days"],
                         title=title, header_length=len(title),
                         additional_metadata=additional_metadata)

        txtfile.write("\n\n")

        txtfile.write("Table lists non-overlapping, joined events for the\n"
                      + f"{region_name}, date-matched between {v1_title} and "
                      + f"{v2_title}, of\nwhich there are a total {n_matched}.\n")

        txtfile.write("\nHere, 'date-matched' simply means checking for any\n"
                      + "overlap of date bounds.\n")

        if sort_by_rank:
            txtfile.write("\nSorted in descending order of average rate\n"
                          + f"of dSIE (magnitude) for {v1_title} VRILEs.\n")
        else:
            txtfile.write("\nRanks are based on the average rate of dSIE.\n")

        txtfile.write(f"\n\n{_generated_txt()}\n\n\n{table}\n")
        
        _txt_file_footer(txtfile)

    if verbose:
        print(f"Saved: {str(filename)}")

    return v1, v2


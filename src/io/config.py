"""Provides a function to read configuration files from the ./cfgs directory
and set paths to data/outputs, and other metadata, as module variables, such
that other modules can access them. Data paths are stored in the dictionary
'data_path' with various keys corresponding to different datasets/groups. The
file 'cfg_default.ini' under ./cfgs includes all mandatory keys required by
this dictionary and is always parsed first.
"""

from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
import os
import warnings

data_path = {}


# Regions/sectors information... bit tricky to 'generalise' this at this stage,
# so just hard-coding the required info as used in the ASC work for now. At
# least putting it here means all other modules/code gets the info from the
# same place rather than hard-coding it individually, so it should be easier
# to update in the future if necessary.
#
# Record info for all defined regions in the region masks files, and use the
# config files to get the actual list of regions to use (i.e., not all of them)
# using the netCDF names as keys (well, they have "mask_" prepended as well):
#

_all_reg_info = {
    "pan_Arctic"                : {"long_name": "Pan Arctic"                     , "short_name": "PAN", "sector_lon_bnds": [(0., 360.)]             },
    "labrador_sea"              : {"long_name": "Labrador Sea"                   , "short_name": "LAB", "sector_lon_bnds": [(275., 315.)]           },
    "gin_seas"                  : {"long_name": "Greenland Sea"                  , "short_name": "GRE", "sector_lon_bnds": [(315., 360.), (0., 20.)]},
    "barents_sea"               : {"long_name": "Barents Sea"                    , "short_name": "BAR", "sector_lon_bnds": [(20., 60.)]             },
    "kara_sea"                  : {"long_name": "Kara Sea"                       , "short_name": "KAR", "sector_lon_bnds": [(60., 100.)]            },
    "laptev_sea"                : {"long_name": "Laptev Sea"                     , "short_name": "LAP", "sector_lon_bnds": [(100., 150.)]           },
    "east_siberian_chukchi_seas": {"long_name": u"East Siberian\u2013Chukchi Sea", "short_name": "ESC", "sector_lon_bnds": [(150., 200.)]           },
    "beaufort_sea"              : {"long_name": "Beaufort Sea"                   , "short_name": "BEA", "sector_lon_bnds": [(200., 245.)]           },
    "canadian_archipelago"      : {"long_name": "Canadian Archipelago"           , "short_name": "CAN", "sector_lon_bnds": [(245., 275.)]           }
}

# Create lists of the above for convenience/public access:
all_reg_nc_names        = list(_all_reg_info.keys())
all_reg_long_names      = [_all_reg_info[k]["long_name"] for k in all_reg_nc_names]
all_reg_short_names     = [_all_reg_info[k]["short_name"] for k in all_reg_nc_names]
all_reg_sector_lon_bnds = [_all_reg_info[k]["sector_lon_bnds"] for k in all_reg_nc_names]

# This dictionary of data paths to be accessed by other modules:
data_path = {}


def set_config(*cfg_files):
    """Read configuration files using the configparser library. Any number of
    input arguments may be passed to this function, each of which corresponds
    to a configuration file (with or without the 'cfg_' prefixes and '.ini'
    suffixes) stored under the repository cfgs directory. The default
    configuration file ('cfg_default.ini') is parsed first, then the inputs
    to this function in the order they are passed in, such that repeated
    parameters are overwritten in that same order.
    """

    # Path containing configuration files:
    cfgs_path = Path(os.path.dirname(__file__), "..", "..", "cfgs").resolve()

    # The fallback/default configuration:
    cfg_file_default = "cfg_default.ini"

    # Convert input(s) to a list of files even if only one:
    cfg_files = list(cfg_files)

    # Append default cfg to list if not present:
    if cfg_file_default not in cfg_files:
        cfg_files = [cfg_file_default] + cfg_files

    # Get list of configuration files as Paths:
    cfg_paths = []
    for j in range(len(cfg_files)):

        cfg_file_j = cfg_files[j]

        # Check it is in the correct format ('cfg_*.ini'):
        if not cfg_file_j.startswith("cfg_"):
            cfg_file_j = "cfg_" + cfg_file_j

        if not cfg_file_j.endswith(".ini"):
            cfg_file_j += ".ini"

        cfg_path_j = Path(cfgs_path, cfg_file_j)

        if cfg_path_j.is_file():
            cfg_paths.append(cfg_path_j)
        else:
            warnings.warn("Configuration file does not exist; skipping: "
                          + str(cfg_path_j))

    # Parse the configuration file(s):
    cfg_loc = ConfigParser(interpolation=ExtendedInterpolation())
    cfg_loc.read(cfg_paths)

    # General metadata (some of these used in writing outputs if not blank):
    globals()["author"]      = cfg_loc.get("METADATA", "author"     , vars=os.environ)
    globals()["title"]       = cfg_loc.get("METADATA", "title"      , vars=os.environ)
    globals()["institution"] = cfg_loc.get("METADATA", "institution", vars=os.environ)
    globals()["contact"]     = cfg_loc.get("METADATA", "contact"    , vars=os.environ)

    for k in ["cice", "amsr", "ssmi", "track", "atmo"]:
        globals()[f"{k}_title"]  = cfg_loc.get(f"{k.upper()} DATA", "title")

    # Paths to 'history' (CICE output data) and post-'processed' data:
    for k in ["hist_d", "hist_m", "hist_3h", "proc_d", "proc_m"]:
        globals()["data_path"][k] = Path(cfg_loc.get("CICE DATA", f"dir_{k}")).resolve()

    # Paths to grid-related data:
    globals()["data_path"]["grid"] = Path(cfg_loc.get("GRID DATA", "file_grid")).resolve()
    for x in ["129", "131"]:
        globals()["data_path"][f"cosa{x}"] = Path(cfg_loc.get("GRID DATA", f"file_cosa{x}")).resolve()
        globals()["data_path"][f"sina{x}"] = Path(cfg_loc.get("GRID DATA", f"file_sina{x}")).resolve()
        globals()["data_path"][f"lmsk{x}"] = Path(cfg_loc.get("GRID DATA", f"file_lmsk{x}")).resolve()
        globals()["data_path"][f"regs{x}"] = Path(cfg_loc.get("GRID DATA", f"file_regs{x}")).resolve()

    # Get region info:
    globals()["reg_nc_names"]        = [x.strip() for x in cfg_loc.get("GRID DATA", "reg_nc_names").split(",")]
    globals()["reg_labels_long"]     = [_all_reg_info[x]["long_name"] for x in globals()["reg_nc_names"]]
    globals()["reg_labels_short"]    = [_all_reg_info[x]["short_name"] for x in globals()["reg_nc_names"]]
    globals()["reg_sector_lon_bnds"] = [_all_reg_info[x]["sector_lon_bnds"] for x in globals()["reg_nc_names"]]

    # AMSR-E/AMSR2 paths:
    globals()["data_path"]["amsr_raw"]       = Path(cfg_loc.get("AMSR DATA", "dir_raw")).resolve()
    globals()["data_path"]["amsr_cice"]      = Path(cfg_loc.get("AMSR DATA", "dir_cice_grid")).resolve()

    # SSM/I paths:
    globals()["data_path"]["ssmi_raw_lonlat"] = Path(cfg_loc.get("SSMI DATA", "file_raw_lonlat")).resolve()
    globals()["data_path"]["ssmi_bt_raw"]     = Path(cfg_loc.get("SSMI DATA", "dir_raw_bt")).resolve()
    globals()["data_path"]["ssmi_nt_raw"]     = Path(cfg_loc.get("SSMI DATA", "dir_raw_nt")).resolve()
    globals()["data_path"]["ssmi_bt_cice"]    = Path(cfg_loc.get("SSMI DATA", "dir_cice_bt")).resolve()
    globals()["data_path"]["ssmi_nt_cice"]    = Path(cfg_loc.get("SSMI DATA", "dir_cice_nt")).resolve()

    # Track data path:
    globals()["data_path"]["tracks"] = Path(cfg_loc.get("TRACK DATA", "dir")).resolve()
    
    # Atmospheric data paths and metadata:
    globals()["data_path"]["atmo_d"]    = Path(cfg_loc.get("ATMO DATA", "dir_atmo_d")).resolve()
    globals()["data_path"]["atmo_m"]    = Path(cfg_loc.get("ATMO DATA", "dir_atmo_m")).resolve()
    globals()["data_path"]["atmo_forc"] = Path(cfg_loc.get("ATMO DATA", "dir_forc")).resolve()
    globals()["data_path"]["atmo_rean"] = Path(cfg_loc.get("ATMO DATA", "dir_rean")).resolve()
    globals()["reanalysis_name"]        = cfg_loc.get("ATMO DATA", "title")
    globals()["reanalysis_frequency"]   = cfg_loc.get("ATMO DATA", "frequency")

    # Output directories:
    globals()["data_path"]["cache"]  = Path(cfg_loc.get("RESULTS", "dir_cache")).resolve()
    globals()["data_path"]["tables"] = Path(cfg_loc.get("RESULTS", "dir_tables")).resolve()
    globals()["fig_save_dir"]        = Path(cfg_loc.get("RESULTS", "dir_figures"   , vars=os.environ)).resolve()
    globals()["ani_save_dir"]        = Path(cfg_loc.get("RESULTS", "dir_animations", vars=os.environ)).resolve()

    # General 'fallback' directory to search for other input data:
    globals()["data_path"]["misc"] = Path(cfg_loc.get("OTHER DATA", "dir", vars=os.environ)).resolve()


def print_config_vars(sort_paths=True):
    """Print the paths and other metadata read in by the configuration.
    
    Optional parameters
    -------------------
    sort_paths : bool, default = True
        Whether to sort the paths keys in alphabetical order.

    """

    print("\nMetadata")
    print("--------")
    print(f"Author      : {author}")
    print(f"Institution : {institution}")
    print(f"Contact     : {contact}")
    print(f"Title       : {title}")

    print("\nData source titles/labels")
    print("-------------------------")
    for k in ["amsr", "atmo", "cice", "ssmi", "track"]:
        print(f"{k.upper():<5} : {globals()[f'{k}_title']}")

    if sort_paths:
        paths_keys = sorted(list(data_path.keys()))
    else:
        paths_keys = list(data_path.keys())

    paths_strs = [str(data_path[k]) for k in paths_keys]

    # Determine if there is a common prefix, to shorten print:
    prefix = os.path.commonpath(paths_strs)

    # Width of longest key:
    w = max([len(key) for key in data_path.keys()])

    print("\nData paths (dict: data_path):")
    print("-----------------------------")

    if len(prefix) > len(str(os.path.abspath(os.sep))):
        print(f"All paths relative to: ./ = {prefix}")
        for j in range(len(paths_keys)):
            print(f"{paths_keys[j]:>{w}} : ./{paths_strs[j][len(prefix)+1:]}")

    else:
        for j in range(len(paths_keys)):
            print(f"{paths_keys[j]:>{w}} : {paths_strs[j]}")

    print("\nRegion names:")
    print("-------------")
    for i in range(len(reg_nc_names)):
        print(f"{i:>2} : '{reg_nc_names[i]}'")

    print("")


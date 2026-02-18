"""Create a multi-panel figure showing the changes in various quantities during
a specified VRILE case study. This is a generic script: command-line arguments
are defined to tailor which quantities are shown for different cases. For the
manuscript figures use the bash script fig_case_study.sh.

Use --help for usage, & see comments in function parse_cmd_args() below details.
"""

from datetime import datetime as dt, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src import script_tools
from src.io import cache, config as cfg
from src.data import atmo, cice, rotate_vectors, ssmi, tracks
from src.plotting import maps, style_ms, symbols, tools

import posterproxy as psp

# Long titles for each possible plotted field (above map with panel label):
ax_titles = {"-": "", "sectors": "Sector boundaries"}
ax_titles["daidt"]      = "Sea ice fraction change"
ax_titles["dvidtd"]     = "Dynamics contribution"
ax_titles["dvidtt"]     = "Thermodynamics contribution"
ax_titles["ssmi"]       = "Observations (SSM/I)"
ax_titles["div_strair"] = "Wind-stress divergence"
ax_titles["qlw"]        = "Downwelling longwave"
ax_titles["qsw"]        = "Downwelling shortwave"
ax_titles["qnet"]       = "Net downwelling radiation"
ax_titles["seb_ai"]     = "Surface energy balance (SEB)"
ax_titles["t2"]         = "Air temperature"
ax_titles["meltb"]      = "Basal melt"
ax_titles["meltl"]      = "Lateral melt"
ax_titles["meltt"]      = "Top melt"

# Color bar labels:
cbar_titles = {"-": "", "sectors": ""}
cbar_titles["daidt"]      = "$" + symbols.delta_aice + "$"
cbar_titles["dvidtd"]     = "$" + symbols.delta_vice + "$ (cm)"
cbar_titles["dvidtt"]     = "$" + symbols.delta_vice + "$ (cm)"
cbar_titles["ssmi"]       = cbar_titles["daidt"]
cbar_titles["div_strair"] = "$" + symbols.div_strair + "$ ($10^{-7}$ N m$^{-3}$)"
cbar_titles["qlw"]        = "$" + symbols.qlw        + "$ (W m$^{-2}$)"
cbar_titles["qsw"]        = "$" + symbols.qsw        + "$ (W m$^{-2}$)"
cbar_titles["qnet"]       = "$" + symbols.qnet       + "$ (W m$^{-2}$)"
cbar_titles["seb_ai"]     = "$" + symbols.seb_ai     + "$ (W m$^{-2}$)"
cbar_titles["t2"]         = "$" + symbols.t2         + "$ ($\degree$C)"
cbar_titles["meltb"]      = "$" + symbols.delta + symbols.meltb + "$ (cm)"
cbar_titles["meltl"]      = "$" + symbols.delta + symbols.meltl + "$ (cm)"
cbar_titles["meltt"]      = "$" + symbols.delta + symbols.meltt + "$ (cm)"


def parse_cmd_args():
    """Define and parse command line arguments using the argparse module.
    Returns the parsed arguments.
    """

    prsr = script_tools.argument_parser(usage="VRILE case study plots")

    # These options are used to determine which VRILE to load:
    prsr.add_argument("-r", "--region", type=int, default=5,
                      choices=[r for r in range(8)], help="Region index")
    prsr.add_argument("-k", "--rank", type=int, default=5,
                      help="Rank of VRILE to select for specified -r/--region")

    # Alternative: specify start and end datetimes and load data for that
    # time period (this allows 'cyclone, but no VRILE' case studies):
    prsr.add_argument("--dt-start", type=int, nargs=3, default=(1950, 1, 1),
                      help="Manual datetime start Y M D (-r and -k ignored if "
                           + "this is later than default 1950-01-01)")

    prsr.add_argument("--dt-end", type=int, nargs=3, default=(1950, 1, 1),
                      help="Manual datetime end Y M D (-r and -k ignored if "
                           + "this is later than default 1950-01-01)")

    # Cyclone tracks are picked automatically based on VRILE specified, but
    # more can be added manually using the --add-tracks option:
    prsr.add_argument("--add-tracks", type=str, nargs="*", default=[],
                      help="Track IDs to add to plot")

    # The automatically-picked tracks (which are sometimes spurious) can be
    # removed using --rm-auto-tracks:
    prsr.add_argument("--rm-auto-tracks", type=str, nargs="*", default=[],
                      help="Track IDs NOT to add to plot")

    # Options for vector (quiver; qv) plots:
    prsr.add_argument("--qv-sample", type=int, default=5,
                      help="Select every --qv-sample data points in space to "
                           + "plot quivers (arrows)")

    # By default, vectors are plotted at the time of maximum vorticity for the
    # cyclone track with peak vorticity or, if there are no tracks, at the
    # start date. This option adds --qv-days-offset days (which can be
    # negative) to the date at which vectors are plotted:
    prsr.add_argument("--qv-days-offset", type=int, default=0,
                      help="Days to adjust plotting of vectors")

    # Alternatively, plot vectors averaged over the VRILE/specified time period:
    prsr.add_argument("--qv-plot-mean", action="store_true",
                      help="Plot mean vectors over start/end dates")

    # Factor by which to adjust scale of plotted vectors. Default (with this
    # parameter = 1.) plots winds with key 20 m/s and wind-stress divergence
    # with key 0.2 N/m2 calibrated for Sep 2018 case study. The underlying
    # scale options given to plt.quiver() are not intuitive so this factor
    # just scales up/down by this factor to adjust as required.
    prsr.add_argument("--qv-adjust-factor", type=float, default=1.,
                      help="Factor to scale quiver key/scale")

    # Options for the map projection (posterproxy package):
    prsr.add_argument("--extent-latitude", type=float, default=70.,
                      help="Outer latitude of plot")
    prsr.add_argument("--central_longitude", type=float, default=0.,
                      help="Central longitude of plot")
    prsr.add_argument("--xy-offset", type=float, nargs=2, default=(0., 0.),
                      help="Translation of viewport in (x,y) space for plot")

    # Specify what to plot in each panel. Six must be specified, corresponding
    # to a 2 rows by 3 column figure, but any can be set to "-" to have that
    # panel removed:
    prsr.add_argument("--panels-pcm", type=str, nargs=6,
                      default=["daidt", "dvidtd"    , "dvidtt",
                               "ssmi" , "div_strair", "sectors"],
                      choices=list(ax_titles.keys()),
                      help="What variable to plot in each panel")

    # Similar for vectors ('u10' and 'v10' both mean wind vectors, not the
    # components: I just like to use the actual variable names for consistency):
    prsr.add_argument("--panels-qv", type=str, nargs=6,
                      default=["u10", "-", "-", "-", "strair", "-"],
                      choices=["u10", "v10", "strair", "-"],
                      help="What vectors to plot in each panel")

    # Indicate whether to plot the tracks and/or ice edge latitude change in
    # each panel (0 = no, 1 = yes):
    prsr.add_argument("--panels-tracks", type=int, nargs=6, default=[1] + [0]*5,
                      help="Whether to plot cyclone tracks in each panel (0 or 1)")

    prsr.add_argument("--panels-iel", type=int, nargs=6, default=[1]*6,
                      help="Plot ice-edge latitude change in each panel (0 or 1)")

    # Options for SSM/I:
    prsr.add_argument("--which-ssmi-dataset", type=str, default="bt",
                      choices=["bt", "nt"], help="Which SSM/I dataset ['bt'/'nt']")

    # Generic plotting command-line arguments (contains --savefig, etc.):
    script_tools.add_plot_cmd_args(prsr, fig_names="fig2")

    return prsr.parse_args()


def label_seas_on_axes(ax_psp):
    """Add text labels naming the Arctic shelf seas to a map plot prepared in
    the polar stereographic projection (ax_psp).
    """

    # Text labels, along with their position (center alignment) in polar
    # stereographic (x,y) coordinates:
    seas = [["Greenland\nSea"    , (-0.025, -0.670)],
            ["Barents\nSea"      , ( 0.474, -0.586)],
            ["Kara\nSea"         , ( 0.491, -0.150)],
            ["Laptev\nSea"       , ( 0.452,  0.372)],
            ["East Siberian\nSea", ( 0.200,  0.791)],
            ["Chukchi\nSea"      , (-0.150,  0.865)],
            ["Beaufort\nSea"     , (-0.545,  0.610)],
            ["Labrador\nSea"     , (-0.760, -0.364)]]

    seas_txt_props = {"fontsize": .45*mpl.rcParams["font.size"],
                      "fontstyle": "italic", "color":"k",
                      "ha": "center", "va": "center",
                      "zorder": maps.get_zorder("text_seas_label")}

    for k in range(len(seas)):
        ax_psp.annotate(seas[k][0], seas[k][1], **seas_txt_props)


def label_sector_definitions(ax_psp, psp_xy_kw, color="tab:red"):
    """Add the sector boundaries as labelled meridians on a map plot prepared
    in the polar stereographic projection (ax_psp).
    """
    
    # Longitudes of each meridian:
    longitudes = [315., 20., 60., 100., 150., 200., 245., 275.]
    longitudes = [np.array([j,j]) for j in longitudes]

    # Longitude text labels:
    longitude_labels = [r"45$\degree$W" , r"20$\degree$E" , r"60$\degree$E" ,
                        r"100$\degree$E", r"150$\degree$E", r"160$\degree$W",
                        r"115$\degree$W", r"85$\degree$W"]

    # Latitude limits for each meridian:
    latitudes = [np.array([60., 89.5]) for j in range(len(longitudes))]

    # Latitudes at which to write each text label:
    latitudes_texts = [86., 73.5, 80., 85., 80., 77.5, 82., 86.]

    # Longitude offset for text label (?)
    txt_dlon = 1.0

    for j in range(len(longitudes)):
        # (x,y) coordinates of the meridian end points:
        xj, yj = psp.lonlat_to_xy_npsp(longitudes[j], latitudes[j], **psp_xy_kw)

        rot_offset = -90. if longitudes[j][0] < 180. else 90.

        longitudes[j][0] += txt_dlon*(-1 if rot_offset<0 else 1)

        # (x,y) coordinates of the text label (center):
        xjtxt, yjtxt = psp.lonlat_to_xy_npsp(longitudes[j][0],
            latitudes_texts[j], **psp_xy_kw)

        ax_psp.plot(xj, yj, color=color, linewidth=1.5*mpl.rcParams["lines.linewidth"],
                    zorder=maps.get_zorder("sector_boundary_line"))

        ax_psp.annotate(longitude_labels[j], (xjtxt, yjtxt), ha="center",
                        rotation=longitudes[j][0]+rot_offset, va="center",
                        fontsize=mpl.rcParams["font.size"]-3, color=color,
                        zorder=1+maps.get_zorder("sector_boundary_line"),
                        bbox={"boxstyle": "square,pad=0.2", "fc": "w", "ec": "none"})


def main():

    cmd = parse_cmd_args()
    cfg.set_config(*cmd.config)

    # Identify what to load: time period for raw data and which cyclone tracks.
    # This depends first on whether the options --dt-{start,end} are set:
    #
    if dt(*cmd.dt_start) > dt(1950, 1, 1) or dt(*cmd.dt_end) > dt(1950, 1, 1):

        # No specific VRILE to load => set region, rank, VRILE indices to NaN:
        cmd.region = np.nan
        cmd.rank   = np.nan
        vrile_id   = np.nan

        # Set the start/end dates; add 1 day to dt_end because cmd argument
        # specifies a day, but by default dt hour == minute == 0, so this
        # ensures we include all of cmd.dt_end:
        dt_start = dt(*cmd.dt_start)
        dt_end   = dt(*cmd.dt_end) + timedelta(days=1)

        # Not loading based on a VRILE, therefore no automatic tracks:
        track_ids_auto = []

    else:

        # Load the VRILE data for specified region only:
        vriles_cice = cache.load("vriles_cice_1979-2023.pkl")[cmd.region]

        # Identify the VRILE index based on the specified rank:
        vrile_id = np.argmin(abs(vriles_cice["vriles_joined_rates_rank"] - cmd.rank))

        # Total number of VRILEs for this sector (used in figure name):
        n_vriles_r = vriles_cice["n_joined_vriles"]

        # Set the start/end dates to be that of the specified VRILE:
        dt_start = vriles_cice["date_bnds_vriles_joined"][vrile_id][0]
        dt_end = vriles_cice["date_bnds_vriles_joined"][vrile_id][1]

        # Determine which tracks to load, if any, based on those 'matching' the
        # VRILE. Load the big list of all tracks, and then the vriles_cice
        # dictionary contains the key 'track_indices' containing the indices of
        # the former where matches occur:
        track_data = cache.load("tracks_filtered_1979-2023.pkl")

        # Here, track_data[2] is a list of all filtered tracks for the whole
        # time period, and vriles_cice['track_indices'] are indices of that
        # list that match VRILE with region index r and VRILE index v:
        track_ids_auto = [track_data[2][x]
                          for x in vriles_cice["track_indices"][vrile_id]]

        # Remove any such tracks if specified from command line:
        for trk_id in cmd.rm_auto_tracks:
            if trk_id in track_ids_auto:
                del track_ids_auto[track_ids_auto.index(trk_id)]

    # Load and prepare track data. First append any manually-selected tracks to
    # the list of track IDs to load. Put the net list of tracks to load into a
    # separate list, as it is helpful to distinguish those loaded manually vs.
    # automatically on the command line (there is a print block further down):
    track_ids = list(cmd.add_tracks) + track_ids_auto

    # Keep the track datetime coordinates, longitudes and latitudes, and
    # vorticities, for each track, in lists of length n_tracks:
    n_tracks  = len(track_ids)
    dts_track = []  # will contain arrays of varying length
    lon_track = []  # ""
    lat_track = []  # ""
    vor_track = []  # ""

    for trk in range(n_tracks):

        date_track_trk, lon_track_trk, lat_track_trk, vor_track_trk = \
            tracks.get_track_data(track_ids[trk], dt_start.year)

        # Filter track data to required datetime range:
        j_time_trk = [(x>=dt_start) and (x<=dt_end) for x in date_track_trk]

        dts_track.append(date_track_trk[j_time_trk])
        lon_track.append(lon_track_trk[j_time_trk])
        lat_track.append(lat_track_trk[j_time_trk])
        vor_track.append(vor_track_trk[j_time_trk])

    # Determine when (time axis indices) to plot quivers/vectors, and also
    # when to plot a marker for cyclone tracks (if at all):
    if cmd.qv_plot_mean:
        # Plot as mean over the VRILE or manually-determined time period:
        dt_qv_start = dt(dt_start.year, dt_start.month, dt_start.day)
        dt_qv_end   = dt(dt_end.year, dt_end.month, dt_end.day)
        dt_qv_end  += timedelta(days=1)
        jt_trk_marker = [None for trk in range(n_tracks)]  # no markers to plot
    else:
        # If there are tracks, plot the daily mean on the day at which
        # vorticity is a maximum (for the track with highest peak vorticity if
        # there is more than one track). Otherwise, plot on dt_start:
        if n_tracks > 0:
            # Determine datetime (dt) of maximum vorticity among all tracks:
            j_vor_max = np.argmax([np.max(vor_track[k]) for k in range(n_tracks)])
            dt_max_vor = dts_track[j_vor_max][np.argmax(vor_track[j_vor_max])]

            dt_qv_start = dt(dt_max_vor.year, dt_max_vor.month, dt_max_vor.day)

        else:
            dt_qv_start = dt_start

        # Add 1 day to get dt_qv_end so that daily mean is computed:
        dt_qv_end = dt_qv_start + timedelta(days=1)

        # Account for any manual offsetting:
        dt_qv_start += timedelta(days=cmd.qv_days_offset)
        dt_qv_end   += timedelta(days=cmd.qv_days_offset)

        # For the track markers, plot the nearest time available
        # for 12:00 on this day:
        jt_trk_marker = []
        for k in range(n_tracks):
            # Find time index of this track nearest to 12:00 on qv day:
            deltas = [abs((dt_qv_start + timedelta(hours=12) - i).total_seconds())
                      for i in dts_track[k]]
            jt_trk_marker.append(np.argmin(deltas))

        # Ad-hoc correction for case study 3: the cyclone track is split into 2.
        # It happens that the last time step of the first track is 12:00 18 Aug 2022
        # which is the date we want and will be returned by the above code for the
        # first element. However it will also give the first index of the second
        # part of the track, which is 06:00 on 19 Aug 2022, which we do not want to
        # plot. So here, just remove that second index:
        if dt_start.year == 2022 and dt_start.month == 8:
            jt_trk_marker[1] = None  # understood by maps.add_cyclone_tracks() function

    # Print summary of what is going to be plotted:
    print(f"VRILE ID                  : ", end="")
    if np.isnan(vrile_id):
        print("-")
    else:
        print(cfg.reg_labels_long[cmd.region] + f", rank = {cmd.rank} of "
              + f"{n_vriles_r} (ID = {vrile_id})")
    print(f"Date start                : {dt_start.strftime('%d %b %Y')}")
    print(f"Date end                  : {dt_end.strftime('%d %b %Y')}")
    print(f"Cyclone track IDs (auto.) : " + ", ".join(track_ids_auto))
    print(f"Cyclone track IDs (manual): " + ", ".join(cmd.add_tracks))
    print(f"Date start (vectors)      : {dt_qv_start.strftime('%HZ %d %b %Y')}")
    print(f"Date end (vectors)        : {dt_qv_end.strftime('%HZ %d %b %Y')}")

    print("\nLoading CICE coordinate data")
    ulon, ulat, tlon, tlat = cice.get_grid_data(["ULON", "ULAT", "TLON", "TLAT"],
                                                slice_to_atm_grid=False)

    # Save 2D data into a dictionary where the keys are choices for --panels-pcm:
    data_2d = dict()

    print("Loading CICE history data")
    # Only need the second return value (first is datetime), and extract the
    # first and only element of the list (corresponds to aice_d):
    aice_cice = cice.get_history_data(["aice_d"], dt_min=dt_start,
                                      dt_max=dt_end, frequency="daily",
                                      set_miss_to_nan=True,
                                      slice_to_atm_grid=False)[1][0]

    aice_cice = np.where(aice_cice > 1., np.nan, aice_cice)
    aice_cice = np.where(aice_cice < 0., np.nan, aice_cice)

    # This time-varying mask is used when integrating volume tendencies due to
    #(thermo)dynamics (for consistency with classification metric):
    cice_extent_mask = np.where(aice_cice < .15, 0., 1.)

    # Assign change in sea ice concentration:
    data_2d["daidt"] = aice_cice[-1,:,:] - aice_cice[0,:,:]
    data_2d["daidt"] = np.where(np.isnan(data_2d["daidt"]), 0., data_2d["daidt"])

    # Separate load for wind-stress components (different/shorter time span):
    _, data_cice = cice.get_history_data(["strairx_d", "strairy_d"],
        dt_min=dt_qv_start, dt_max=dt_qv_end, frequency="daily",
        set_miss_to_nan=True, slice_to_atm_grid=False)

    strairx = data_cice[0]
    strairy = data_cice[1]

    # Prepare the wind-stress components: rotate from CICE grid to regular,
    # lon-lat-based u/v components (deal with map projection later):
    strairx, strairy = rotate_vectors(strairx, strairy)
    strairx = np.mean(strairx, axis=0)  # time average
    strairy = np.mean(strairy, axis=0)  # ""

    # Select every nw = cmd.qv_sample spatial points (otherwise vectors are
    # too close together):
    nw = cmd.qv_sample
    lon_strair = ulon[nw::nw,nw::nw]
    lat_strair = ulat[nw::nw,nw::nw]
    strairx = strairx[nw::nw,nw::nw]
    strairy = strairy[nw::nw,nw::nw]

    print("Loading CICE processed data")
    # Only need the second return value (first is datetime), and extract the
    # first and only element of the list (corresponds to div_strair), and put
    # in units of 1e-7 N m-3. Assign this immediately to the data_2d dictionary
    # but need to also to calculate mean (done afterwards):
    data_2d["div_strair"] = cice.get_processed_data("div_curl",
        ["div_strair_d"], dt_min=dt_start, dt_max=dt_end, frequency="daily",
        slice_to_atm_grid=False)[1][0]*1.E7

    data_2d["div_strair"] = np.nanmean(data_2d["div_strair"], axis=0)

    # Detrended history data: always load detrended volume tendencies (as these
    # enter into the VRILE classification index):
    _, data_proc = cice.get_processed_data("hist_detrended",
        [f"detrended_{x}_d"
         for x in ["dvidtt", "dvidtd", "meltb", "meltl", "meltt","seb_ai"]],
        dt_min=dt_start, dt_max=dt_end, frequency="daily",
        slice_to_atm_grid=False)

    # Integrate/average these terms using the sea ice extent mask:
    for j, key in zip(range(5), ["dvidtt", "dvidtd", "meltb", "meltl", "meltt"]):
        data_2d[key] = np.nansum(data_proc[j]*cice_extent_mask, axis=0)

    # Convert integrated melt* terms into integrated volume changes:
    for x in "blt":
        data_2d[f"melt{x}"] *= -1.

    data_2d["seb_ai"] = np.nanmean(data_proc[5]*cice_extent_mask, axis=0)

    print("Loading atmospheric data")
    nc_vars = [f"detrended_{x}_d" for x in ["t2", "qlw", "qsw", "qnet"]]
    data_atmo = atmo.get_atmospheric_data_time_averages(nc_vars,
        freq="daily", dt_min=dt_start, dt_max=dt_end, auto_units=True,
        nc_file_prefix="atmo_detrended", slice_to_cice_grid=False)

    alon            = data_atmo[1]  # data_atmo[0] is datetime
    alat            = data_atmo[2]
    data_2d["t2"]   = np.nanmean(data_atmo[3], axis=0)
    data_2d["qlw"]  = np.nanmean(data_atmo[4], axis=0)
    data_2d["qsw"]  = np.nanmean(data_atmo[5], axis=0)
    data_2d["qnet"] = np.nanmean(data_atmo[6], axis=0)

    print("Loading atmospheric forcing data (wind vector components)")
    _, lon_wind, lat_wind, u10 = atmo.get_atmospheric_forcing_data("u10",
        dt_min=dt_qv_start, dt_max=dt_qv_end, slice_to_cice_grid=False,
        auto_units=True)

    v10 = atmo.get_atmospheric_forcing_data("v10", dt_min=dt_qv_start,
        dt_max=dt_qv_end, slice_to_cice_grid=False, auto_units=True)[-1]

    u10, v10 = rotate_vectors(u10, v10)
    u10 = np.mean(u10, axis=0)
    v10 = np.mean(v10, axis=0)

    lon_wind = lon_wind[nw::nw,nw::nw]
    lat_wind = lat_wind[nw::nw,nw::nw]
    u10 = u10[nw::nw,nw::nw]
    v10 = v10[nw::nw,nw::nw]

    ssmi_long_name = ("Bootstrap" if cmd.which_ssmi_dataset == "bt"
                      else "NASA Team")

    print(f"Loading SSM/I {ssmi_long_name} data")
    _, coord_vars_ssmi, data_ssmi = ssmi.load_data("aice", frequency="daily",
        which_dataset=cmd.which_ssmi_dataset, dt_range=(dt_start, dt_end))

    lon_obs = coord_vars_ssmi[0]
    lat_obs = coord_vars_ssmi[1]

    data_2d["ssmi"] = data_ssmi[0][-1,:,:] - data_ssmi[0][0,:,:]
    data_2d["ssmi"] = np.where(np.isnan(data_2d["ssmi"]), 0, data_2d["ssmi"])


    # Set plotting parameters -- mostly hardcoded for simplicity.
    #
    # Levels for pcolormesh/colorbars:
    levels = {}
    levels["daidt"]      = np.arange(-.9, .901, .2)
    levels["div_strair"] = np.arange(-4.5, 4.501, 1.)
    levels["dvidtd"]     = np.arange(-36., 36.001, 8.)
    levels["dvidtt"]     = levels["dvidtd"]
    levels["qlw"]        = np.arange(-50., 50.01, 10.)
    levels["qsw"]        = np.arange(-90., 90.01, 20.)
    levels["qnet"]       = levels["qsw"]
    levels["seb_ai"]     = levels["qlw"]
    levels["t2"]         = np.arange(-4.5, 4.501, 1.)
    levels["meltb"]      = np.arange(-18., 18.01, 4.)
    levels["meltl"]      = levels["meltb"]
    levels["meltt"]      = levels["meltb"]

    # Format of colorbars: the ticks/tick labels
    # (should be consistent with 'levels' above):
    cbar_tick_labels = {}
    cbar_tick_labels["daidt"]      = [-.9, -.5, .5, .9]
    cbar_tick_labels["div_strair"] = [-4.5, -2.5, 2.5, 4.5]
    cbar_tick_labels["dvidtd"]     = [-36., -20.,  20., 36.]
    cbar_tick_labels["dvidtt"]     = cbar_tick_labels["dvidtd"]
    cbar_tick_labels["qlw"]        = [-50., -30., 30., 50.]
    cbar_tick_labels["qsw"]        = [-90., -50., 50., 90.]
    cbar_tick_labels["qnet"]       = cbar_tick_labels["qsw"]
    cbar_tick_labels["seb_ai"]     = cbar_tick_labels["qlw"]
    cbar_tick_labels["t2"]         = [-4.5, -2.5, 2.5, 4.5]
    cbar_tick_labels["meltb"]      = [-18., -10., 10., 18.]
    cbar_tick_labels["meltl"]      = cbar_tick_labels["meltb"]
    cbar_tick_labels["meltt"]      = cbar_tick_labels["meltb"]

    # Float format for colorbar tick labels. Start by setting a default
    # then update as required:
    cbar_tick_labels_fmt = dict.fromkeys(cbar_tick_labels.keys(), "{:.0f}")
    
    for x in ["daidt", "div_strair", "t2"]:
        cbar_tick_labels_fmt[x] = "{:.1f}"

    # Keyword arguments to plt.pcolormesh() for each variable:
    pcm_kw = {}
    for k in cbar_tick_labels.keys():
        pcm_kw[k] = maps.pcm_kw(levels[k], style_ms.cmap[k])

    # Set SSMI to be the the same as daidt:
    pcm_kw["ssmi"] = pcm_kw["daidt"]
    cbar_tick_labels["ssmi"] = cbar_tick_labels["daidt"]
    cbar_tick_labels_fmt["ssmi"] = cbar_tick_labels_fmt["daidt"]

    # Keyword arguments to plt.quiver() for u10 and div_strair:
    qv_kw = {"u10":    {"scale"      : cmd.qv_adjust_factor * 70.,
                        "scale_units": "inches",
                        "color"      : "k",
                        "width"      : .0075,
                        "zorder"     : maps.get_zorder("vectors_atm")},
             "strair": {"scale"      : cmd.qv_adjust_factor * .5,
                        "scale_units": "inches",
                        "color"      : "k",
                        "width"      : .0075,
                        "zorder"     : maps.get_zorder("vectors_atm")}}

    # Values for quiver keys (determines size of arrows in key):
    qk_values = {"u10"   : cmd.qv_adjust_factor * 20.,
                 "strair": cmd.qv_adjust_factor * .2}

    # Labels for quiver keys:
    qk_labels = {"u10"   : r"%.0f m s$^{-1}$" % qk_values["u10"],
                 "strair": r"%.1f N m$^{-2}$" % qk_values["strair"]}

    # Transform lon/lat to north polar stereographic (x,y)
    # Arguments passed to posterproxy lonlat_to_xy_npsp() function:
    psp_xy_kw = {"extent_latitude"  : cmd.extent_latitude,    # default 70.
                 "central_longitude": cmd.central_longitude,  # default  0.
                 "xy_offset"        : cmd.xy_offset}          # default (0,0)

    # Vector components need to be rotated from east/west to dx/dy:
    uv = {"u10"   : psp.rotate_vectors_npsp(lon_wind  , u10    , v10),
          "strair": psp.rotate_vectors_npsp(lon_strair, strairx, strairy)}

    # Coordinates in (x,y) space:
    xy = dict.fromkeys(["daidt", "dvidtd", "dvidtt", "meltb", "meltl", "meltt",
                        "seb_ai", "div_strair"],
                       psp.lonlat_to_xy_npsp(ulon, ulat, **psp_xy_kw))

    xy["ssmi"] = psp.lonlat_to_xy_npsp(lon_obs, lat_obs, **psp_xy_kw)

    xy.update(dict.fromkeys(["qlw", "qsw", "qnet", "t2"],
                            psp.lonlat_to_xy_npsp(alon, alat, **psp_xy_kw)))

    xy["u10"]    = psp.lonlat_to_xy_npsp(lon_wind  , lat_wind  , **psp_xy_kw)
    xy["strair"] = psp.lonlat_to_xy_npsp(lon_strair, lat_strair, **psp_xy_kw)

    # For ice edge contours (use CICE T grid):
    xt, yt = psp.lonlat_to_xy_npsp(tlon, tlat, **psp_xy_kw)

    # Track coordinates in (x,y) space (list of x and of y for each track ID):
    x_trk = []
    y_trk = []
    for j in range(n_tracks):
        xk, yk = psp.lonlat_to_xy_npsp(lon_track[j], lat_track[j], **psp_xy_kw)
        x_trk.append(xk)
        y_trk.append(yk)

    # ----------------------------------------------------------------------- #

    fig, axs = plt.subplots(ncols=3, nrows=2,
                            figsize=(   style_ms.fig_width_double_column,
                                     .8*style_ms.fig_width_double_column ))

    psp.prepare_axes(axs)  # prepare for map plotting in psp (must come first)

    # Align the subplots, also allowing space for colorbar (cbar) axes
    # [must come after psp.prepare_axes() as that repositions axes]:
    tools.distribute_subplots(axs, l=.015, r=.015, t=.035, b=.095,
                                   s_hor=0., s_ver=.14)

    # Now flatten the 2x3 array of axs to simplify loops/indexing
    # (must come after distribute_subplots() which relies on 2x3 structure):
    axs = axs.flatten()

    # Create axes for horizontal colorbars at the bottom of each panel:
    cbar_axs    = []    # list of cbar axes (not cbars themselves)
    cbar_dy0    = .04   # space from bottom of axis to cbar top (figure units)
    cbar_height = .018  # height of cbars (figure units)

    for j in range(len(axs)):
        # Argument to fig.add_axes() is [lower left, lower right, width, height]
        # coordinates of the new axes to create in figure units:
        cbar_axs.append(fig.add_axes([
            axs[j].get_position().x0,
            axs[j].get_position().y0 - abs(cbar_dy0) - cbar_height,
            axs[j].get_position().width, cbar_height]))

    cbar_axs = np.array(cbar_axs)

    # Add titles/labels to subplots and colorbars:
    tools.add_subplot_panel_titles(axs,
                                   titles=[ax_titles[k] for k in cmd.panels_pcm],
                                   title_kw={"pad": 4, "zorder": 10000})

    tools.add_subplot_panel_titles(cbar_axs,
                                   titles=[cbar_titles[k] for k in cmd.panels_pcm],
                                   add_panel_labels=False,
                                   title_kw={"loc": "center", "pad": 4})

    # Note: zorders are set in the wrapper functions so the order of plotting
    # different elements should not generally matter (see maps.py)
    maps.psp_xy_land_overlay(axs, psp_xy_kw)
    maps.psp_xy_gridlines(axs[[x != "sectors" for x in cmd.panels_pcm]], psp_xy_kw)
    maps.psp_xy_gridlines(axs[[x == "sectors" for x in cmd.panels_pcm]], psp_xy_kw,
                          lat_labels=False)

    # Main plots (pcolormesh and vectors):
    for j in range(len(axs)):

        key_j = cmd.panels_pcm[j]

        if key_j == "-":
            # => remove panel (remove artist from figure but do not change axs or
            #    cbar_axs arrays so as to not mess up loop indexing):
            axs[j].remove()
            cbar_axs[j].remove()
        elif key_j == "sectors":
            cbar_axs[j].remove()  # no colorbar needed
            label_seas_on_axes(axs[j])
            label_sector_definitions(axs[j], psp_xy_kw)
        else:
            axs[j].set_rasterization_zorder(maps.get_zorder("rasterization"))

            pcm_j = axs[j].pcolormesh(xy[key_j][0], xy[key_j][1],
                                      data_2d[key_j][1:,1:], **pcm_kw[key_j])

            maps.set_colorbars([pcm_j], [cbar_axs[j]],
                               [cbar_tick_labels[key_j]],
                               cbar_tick_labels_fmt=[cbar_tick_labels_fmt[key_j]])

        key_j = cmd.panels_qv[j]

        if key_j != "-":
            # Note that zorder of quiver key is overridden by the zorder of the
            # quiver itself (so need to make sure they are above land)
            qv_j = axs[j].quiver(xy[key_j][0], xy[key_j][1], uv[key_j][0],
                                 uv[key_j][1], **qv_kw[key_j])

            maps.add_quiver_keys([axs[j]], [qv_j], [qk_values[key_j]],
                                 [qk_labels[key_j]])

    # Add ice edge change and track overlays where required:
    maps.add_ice_edge_contours(axs[[x==1 for x in cmd.panels_iel]],
                               xt, yt, aice_cice)
    maps.add_cyclone_tracks(axs[[x==1 for x in cmd.panels_tracks]],
                            x_trk, y_trk, jt_markers=jt_trk_marker)

    # Final part is to set figure metadata, window title for interactive use,
    # both containing some information about what is plotted, then show/save
    #
    if cmd.savefig_title is None:  # construct and use default metadata title
        if np.isnan(vrile_id):
            fig_title = "Cyclone case study, "  # presumably!
        else:
            fig_title = ("VRILE case study: " + cfg.reg_labels_long[cmd.region]
                         + f" sector, VRILE ID {vrile_id}, rank {cmd.rank} of "
                         + f"{n_vriles_r}, ")

        if dt_start.month == dt_end.month:
            fig_title += dt_start.strftime("%d") + u"\u2013"
        else:
            fig_title += dt_start.strftime("%d %b") + u" \u2013 "
        fig_title += dt_end.strftime("%d %b %Y")

    else:  # using custom title
        fig_title= cmd.savefig_title

    # For interactive use: create short descriptive string for figure
    # window title (this isn't saved with the figure):
    if np.isnan(vrile_id):
        descr_str = ""
    else:
        descr_str = f"{cfg.reg_labels_long[cmd.region]} VRILE {vrile_id}, "
        descr_str += f"rank {cmd.rank}/{n_vriles_r}, "

    if dt_start.month == dt_end.month:
        descr_str += dt_start.strftime("%d") + u"\u2013"
        descr_str += dt_end.strftime("%d %b %Y")

        if any([x != "-" for x in cmd.panels_qv]):
            if dt_qv_start.month == dt_start.month:
                descr_str += f" (qv: {dt_qv_start.strftime('%d')})"
            else:
                descr_str += f" (qv: {dt_qv_start.strftime('%d %b')})"

    else:
        descr_str += dt_start.strftime("%d %b") + u" \u2013 "
        descr_str += dt_end.strftime("%d %b %Y")

        if any([x != "-" for x in cmd.panels_qv]):
            descr_str += f" (qv: {dt_qv_start.strftime('%d %b')})"

    fig.canvas.manager.set_window_title(descr_str + " ")

    tools.finish_fig(fig, savefig=cmd.savefig, file_name=cmd.savefig_name,
                     set_raster_level=True, fig_metadata={"Title": fig_title})


if __name__ == "__main__":
    main()

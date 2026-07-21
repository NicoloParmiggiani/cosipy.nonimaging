from astropy.io import fits
import numpy as np
from datetime import datetime, timezone
import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

class ACSPrepareL2:

    """
    Utility class for reading, rebinning, validating,
    plotting, and exporting ACS/BGO light curves.
    """
    
    def __init__(self):
        """
        Initialize the preprocessing class.
        """
        pass

    def read_event_list_csv(self, csv_path):
        """
        Read the GRB event list from a CSV file.
        """

        columns = [
            "Event",
            "Start_time",
            "Peak_start_time",
            "Lightcurve_duration",
            "GBM_T90",
            "Peak_photon_flux_10_10000",
            "Peak_energy_flux_10_10000",
            "Peak_photon_flux_80_2000",
            "Zenith",
            "Azimuth",
            "Spectral_model",
            "Spectral_parameters"
        ]

        df = pd.read_csv(
            csv_path,
            sep=r'\s+(?![^\[]*\])',
            engine="python",
            names=columns,
            skiprows=1
        )

        return df

    def raed_acs_lc_from_hdf5(
        self,
        hdf5_path,
        events_df,
        start_ori,
        detector_list
    ):
        """
        Read ACS light curves from an HDF5 file.
        """

        results = []

        with h5py.File(hdf5_path, "r") as f:

            time_bins = f["time bins (s)"][:]

            for _, row in events_df.iterrows():

                name = row["Event"]

                t_start = row["Start_time"] + start_ori - 30
                t_end = t_start + 60

                # Select time bins inside the requested window
                mask = (time_bins >= t_start) & (time_bins <= t_end)
                indices = np.where(mask)[0]

                if len(indices) == 0:

                    print("No Data")

                    results.append({
                        "name": name,
                        "counts": None
                    })

                    continue

                selected_time = time_bins[indices]

                counts_per_detector = {}

                for det in detector_list:

                    data = f[det][indices, 0]
                    counts_per_detector[det] = data

                results.append({
                    "name": name,
                    "counts": counts_per_detector,
                    "time": selected_time,
                    "trigger_time": row["Start_time"] + start_ori,
                    "time_start": t_start,
                    "time_stop": t_end,
                    "theta": row["Zenith"],
                    "phi": row["Azimuth"]
                })

            print(f"Total evens processed: {len(results)}")

            return results

    def rebin_lightcurve(self, time, counts):
        
        """Rebin con bin variabili, usando tick interi da 50 ms."""

        base_dt = 0.050

        # Bordi espressi in unità da 50 ms
        ticks_1 = np.arange(-600, -200 + 20, 20)  # -30 -> -10, passo 1 s
        ticks_2 = np.arange(-200,  -20 + 5,  5)   # -10 -> -1, passo 250 ms
        ticks_3 = np.arange( -20,   80 + 1,  1)   # -1  -> 4,  passo 50 ms
        ticks_4 = np.arange(  80,  600 + 5,  5)   # 4   -> 30, passo 250 ms

        edge_ticks = np.concatenate([
            ticks_1,
            ticks_2[1:],
            ticks_3[1:],
            ticks_4[1:]
        ])

        # Shift: da [-30, 30] a [0, 60]
        edge_ticks = edge_ticks + 600

        # Quantizzazione dei tempi sulla griglia esatta da 50 ms
        time_ticks = np.rint(
            np.asarray(time, dtype=float) / base_dt
        ).astype(int)

        counts = np.asarray(counts)

        rebinned_counts, _ = np.histogram(
            time_ticks,
            bins=edge_ticks,
            weights=counts
        )

        n_samples, _ = np.histogram(
            time_ticks,
            bins=edge_ticks
        )

        edges = edge_ticks * base_dt
        dt = np.diff(edge_ticks) * base_dt
        t_centers = 0.5 * (edges[:-1] + edges[1:])

        return t_centers, rebinned_counts, dt, edges, n_samples
    
    def read_background_data(
        self,
        input_path,
        event_time,
        half_window=30.0,
    ):
        """
        Load background event times and select a time window around an event.

        Parameters
        ----------
        input_path : str or Path
            Path to the NPZ file containing one array for each BGO panel.

        event_time : float
            Absolute event time in seconds.

        half_window : float, optional
            Half-width of the selected time window in seconds.
            The default is 30 seconds.

        Returns
        -------
        dict
            Dictionary with keys z1, z0, x1, x0, y1, y0.
            Each value is a NumPy array containing the event times inside
            the interval:

            [event_time - half_window, event_time + half_window)
        """

        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        if half_window <= 0:
            raise ValueError("half_window must be greater than zero")

        panel_names = ["z1", "z0", "x1", "x0", "y1", "y0"]

        window_start = float(event_time) - float(half_window)
        window_end = float(event_time) + float(half_window)

        background_window = {}

        with np.load(input_path) as data:

            missing_panels = [
                panel
                for panel in panel_names
                if panel not in data.files
            ]

            if missing_panels:
                raise KeyError(
                    f"Missing panels in {input_path.name}: {missing_panels}"
                )

            for panel in panel_names:

                times = np.asarray(
                    data[panel],
                    dtype=float,
                )

                mask = (
                    np.isfinite(times)
                    & (times >= window_start)
                    & (times < window_end)
                )

                selected_times = times[mask]
                background_window[panel] = selected_times

                print(
                    f"{panel}: {selected_times.size:,} events "
                    f"between t = {window_start:.2f} s "
                    f"and t = {window_end:.2f} s"
                )

        return background_window

 
    def read_grb_data_from_sim(self,grb_path, event_time):
        """
        Read a GRB file and return the absolute count times
        for each BGO detector panel.

        Parameters
        ----------
        grb_path : str or Path
            Path to the GRB CSV file.

        event_time : float
            Absolute time used to shift the relative GRB timestamps.

        Returns
        -------
        dict
            Dictionary with the following keys:
            z1, z0, x1, x0, y1, y0.

            Each value is a NumPy array containing:
            event_time + timestamp[s]
        """

        grb_path = Path(grb_path)

        if not grb_path.exists():
            raise FileNotFoundError(f"File not found: {grb_path}")

        grb_df = pd.read_csv(grb_path)

        panel_columns = {
            "z1": "bgo_z1[keV]",
            "z0": "bgo_z0[keV]",
            "x1": "bgo_x1[keV]",
            "x0": "bgo_x0[keV]",
            "y1": "bgo_y1[keV]",
            "y0": "bgo_y0[keV]",
        }

        required_columns = ["timestamp[s]", *panel_columns.values()]
        missing_columns = [
            column for column in required_columns
            if column not in grb_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in {grb_path.name}: {missing_columns}"
            )

        timestamps = pd.to_numeric(
            grb_df["timestamp[s]"],
            errors="coerce",
        )

        grb_times_at_event = {}

        for panel, energy_column in panel_columns.items():
            energies = pd.to_numeric(
                grb_df[energy_column],
                errors="coerce",
            )

            valid_events = energies.gt(0) & timestamps.notna()
            relative_times = timestamps.loc[valid_events].to_numpy(dtype=float)

            grb_times_at_event[panel] = float(event_time) + relative_times

        return grb_times_at_event



    def plot_lc(self,lc_data,detector_name):
     
        det = detector_name

        # --- ORIGINAL ---
        time = lc_data["time"]
        counts = lc_data["counts"][det]

        T0 = lc_data["trigger_time"]
        time_rel = time - T0

        dt_original = 0.05
        rate_original = counts / dt_original

        # --- REBINNED ---
        c_reb = lc_data["counts_reb"][det]
        dt_reb = lc_data["dt_reb"]
        edges_reb = lc_data["edges_reb"]-30

        rate_reb = c_reb / dt_reb

        # --- PLOT ---
        plt.figure(figsize=(10,5))

        # originale
        plt.plot(
            time_rel,
            rate_original,
            alpha=0.35,
            color="gray",
            label="Original (50 ms)"
        )

        # rebinned CORRETTO
        plt.stairs(
            rate_reb,
            edges_reb,
            linewidth=2,
            color="red",
            label="Rebinned"
        )

        # linee per vedere i cambi di regime
        plt.axvline(-10, linestyle="--", alpha=0.4)
        plt.axvline(-1, linestyle="--", alpha=0.4)
        plt.axvline(4, linestyle="--", alpha=0.4)

        plt.xlabel("Time (s)")
        plt.ylabel("Counts / s")
        plt.title(lc_data["name"] + "")
        plt.legend()
        print(lc_data["name"])
        plt.show()
        
        
    def plot_raw_data(
        self,
        panel_times,
        event_time,
        bin_width=0.050,
        figsize=(14, 10),
        data_label="Events",
    ):
        """
        Plot event light curves for the six BGO panels.

        Parameters
        ----------
        panel_times : dict
            Dictionary with keys z1, z0, x1, x0, y1, y0.
            Each value contains the event timestamps for one panel.

        event_time : float
            Absolute reference event time.

        bin_width : float, optional
            Bin width in seconds. Default is 0.050 s.

        figsize : tuple, optional
            Figure size.

        data_label : str, optional
            Label used in the figure title. Default is "Events".

        Returns
        -------
        fig, axes, counts_per_detector
            Figure, axes, and binned counts for each panel.
        """

        panel_order = ["z1", "z0", "x1", "x0", "y1", "y0"]

        panel_data = {
            panel: np.asarray(panel_times.get(panel, []), dtype=float)
            for panel in panel_order
        }

        panel_data = {
            panel: times[np.isfinite(times)]
            for panel, times in panel_data.items()
        }

        valid_arrays = [
            times
            for times in panel_data.values()
            if times.size > 0
        ]

        if not valid_arrays:
            raise ValueError("No events available for plotting")

        all_times = np.concatenate(valid_arrays)

        t_min = np.floor(all_times.min() / bin_width) * bin_width
        t_max = np.ceil(all_times.max() / bin_width) * bin_width

        n_bins = int(np.ceil((t_max - t_min) / bin_width))

        bin_edges = t_min + np.arange(n_bins + 1) * bin_width
        bin_centers = bin_edges[:-1] + bin_width / 2.0

        counts_per_detector = {
            panel: np.histogram(times, bins=bin_edges)[0]
            for panel, times in panel_data.items()
        }

        fig, axes = plt.subplots(
            3,
            2,
            figsize=figsize,
            sharex=True,
        )

        for ax, panel in zip(axes.flat, panel_order):
            times = panel_data[panel]
            counts = counts_per_detector[panel]
            rate = counts / bin_width

            ax.step(
                bin_centers,
                rate,
                where="mid",
                linewidth=1,
            )

            ax.axvline(
                event_time,
                linestyle="--",
                linewidth=1,
                label="Event time",
            )

            ax.set_title(
                f"{panel.upper()} — {times.size:,} events"
            )
            ax.set_ylabel("Rate [counts/s]")
            ax.grid(alpha=0.3)

        for ax in axes[-1, :]:
            ax.set_xlabel("Absolute time [s]")

        fig.suptitle(
            f"{data_label} in the six BGO panels — "
            f"{bin_width * 1000:.0f} ms bins",
            fontsize=14,
        )

        fig.tight_layout()
        plt.show()

        return fig, axes, counts_per_detector

    def sanity_checks(self, acs_lc, detector_list, tol=1e-6):
        """Check count conservation after rebinning."""

        if acs_lc["counts"] is None:
            print("Sanity check fallito: counts è None")
            return False

        if "counts_reb" not in acs_lc:
            print("Sanity check fallito: counts_reb assente")
            return False

        all_ok = True

        for det in detector_list:
            original = np.sum(acs_lc["counts"][det])
            rebinned = np.sum(acs_lc["counts_reb"][det])
            diff = rebinned - original

            if not np.isclose(rebinned, original, rtol=tol, atol=tol):
                all_ok = False
                print(
                    f"{det}: check fallito | "
                    f"originale={original:.6f}, "
                    f"rebinned={rebinned:.6f}, "
                    f"diff={diff:.6f}"
                )

        return all_ok

    def plot_acs_lc_from_fits(
        self,
        filename,
        plot_counts=False,
        save=False
    ):
        """
        Plot ACS light curves from a FITS file.
        Figures can be saved to disk or shown on screen.
        """

        name = os.path.splitext(
            os.path.basename(filename)
        )[0]

        with fits.open(filename) as hdul:

            header = hdul[0].header
            data = hdul[1].data

            # Observation start time
            time_start = header["DATE-OBS"]

            print(f"Observation start: {time_start}")

            # Time bins
            t_min = data["T_MIN"]
            t_max = data["T_MAX"]

            # Bin center and width
            t_center = 0.5 * (t_min + t_max)
            dt = t_max - t_min

            # Detector panels
            panels = {
                "SCBA_A0": data["COUNTS_SCBA_A0_G"],
                "SCBA_A1": data["COUNTS_SCBA_A1_G"],
                "SCBB_A0": data["COUNTS_SCBB_A0_G"],
                "SCBB_A1": data["COUNTS_SCBB_A1_G"],
                "SCBC_A0": data["COUNTS_SCBC_A0_G"],
                "SCBC_A1": data["COUNTS_SCBC_A1_G"],
            }

            # =========================
            # Plot counts
            # =========================
            if plot_counts:

                fig_counts, axes = plt.subplots(
                    3,
                    2,
                    figsize=(12, 10),
                    sharex=True
                )

                axes = axes.flatten()

                for ax, (panel_name, counts) in zip(
                    axes,
                    panels.items()
                ):

                    ax.step(
                        t_center,
                        counts,
                        where="mid"
                    )

                    ax.set_title(panel_name)
                    ax.set_ylabel("Counts")

                axes[-1].set_xlabel("Time [s]")

                fig_counts.suptitle(
                    "Detector Counts",
                    fontsize=14
                )

                plt.tight_layout()

                if save:

                    counts_filename = (
                        f"acs_lc_counts_{name}.png"
                    )

                    fig_counts.savefig(
                        counts_filename,
                        dpi=300
                    )

                    plt.close(fig_counts)

                    print(f"Saved: {counts_filename}")

                else:
                    plt.show()

            # =========================
            # Plot rates
            # =========================
            fig_rate, axes = plt.subplots(
                3,
                2,
                figsize=(12, 10),
                sharex=True
            )

            axes = axes.flatten()

            for ax, (panel_name, counts) in zip(
                axes,
                panels.items()
            ):

                rate = counts / dt

                ax.step(
                    t_center,
                    rate,
                    where="mid",
                    color="b"
                )

                ax.set_title(panel_name)
                ax.set_ylabel("Counts/s")

            axes[-1].set_xlabel("Time [s]")

            fig_rate.suptitle(
                "Detector Rates",
                fontsize=14
            )

            plt.tight_layout()

            if save:

                rate_filename = (
                    f"acs_lc_rate_{name}.png"
                )

                fig_rate.savefig(
                    rate_filename,
                    dpi=300
                )

                plt.close(fig_rate)

                print(f"Saved: {rate_filename}")

            else:
                plt.show()

    def create_acs_lc_fits(self, acs_lc, output_dir):

        mjd_ref_timestamp = 1735689600.184
    
        name = acs_lc["name"]
        edges = np.asarray(acs_lc["edges_reb"], dtype=np.float64)
        trigger_time = float(acs_lc["trigger_time"])
        
        time_start = acs_lc['time_start']
        time_stop = acs_lc['time_stop']
        
        # tempi assoluti
        t_min = edges[:-1] + time_start - mjd_ref_timestamp
        t_max = edges[1:] + time_start - mjd_ref_timestamp
        
        deltas = t_max-t_min
        
        n = len(t_min)
        
        date_obs = datetime.fromtimestamp(time_start).strftime("%Y-%m-%dT%H:%M:%S")
        date_end = datetime.fromtimestamp(time_stop).strftime("%Y-%m-%dT%H:%M:%S")

        
        integration_time = time_stop - time_start

        obs_id = datetime.fromtimestamp(time_start).strftime("%Y%m%d")
        
        time_start_rel = time_start - mjd_ref_timestamp
        time_stop_rel = time_stop - mjd_ref_timestamp

        # =========================
        # COUNTS (mapping)
        # =========================
        cr = acs_lc["counts_reb"]

        x0 = np.asarray(cr["x0"], dtype=np.int16)
        x1 = np.asarray(cr["x1"], dtype=np.int16)
        y0 = np.asarray(cr["y0"], dtype=np.int16)
        y1 = np.asarray(cr["y1"], dtype=np.int16)
        z0 = np.asarray(cr["z0"], dtype=np.int16)
        z1 = np.asarray(cr["z1"], dtype=np.int16)
        
        final_counts = np.column_stack([
            z1,
            z0,
            y1,
            y0,
            x1,
            x0,
        ]).astype(np.int16)

        zeros = np.zeros(n, dtype=np.int16)
        zeros_d = np.zeros(n, dtype=np.float64)

        creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        # =========================
        # PRIMARY HDU
        # =========================
        primary = fits.PrimaryHDU()
        phdr = primary.header

        phdr["TELESCOP"] = ("COSI", "Telescope mission name")
        phdr["INSTRUME"] = ("ACS", "Instrument name")
        phdr["OBS_ID"] = (obs_id, "Observation ID")
        phdr["DATE-OBS"] = (date_obs, "Start Date")
        phdr["DATE-END"] = (date_end, "Stop Date")
        phdr["ORIGIN"] = ("SSL", "Origin of the FITS file")
        phdr["DATE"] = (creation_date, "File creation date")
        phdr["CREATOR"] = ("rebinned_pipeline", "Software that create 1st the file")

        # =========================
        # TABLE
        # =========================
        cols = [
            fits.Column(name="TIME", format="1D", unit="s", array=t_min),

            fits.Column(
                name="TIMEDEL",
                format="6D",
                unit="s",
                array=deltas
            ),

            fits.Column(
                name="COUNT",
                format="6I",
                unit="count",
                array=final_counts
            ),
        ]

        table = fits.BinTableHDU.from_columns(cols)
        hdr = table.header
        
        hdr['TTYPE1']  = ('TIME', 'Mission Time in sec since 01 Jan 2025 00:00:00')
        hdr['TFORM1']  = ('1D', 'data format, double precision floating pt')
        hdr['TUNIT1']  = ('s', 'physical unit of field')

        hdr['TTYPE2']  = ('TIMEDEL', 'deadtime corrected length of the integrated bin, for each ASIC')
        hdr['TFORM2']  = ('6D', 'data format, vector of double precision floating pt')
        hdr['TUNIT2']  = ('s', 'physical unit of field')

        hdr['TTYPE3']  = ('COUNT', 'Counts from 6 ACS ASICs')
        hdr['TFORM3']  = ('6I', 'data format, 16-bit integer')
        hdr['TUNIT3']  = ('count', 'physical unit of field')   

        # =========================
        # HEADER EXTENSION
        # =========================
        hdr["EXTNAME"] = ("ACS_LC", "name of this HDU")
        hdr["TELESCOP"] = ("COSI", "Telescope mission name")
        hdr["INSTRUME"] = ("ACS", "Instrument  name")
        hdr["CHANTYPE"] = ("GAMMA","PROTON or GAMMA channel")
        hdr["OBS_ID"] = (obs_id, "Observation ID")
        
        hdr["MJDREFI"] = (60676, "MJD reference day 01 Jan 2025 00:00:00")
        hdr["MJDREFF"] = (8.007407407407E-04, "MDJ reference (fraction of day)")
        
        hdr["TIMEREF"] = ("LOCAL", "Reference Frame")
        hdr["TASSIGN"] = ("SATELLITE", "Time assigned")
        hdr["TIMESYS"] = ("TT", "Time System")
        hdr["TIMEUNIT"] = ("s", "Time unit for timing header keywords")
        hdr["TIMPIXR"] = (0.0, "TIME parameter reflects bin start (0), bin end (1) or bin center (2)")

        hdr["CLOCKAPP"] = (False, "If clock corrections are applied (T/F)")
        hdr["DEADAPP"] = (False, "If deadtime correction is applied to TIMEDEL")
        hdr["DATE-OBS"] = (date_obs, "Start Date")
        hdr["DATE-END"] = (date_end, "Stop Date")
        hdr["TSTART"] = (time_start, "Start Time")
        hdr["TSTOP"] = (time_stop, "Stop Time")
        
        hdr["E_MIN"] = (80, "/[keV] Energy Minimum")
        hdr["E_MAX"] = (2000, "/[keV] Energy Maximum")

        hdr["HDUCLASS"] = ("OGIP", "format conforms to OGIP standard")
        hdr["HDUCLAS1"] = ("TEMPORALDATA", "hduclass1")
        hdr["HDUCLAS2"] = ("EVRATE", "hduclass2")

        hdr["CREATOR"] = ("Eliza_Sim", "Software that create 1st the file")
        hdr["PROCVER"] = ("00.00.00.00", "Processing version")
        hdr["CALDBVER"] = ("cs20230401", "CALDB index version used")
        hdr["SEQPNUM"] = (1, "Times the dataset has been processed")

        hdr["ORIGIN"] = ("SSL", "Origin of the FITS files")
        hdr["DATE"] = creation_date

        # =========================
        # GTI EXTENSION (vuota)
        # =========================
        gti_start = np.array([time_start_rel], dtype=np.float64)
        gti_stop  = np.array([time_stop_rel], dtype=np.float64)

        gti_cols = [
            fits.Column(
                name="START",
                format="1D",
                unit="s",
                array=gti_start,
            ),
            fits.Column(
                name="STOP",
                format="1D",
                unit="s",
                array=gti_stop,
            ),
        ]

        gti_hdu = fits.BinTableHDU.from_columns(gti_cols)
        ghdr = gti_hdu.header

        ghdr["EXTNAME"] = ("GTI", "Binary table extension name")
        ghdr["HDUCLASS"] = ("OGIP", "Format conforms to OGIP/GSFC standards")
        ghdr["HDUCLAS1"] = ("GTI", "First class level")
        ghdr["HDUCLAS2"] = ("STANDARD", "Second class level")

        ghdr["TELESCOP"] = ("COSI", "Telescope (mission) name")
        ghdr["INSTRUME"] = ("ACS", "Instrument name")
        ghdr["CHANTYPE"] = ("GAMMA","PROTON or GAMMA channel")
        ghdr["DATAMODE"] = ("", "Instrument datamode")
        ghdr["OBSERVER"] = ("John Tomsick", "Principal Investigator")
        ghdr["OBS_ID"] = (obs_id, "Observation ID")
        ghdr["OBJECT"] = ("", "Object/Target name")

        ghdr["TIMESYS"] = ("TT", "Reference Time System")
        ghdr["MJDREFI"] = (60676, "MJD reference day 1 Jan 2025 00:00:00")
        ghdr["MJDREFF"] = (8.007407407407E-04, "MJD reference fraction of day")
        ghdr["TIMEREF"] = ("LOCAL", "Reference Frame")
        ghdr["TASSIGN"] = ("SATELLITE", "Time assigned by clock")
        ghdr["TIMEUNIT"] = ("s", "Time unit for timing header keyword")

        ghdr["TSTART"] = (time_start, "[s] Observation start time")
        ghdr["TSTOP"] = (time_stop, "[s] Observation stop time")

        ghdr["DATE-OBS"] = (date_obs, "Start date of observations")
        ghdr["DATE-END"] = (date_end, "End date of observations")
        ghdr["CLOCKAPP"] = (False, "Clock correction applied ?")

        ghdr["ORIGIN"] = ("SSL", "Origin of fits file")
        ghdr["PROCVER"] = ("00.00.00.00", "Processing script version number")
        ghdr["CALDBVER"] = ("cs20230401", "CALDB index version used")
        ghdr["CREATOR"] = ("rebinned_pipeline", "Software creator of the file")
        ghdr["DATE"] = (creation_date, "File creation date")

        # =========================
        # WRITE
        # =========================
        filename = os.path.join(output_dir, f"{name}.fits")

        hdul = fits.HDUList([primary, table,gti_hdu])
        hdul.writeto(filename, overwrite=True, checksum=True)

        print(f"created: {filename}")
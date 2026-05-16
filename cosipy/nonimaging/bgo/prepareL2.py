from astropy.io import fits
import numpy as np
from datetime import datetime, timezone
import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt

class BGOPrepareL2:

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
        """
        Rebin a light curve using variable-width bins.
        """

        edges_1 = np.arange(-30, -10 + 1.0, 1.0)
        edges_2 = np.arange(-10, -1 + 0.25, 0.25)
        edges_3 = np.arange(-1, 4 + 0.05, 0.05)
        edges_4 = np.arange(4, 30 + 0.25, 0.25)

        # Merge edges without duplicating common boundaries
        edges = np.concatenate([
            edges_1,
            edges_2[1:],
            edges_3[1:],
            edges_4[1:]
        ])

        edges = edges + 30
        
        rebinned_counts, _ = np.histogram(
            time,
            bins=edges,
            weights=counts
        )

        n_samples, _ = np.histogram(time, bins=edges)

        dt = np.diff(edges)
        t_centers = 0.5 * (edges[:-1] + edges[1:])

        return t_centers, rebinned_counts, dt, edges, n_samples

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

    def sanity_checks(self, acs_lc, detector_list, tol=1e-6):
        """
        Check count conservation after rebinning.
        """

        # Preliminary check
        if acs_lc["counts"] is None or "counts_reb" not in acs_lc:
            print("no counts or counts rebinned")
            return False

        all_ok = True

        for det in detector_list:

            original_sum = np.sum(
                acs_lc["counts"][det]
            )

            rebinned_sum = np.sum(
                acs_lc["counts_reb"][det]
            )

            diff = rebinned_sum - original_sum
            
            # Compare original and rebinned counts
            ok = np.isclose(
                rebinned_sum,
                original_sum,
                rtol=tol
            )

            if not ok:
                all_ok = False

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

        mjd_ref_timestamp = 1735689669.184

        name = acs_lc["name"]
        edges = np.asarray(acs_lc["edges_reb"], dtype=np.float64)
        trigger_time = float(acs_lc["trigger_time"])

        time_start = acs_lc['time_start']
        time_stop = acs_lc['time_stop']

        # Absolute times
        t_min = edges[:-1] + time_start - mjd_ref_timestamp
        t_max = edges[1:] + time_start - mjd_ref_timestamp

        n = len(t_min)

        date_obs = datetime.fromtimestamp(time_start).strftime("%Y-%m-%dT%H:%M:%S")
        date_end = datetime.fromtimestamp(time_stop).strftime("%Y-%m-%dT%H:%M:%S")

        integration_time = time_stop - time_start
        obs_id = datetime.fromtimestamp(time_start).strftime("%Y%m%d")

        time_start_rel = time_start - mjd_ref_timestamp
        time_stop_rel = time_stop - mjd_ref_timestamp

        # Counts mapping
        cr = acs_lc["counts_reb"]

        x0 = np.asarray(cr["x0"], dtype=np.int16)
        x1 = np.asarray(cr["x1"], dtype=np.int16)
        y0 = np.asarray(cr["y0"], dtype=np.int16)
        y1 = np.asarray(cr["y1"], dtype=np.int16)
        z0 = np.asarray(cr["z0"], dtype=np.int16)
        z1 = np.asarray(cr["z1"], dtype=np.int16)

        zeros = np.zeros(n, dtype=np.int16)
        zeros_d = np.zeros(n, dtype=np.float64)

        creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        # Primary HDU
        primary = fits.PrimaryHDU()
        phdr = primary.header

        phdr["TELESCOP"] = ("COSI", "Telescope mission name")
        phdr["INSTRUME"] = ("ACS", "Instrument name")
        phdr["OBS_ID"] = (name, "Observation ID")
        phdr["DATE-OBS"] = (date_obs, "Start Date")
        phdr["DATE-END"] = (date_end, "Stop Date")
        phdr["ORIGIN"] = ("SSL", "Origin of the FITS file")
        phdr["DATE"] = (creation_date, "File creation date")
        phdr["CREATOR"] = ("rebinned_pipeline", "")

        # Table
        cols = []

        cols.append(fits.Column(name="T_MIN", format="1D", unit="s", array=t_min))
        cols.append(fits.Column(name="T_MAX", format="1D", unit="s", array=t_max))

        # SCB-A
        cols.append(fits.Column(name="COUNTS_SCBA_A0_G", format="1I", unit="count", array=z1))
        cols.append(fits.Column(name="COUNTS_SCBA_A0_P", format="1I", unit="count", array=zeros))
        cols.append(fits.Column(name="DEADTIME_SCBA_A0", format="1D", unit="s", array=zeros_d))

        cols.append(fits.Column(name="COUNTS_SCBA_A1_G", format="1I", unit="count", array=z0))
        cols.append(fits.Column(name="COUNTS_SCBA_A1_P", format="1I", unit="count", array=zeros))
        cols.append(fits.Column(name="DEADTIME_SCBA_A1", format="1D", unit="s", array=zeros_d))

        # SCB-B
        cols.append(fits.Column(name="COUNTS_SCBB_A0_G", format="1I", unit="count", array=y1))
        cols.append(fits.Column(name="COUNTS_SCBB_A0_P", format="1I", unit="count", array=zeros))
        cols.append(fits.Column(name="DEADTIME_SCBB_A0", format="1D", unit="s", array=zeros_d))

        cols.append(fits.Column(name="COUNTS_SCBB_A1_G", format="1I", unit="count", array=y0))
        cols.append(fits.Column(name="COUNTS_SCBB_A1_P", format="1I", unit="count", array=zeros))
        cols.append(fits.Column(name="DEADTIME_SCBB_A1", format="1D", unit="s", array=zeros_d))

        # SCB-C
        cols.append(fits.Column(name="COUNTS_SCBC_A0_G", format="1I", unit="count", array=x1))
        cols.append(fits.Column(name="COUNTS_SCBC_A0_P", format="1I", unit="count", array=zeros))
        cols.append(fits.Column(name="DEADTIME_SCBC_A0", format="1D", unit="s", array=zeros_d))

        cols.append(fits.Column(name="COUNTS_SCBC_A1_G", format="1I", unit="count", array=x0))
        cols.append(fits.Column(name="COUNTS_SCBC_A1_P", format="1I", unit="count", array=zeros))
        cols.append(fits.Column(name="DEADTIME_SCBC_A1", format="1D", unit="s", array=zeros_d))

        table = fits.BinTableHDU.from_columns(cols)
        hdr = table.header

        # Header extension
        hdr["EXTNAME"] = ("ACS_LC", "name of this HDU")
        hdr["TELESCOP"] = ("COSI", "Telescope mission name")
        hdr["INSTRUME"] = ("ACS", "Instrument  name")
        hdr["OBS_ID"] = (obs_id, "Observation ID")

        hdr["MJDREFI"] = (60676, "MJD reference day 01 Jan 2025 00:00:00")
        hdr["MJDREFF"] = (8.007407407407E-04, "MDJ reference fraction of day")

        hdr["TIMEREF"] = ("LOCAL", "Reference Frame")
        hdr["TASSIGN"] = ("SATELLITE", "Time assigned")
        hdr["TIMESYS"] = ("TT", "Time System")
        hdr["TIMEUNIT"] = ("s", "Time unit for timing header keywords")
        hdr["TIMEDEL"] = (integration_time, "Integration time")

        hdr["CLOCKAPP"] = (False, "If clock corrections are applied (T/F)")
        hdr["DATE-OBS"] = (date_obs, "Start Date")
        hdr["DATE-END"] = (date_end, "Stop Date")
        hdr["TSTART"] = (time_start, "Start Time")
        hdr["TSTOP"] = (time_start, "Stop Time")

        hdr["E_MIN"] = (80, "/[keV] Energy Minimum")
        hdr["E_MAX"] = (2000, "/[keV] Energy Maximum")

        hdr["HDUCLASS"] = ("OGIP", "format conforms to OGIP standard")
        hdr["HDUCLAS1"] = ("TEMPORALDATA", "hduclass1")
        hdr["HDUCLAS2"] = ("EVRATE", "hduclass2")

        hdr["CREATOR"] = ("Eliza_Sim", "Software that create 1st the file")
        hdr["PROCVER"] = ("1.0.0", "Processing version")
        hdr["CALDBVER"] = ("1.0.0", "CALDB version")
        hdr["SEQPNUM"] = (1, "Times the dataset has been processed")

        hdr["ORIGIN"] = ("INAF/Bologna", "Origin of the FITS files")
        hdr["DATE"] = creation_date

        # GTI extension
        gti_start = np.array([time_start_rel], dtype=np.float64)
        gti_stop = np.array([time_stop_rel], dtype=np.float64)

        gti_cols = [
            fits.Column(name="START", format="1D", unit="s", array=gti_start),
            fits.Column(name="STOP", format="1D", unit="s", array=gti_stop),
        ]

        gti_hdu = fits.BinTableHDU.from_columns(gti_cols)
        ghdr = gti_hdu.header

        ghdr["EXTNAME"] = ("GTI", "Binary table extension name")
        ghdr["HDUCLASS"] = ("OGIP", "Format conforms to OGIP/GSFC standards")
        ghdr["HDUCLAS1"] = ("GTI", "First class level")
        ghdr["HDUCLAS2"] = ("", "Second class level")

        ghdr["TELESCOP"] = ("COSI", "Telescope (mission) name")
        ghdr["INSTRUME"] = ("ACS", "Instrument name")
        ghdr["OBS_ID"] = (name, "Observation ID")

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

        ghdr["ORIGIN"] = ("INAF/Bologna", "Origin of fits file")
        ghdr["PROCVER"] = ("v1.0", "Processing script version number")
        ghdr["SOFTVER"] = ("Hea_ddmmmyyyy_v1.0", "Software version")
        ghdr["CALDBVER"] = ("coYYYYMMDD", "CALDB index version used")
        ghdr["CREATOR"] = ("rebinned_pipeline", "Software creator of the file")
        ghdr["DATE"] = (creation_date, "File creation date")

        # Write FITS file
        filename = os.path.join(output_dir, f"{name}.fits")

        hdul = fits.HDUList([primary, table, gti_hdu])
        hdul.writeto(filename, overwrite=True, checksum=True)

        print(f"created: {filename}")
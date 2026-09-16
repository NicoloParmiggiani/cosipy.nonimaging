import healpy as hp
from mhealpy.containers.healpix_map import HealpixMap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from astropy.io import fits
from gdt.core.background.binned import Polynomial
from gdt.core.data_primitives import TimeBins
from bctools.analysis import BayesianBlocksLightcurve
import numpy as np

class ACSDataAnalyzer:
    """
    Analyzer for ACS light curve data stored in FITS files.

    This class provides utilities to:

    - Read ACS FITS files
    - Extract detector panel count data
    - Build light curves using GDT TimeBins
    - Detect transient signals with Bayesian Blocks
    - Estimate burst durations (e.g. T90)
    - Compute signal significance using Li & Ma statistics
    - Fit polynomial background models
    - Plot raw and processed light curves

    The class is designed for transient analysis workflows
    such as GRB or high-energy event detection using ACS panel data.

    Main features
    -------------
    - FITS data extraction
    - Automatic best-panel selection
    - Bayesian Blocks segmentation
    - Background estimation
    - Signal/background count extraction
    - Visualization utilities

    Notes
    -----
    Expected FITS structure:
        - Primary HDU contains metadata
        - Extension HDU contains:
            T_MIN, T_MAX and ACS panel counts

    Panel mapping convention:
        SCBA -> z
        SCBB -> y
        SCBC -> x
    """

    def __init__(self):
        pass
        
    def open_fits_file(self, fits_path):

        # =========================
        # OPEN FITS
        # =========================
        hdul = fits.open(fits_path)

        header = hdul[0].header
        time_start = header["DATE-OBS"]

        data = hdul[1].data

        # =========================
        # TIME
        # =========================
        t_min = data["TIME"]

        dt = data["TIMEDEL"][:, 0]

        t_max = t_min + dt

        # =========================
        # COUNTS
        # =========================
        counts = data["COUNT"]

        panels = {
            "SCBA_A0": counts[:, 0],
            "SCBA_A1": counts[:, 1],
            "SCBB_A0": counts[:, 2],
            "SCBB_A1": counts[:, 3],
            "SCBC_A0": counts[:, 4],
            "SCBC_A1": counts[:, 5],
        }

        hdul.close()

        return t_min, t_max, panels

    def open_fits_file_old(self,fits_path):
        
        # =========================
        # OPEN FITS
        # =========================
        hdul = fits.open(fits_path)
        header = hdul[0].header
        
        time_start = header['DATE-OBS']
        
        data = hdul[1].data

        t_min = data["T_MIN"]
        t_max = data["T_MAX"]
        t_center = 0.5 * (t_min + t_max)
        
        panels = {
            "SCBA_A0": data["COUNTS_SCBA_A0_G"],
            "SCBA_A1": data["COUNTS_SCBA_A1_G"],
            "SCBB_A0": data["COUNTS_SCBB_A0_G"],
            "SCBB_A1": data["COUNTS_SCBB_A1_G"],
            "SCBC_A0": data["COUNTS_SCBC_A0_G"],
            "SCBC_A1": data["COUNTS_SCBC_A1_G"],
        }
        
        hdul.close()
        
        return t_min,t_max,panels
        
    def plot_acs_from_fits(self,fits_path,plot_counts=False):

        t_min,t_max,panels = self.open_fits_file(fits_path)
        
        dt = t_max - t_min

        # =========================
        # PLOT COUNTS
        # =========================
        if plot_counts:
            fig1, axes1 = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
            axes1 = axes1.flatten()

            for i, (name, counts) in enumerate(panels.items()):
                ax = axes1[i]
                ax.step(t_max, counts, where="pre",color="b")
                ax.set_title(f"{name} (Counts)")
                ax.set_ylabel("Counts")

            axes1[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            plt.show()

        # =========================
        # PLOT RATE (counts/sec)
        # =========================
        fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        axes2 = axes2.flatten()

        for i, (name, counts) in enumerate(panels.items()):
            rate = counts / dt

            ax = axes2[i]
            ax.step(t_max, rate, where="pre",color="b")
            ax.set_title(f"{name} (Rate)")
            ax.set_ylabel("Counts/s")

        axes2[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plt.show()
        
        return t_min,t_max,panels
    
    def analyze_lc_with_bblocks(self,lightcurve, p0=0.05, isRate=False,panels=['z1', 'z0', 'x1', 'x0', 'y1', 'y0'], bkg_buffer=1.0, bkg_order=2, sanity_check=False):
        """
        Analyze the light curve data.

        Input:
            - lightcurve: dictionary containing:
                {
                    "t_min": array,
                    "t_max": array,
                    "panels": {
                        "SCBA_A0": array,
                        "SCBA_A1": array,
                        "SCBB_A0": array,
                        "SCBB_A1": array,
                        "SCBC_A0": array,
                        "SCBC_A1": array
                    }
                }

            - p0: false alarm probability for Bayesian Blocks
            - isRate: if True, input contains rates → converted to counts
            - panels: internal panel names to analyze (x,y,z convention)
            - bkg_buffer: extra time excluded around the T90 window in the background fit
            - bkg_order: polynomial order of the background fit
            - sanity_check: if True, attach a "sanity" diagnostics dict to the output

        Output:
            dictionary containing:
                {
                    "lc_sel": selected lightcurve,
                    "bb_lc": BayesianBlocksLightcurve object,
                    "lc": dictionary of all lightcurves,
                    "best_panel": panel with the highest peak SNR,
                    "best_snr": peak SNR of the selected panel,
                    "seed_panel": panel used to get the first T90 window,
                    "bkg_fits": polynomial background fits reused later,
                    "panel_snr": per-panel peak SNR diagnostics,
                    "snr_bin_selection": greedy high-SNR bin set from the best panel,
                    "sanity": diagnostics dict if sanity_check=True, else None,
                    "signal_tstart": signal start time,
                    "signal_tstop": signal stop time,
                    "t90": T90 duration,
                    "t90_err_low": lower T90 uncertainty,
                    "t90_err_high": upper T90 uncertainty,
                    "significance": integrated Li&Ma significance,
                    "significance_peak": peak-bin significance,
                    "t_min": input t_min array,
                    "t_max": input t_max array
                }
        """

        # =========================
        # TIME BINS
        # =========================
        t_min = np.asarray(lightcurve["t_min"])
        t_max = np.asarray(lightcurve["t_max"])

        # durata dei bin (non uniforme!)
        exposure = t_max - t_min

        # =========================
        # READ PANELS + MAPPING
        # =========================
        pan = lightcurve["panels"]

        # mapping SCB → xyz (definito da te)
        signal = {}

        signal["z1"] = np.asarray(pan["SCBA_A0"])
        signal["z0"] = np.asarray(pan["SCBA_A1"])

        signal["y1"] = np.asarray(pan["SCBB_A0"])
        signal["y0"] = np.asarray(pan["SCBB_A1"])

        signal["x1"] = np.asarray(pan["SCBC_A0"])
        signal["x0"] = np.asarray(pan["SCBC_A1"])

        # if rate -> convert to counts
        if isRate:
            for k in signal:
                signal[k] = signal[k] * exposure

        # =========================
        # LIGHT CURVES CONTSRUCTION
        # =========================
        lc = {}

        for panel in panels:
            lc[panel] = TimeBins(
                signal[panel],
                t_min,
                t_max,
                exposure
            )

        # =========================
        # BAYESIAN BLOCKS (SEED PANEL)
        # =========================
        # A T90 window is needed to fit background without including the burst.
        # Use the highest-rate panel only as a seed; the final panel is chosen by SNR.
        def _compute_bb(panel_lc):
            bb = BayesianBlocksLightcurve(panel_lc)
            bb.compute_bayesian_blocks(p0=p0)
            t90_val = bb.duration(quantile=.9)
            t90_err = bb.duration_error(.9, nsamples=100)
            tstart, tstop = bb.quantile_range(quantile=.9)
            return bb, t90_val, t90_err, tstart, tstop

        seed_panel = max(panels, key=lambda p: np.max(lc[p].rates))
        lc_sel = lc[seed_panel]

        try:
            bb_lc, t90, t90_error, t90_tstart, t90_tstop = _compute_bb(lc_sel)
        except Exception as e:
            print(e)
            print("WARNING")
            return {
                "lc_sel": lc_sel,
                "bb_lc": None,
                "lc": None,
                "best_panel": seed_panel,
                "best_snr": -9999,
                "seed_panel": seed_panel,
                "bkg_fits": {},
                "panel_snr": {},
                "snr_bin_selection": None,
                "sanity": None,
                "signal_tstart": -9999,
                "signal_tstop": -9999,
                "t90": -9999,
                "t90_err_low": -9999,
                "t90_err_high": -9999,
                "significance": -9999,
                "significance_peak": -9999,
                "t_min": None,
                "t_max": None
            }

        signal_range = (t90_tstart, t90_tstop)

        # =========================
        # BACKGROUND FITS + BEST PANEL
        # =========================
        # One polynomial fit per panel, excluding T90 ± buffer.
        # SNR = sqrt((d - b) / d) on the peak bin, with b from the fitted background.
        bkg_fits = {}
        panel_snr = {}
        best_panel = None
        best_snr = float("-inf")

        for panel in panels:
            panel_lc = lc[panel]
            ipeak = int(np.argmax(panel_lc.rates))
            d = float(panel_lc.counts[ipeak])
            b = np.nan
            snr = 0.0

            try:
                res = self.fit_background(
                    panel_lc,
                    signal_range,
                    buffer=bkg_buffer,
                    order=bkg_order
                )
            except RuntimeError as e:
                print(e)
                bkg_fits[panel] = None
            else:
                bkg_fits[panel] = res
                b = float(res["bkg_counts"][ipeak])
                if d > 0 and d > b:
                    snr = np.sqrt((d - b) / d)

            panel_snr[panel] = {
                "snr": snr,
                "ipeak": ipeak,
                "d": d,
                "b": b,
                "t_lo": float(panel_lc.lo_edges[ipeak]),
                "t_hi": float(panel_lc.hi_edges[ipeak]),
                "t_center": float(panel_lc.centroids[ipeak]),
                "rate_peak": float(panel_lc.rates[ipeak]),
            }

            if snr > best_snr:
                best_snr = snr
                best_panel = panel

        if best_panel != seed_panel:
            try:
                bb_new, t90_new, t90_err_new, tstart_new, tstop_new = _compute_bb(lc[best_panel])
            except Exception as e:
                print(e)
                print("WARNING: Bayesian blocks failed on highest-SNR panel, keeping seed panel")
                best_panel = seed_panel
            else:
                bb_lc = bb_new
                t90 = t90_new
                t90_error = t90_err_new
                t90_tstart = tstart_new
                t90_tstop = tstop_new

        lc_sel = lc[best_panel]

        # =========================
        # GREEDY HIGH-SNR BIN SET
        # =========================
        # Rank bins on the best panel by per-bin SNR, then add them in that
        # order while the cumulative SNR increases. The resulting bin indices
        # are reused for every panel.
        snr_bin_selection = None
        best_fit = bkg_fits.get(best_panel)
        if best_fit is not None:
            snr_bin_selection = self.select_optimal_snr_bins(
                lc_sel.counts,
                best_fit["bkg_counts"],
            )

        # =========================
        # SIGNAL + BACKGROUND
        # =========================
        signal_lc = lc_sel.slice(t90_tstart, t90_tstop)

        # background prima e dopo (FIX rispetto al tuo codice originale)
        bkg_lc1 = lc_sel.slice(lc_sel.centroids[0], t90_tstart)
        bkg_lc2 = lc_sel.slice(t90_tstop, lc_sel.centroids[-1])

        t_on = np.sum(signal_lc.exposure)
        t_off = np.sum(bkg_lc1.exposure) + np.sum(bkg_lc2.exposure)

        N_on = np.sum(signal_lc.rates * signal_lc.exposure)
        N_off = (
            np.sum(bkg_lc1.rates * bkg_lc1.exposure) +
            np.sum(bkg_lc2.rates * bkg_lc2.exposure)
        )

        alpha = t_on / t_off if t_off > 0 else 0

        if alpha > 0 and N_on > 0 and N_off > 0:
            S = np.sqrt(2) * (
                N_on * np.log(((1 + alpha) / alpha) * (N_on / (N_on + N_off))) +
                N_off * np.log((1 + alpha) * (N_off / (N_on + N_off)))
            )**0.5
        else:
            S = 0

        # # =========================
        # # PEAK SIGNIFICANCE
        # # =========================
        # significance = []

        # for rate, exp in zip(signal_lc.rates, signal_lc.exposure):
        #     N_on_bin = rate * exp
        #     alpha_bin = exp / t_off if t_off > 0 else 0

        #     if alpha_bin > 0 and N_on_bin > 0 and N_off > 0:
        #         S_bin = np.sqrt(2) * (
        #             N_on_bin * np.log(((1 + alpha_bin) / alpha_bin) *
        #                             (N_on_bin / (N_on_bin + N_off))) +
        #             N_off * np.log((1 + alpha_bin) *
        #                         (N_off / (N_on_bin + N_off)))
        #         )**0.5
        #     else:
        #         S_bin = 0

        #     significance.append(S_bin)

        # significance = np.array(significance)
        # S_peak = np.max(significance)

        sanity = None
        if sanity_check:
            snr_sel = snr_bin_selection or {}
            selected_bins = np.asarray(snr_sel.get("indices_time", []), dtype=int)
            selected_time_range = None
            if selected_bins.size > 0:
                selected_time_range = (
                    float(lc_sel.lo_edges[selected_bins].min()),
                    float(lc_sel.hi_edges[selected_bins].max()),
                )

            sanity = {
                "best_panel": best_panel,
                "best_snr": best_snr,
                "seed_panel": seed_panel,
                "bb_recomputed": best_panel != seed_panel,
                "selection_criterion": "SNR = sqrt((d - b) / d) on the peak-rate bin",
                "t90_window": (float(t90_tstart), float(t90_tstop)),
                "peak_bin": panel_snr.get(best_panel, {}),
                "panel_snr": panel_snr,
                "snr_bin_selection": snr_bin_selection,
                "selected_bins": selected_bins,
                "selected_time_range": selected_time_range,
                "panels": list(panels),
            }

        # =========================
        # OUTPUT
        # =========================
        return {
            "lc_sel": lc_sel,
            "bb_lc": bb_lc,
            "lc": lc,
            "best_panel": best_panel,
            "best_snr": best_snr,
            "seed_panel": seed_panel,
            "bkg_fits": bkg_fits,
            "panel_snr": panel_snr,
            "snr_bin_selection": snr_bin_selection,
            "sanity": sanity,
            "signal_tstart":t90_tstart,
            "signal_tstop": t90_tstop,
            "t90": t90,
            "t90_err_low": t90_error[0],
            "t90_err_high": t90_error[1],
            "significance": S,
            "significance_peak": -1,
            "t_min": t_min,
            "t_max": t_max
        }
    
    def select_optimal_snr_bins(self, counts, bkg_counts):
        """
        Select the bin set that maximises cumulative SNR on one light curve.

        1. Compute per-bin SNR = sqrt((d - b) / d)
        2. Sort bins by decreasing per-bin SNR
        3. Add bins in that order, accumulating observed and fitted background counts
        4. Stop when the cumulative SNR decreases

        Parameters
        ----------
        counts : array
            Observed counts per bin.
        bkg_counts : array
            Fitted background counts per bin.

        Returns
        -------
        result : dict
            - "indices": selected bin indices in greedy order
            - "indices_time": selected bin indices sorted in time
            - "order": all positive-SNR bins sorted by decreasing per-bin SNR
            - "bin_snr": per-bin SNR for the full light curve
            - "snr_cumulative": cumulative SNR after each added bin
            - "snr_optimal": cumulative SNR of the selected set
            - "d_sum", "b_sum": accumulated observed and background counts
        """

        counts = np.asarray(counts, dtype=float)
        bkg_counts = np.asarray(bkg_counts, dtype=float)

        n = counts.size
        bin_snr = np.zeros(n, dtype=float)
        valid = (counts > 0) & (counts > bkg_counts)
        bin_snr[valid] = np.sqrt((counts[valid] - bkg_counts[valid]) / counts[valid])

        order = np.argsort(-bin_snr)
        order = order[bin_snr[order] > 0]

        selected = []
        d_sum = 0.0
        b_sum = 0.0
        snr_cum = -np.inf
        snr_history = []

        for i in order:
            d_new = d_sum + counts[i]
            b_new = b_sum + bkg_counts[i]
            if d_new > 0 and d_new > b_new:
                snr_new = np.sqrt((d_new - b_new) / d_new)
            else:
                snr_new = 0.0

            if snr_new < snr_cum:
                break

            selected.append(int(i))
            d_sum = d_new
            b_sum = b_new
            snr_cum = snr_new
            snr_history.append(snr_new)

        if not selected and n > 0:
            ipeak = int(np.argmax(counts))
            selected = [ipeak]
            d_sum = float(counts[ipeak])
            b_sum = float(bkg_counts[ipeak])
            if d_sum > 0 and d_sum > b_sum:
                snr_cum = np.sqrt((d_sum - b_sum) / d_sum)
            else:
                snr_cum = 0.0
            snr_history = [snr_cum]

        selected = np.asarray(selected, dtype=int)

        return {
            "indices": selected,
            "indices_time": np.sort(selected),
            "order": order,
            "bin_snr": bin_snr,
            "snr_cumulative": np.asarray(snr_history, dtype=float),
            "snr_optimal": float(snr_cum if selected.size else 0.0),
            "d_sum": float(d_sum),
            "b_sum": float(b_sum),
        }

    def fit_background(self,lc, signal_range, buffer=0.0, order=2):
        
        """
        Fit del background polinomiale su una light curve GDT (TimeBins),
        escludendo la finestra del segnale.

        Parameters
        ----------
        lc : TimeBins
            Light curve del detector.
            Deve avere almeno: counts, lo_edges, hi_edges, exposure
        signal_range : tuple
            (tstart, tstop) del segnale/burst da escludere dal fit
        buffer : float, optional
            Margine extra da escludere attorno al segnale
        order : int, optional
            Ordine del polinomio

        Returns
        -------
        result : dict
            Dizionario con:
            - "model"           : oggetto Polynomial fittato
            - "mask_bkg"        : maschera booleana dei bin usati nel fit
            - "bkg_rate"        : background stimato in rate
            - "bkg_rate_err"    : errore sul background rate
            - "bkg_counts"      : background stimato in counts/bin
            - "bkg_counts_err"  : errore in counts/bin
            - "net_counts"      : counts osservati - background counts
            - "net_rate"        : rate osservato - background rate
        """

        tstart_sig = signal_range[0]
        tstop_sig = signal_range[1]
        excl_start = tstart_sig - buffer
        excl_stop = tstop_sig + buffer

        # bin completamente fuori dalla regione esclusa
        mask_bkg = (lc.hi_edges <= excl_start) | (lc.lo_edges >= excl_stop)

        n_bkg_bins = np.sum(mask_bkg)
        if n_bkg_bins < (order + 2):
            raise RuntimeError(
                f"Troppi pochi bin di background ({n_bkg_bins}) "
                f"per un polinomio di ordine {order}"
            )

        # costruiamo il modello come in bctools
        bkg_model = Polynomial(
            counts=lc.counts[mask_bkg][:, np.newaxis],
            tstart=lc.lo_edges[mask_bkg],
            tstop=lc.hi_edges[mask_bkg],
            exposure=lc.exposure[mask_bkg]
        )

        bkg_model.fit(order=order)

        #WARNING interpolate() return RATE, not counts
        bkg_rate, bkg_rate_err = bkg_model.interpolate(
            tstart=lc.lo_edges,
            tstop=lc.hi_edges
        )

        # da shape (N, 1) a (N,)
        bkg_rate = np.squeeze(bkg_rate)
        bkg_rate_err = np.squeeze(bkg_rate_err)

        # conversione a counts/bin
        bkg_counts = bkg_rate * lc.exposure
        bkg_counts_err = bkg_rate_err * lc.exposure

        # osservati
        obs_rate = lc.counts / lc.exposure

        # netti
        net_counts = lc.counts - bkg_counts
        net_rate = obs_rate - bkg_rate

        return {
            "model": bkg_model,
            "mask_bkg": mask_bkg,
            "bkg_rate": bkg_rate,
            "bkg_rate_err": bkg_rate_err,
            "bkg_counts": bkg_counts,
            "bkg_counts_err": bkg_counts_err,
            "net_counts": net_counts,
            "net_rate": net_rate,
        }

    def plot_lc(self, lc_sel, bb_lc,signal_range, save=False,prefix=""):

        
        # =========================
        # RAW LIGHT CURVE
        # =========================
        fig = plt.figure(figsize=(10,4))

        plt.step(lc_sel.centroids, lc_sel.rates, where="mid")

        plt.xlabel("Time [s]")
        plt.ylabel("Counts / s")
        plt.title("Light curve")
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(f"{self.output_dir}/acs_raw_lc.png", bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        # =========================
        # BAYESIAN BLOCKS
        # =========================
        fig = plt.figure(figsize=(10,4))

        plt.plot(
            lc_sel.centroids,
            bb_lc.bkg_counts / lc_sel.exposure,
            color='red',
            ls=':',
            label="Fitted background"
        )

        plt.errorbar(
            lc_sel.centroids,
            lc_sel.rates,
            xerr=[
                lc_sel.centroids - lc_sel.lo_edges,
                lc_sel.hi_edges - lc_sel.centroids
            ],
            yerr=lc_sel.rate_uncertainty,
            ls='none',
            color='.7',
            label='Raw data'
        )

        lc_bayes = bb_lc.bb_lightcurve
        print(lc_bayes)

        plt.plot(
            np.append(lc_bayes.lo_edges, lc_bayes.hi_edges[-1]),
            np.append(lc_bayes.rates, lc_bayes.rates[-1]),
            drawstyle='steps-post',
            label='Bayesian blocks',
            color="#1f77b4"
        )
        
        plt.xlabel("Time (s)")
        plt.ylabel("Counts")

        if signal_range is not None:
            # Vertical lines showing the start and stop of the identified signal
            plt.axvline(
                signal_range[0],
                ls="--",
                color='olive',
                label="Signal start/stop"
            )

            plt.axvline(
                signal_range[1],
                ls="--",
                color='olive'
            )

        plt.legend()

        if save:
            plt.savefig(f"{self.output_dir}/{prefix}_acs_bayesian_blocks.png", bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_background(
        self,
        lc,
        res,
        signal_range,
        signal_counts,
        background_counts,
        panel_name,
        save=False,
        prefix="",
        selected_indices=None
    ):
        """
        Plot observed and background light curves.
        """

        t = 0.5 * (lc.lo_edges + lc.hi_edges)

        plt.figure(figsize=(10, 5))

        plt.step(
            t, lc.rates,
            where="mid",
            label="Observed counts",
            color="#1f77b4"
        )

        plt.plot(
            t, res["bkg_rate"],
            label="Background",
            color="red"
        )

        if selected_indices is not None and len(selected_indices) > 0:
            for k, i in enumerate(selected_indices):
                plt.axvspan(
                    lc.lo_edges[i],
                    lc.hi_edges[i],
                    alpha=0.25,
                    color="#1f77b4",
                    label="SNR-selected bins" if k == 0 else None
                )
        elif signal_range is not None:
            plt.axvspan(
                signal_range[0],
                signal_range[1],
                alpha=0.2,
                label="Signal window",
                color="#1f77b4"
            )

        plt.xlabel("Time")
        plt.ylabel("Counts / bin")

        plt.title(
            f"{panel_name} | "
            f"signal={signal_counts:.1f}, "
            f"bkg={background_counts:.1f}"
        )

        plt.legend()
        plt.tight_layout()

        if save:
            filename = f"{prefix}_background_{panel_name}.png"
            plt.savefig(self.output_dir+"/"+filename, dpi=300)
            plt.close()
            print(f"Saved: {filename}")
        else:
            plt.show()

    def print_panel_snr_sanity_check(self, results):
        """
        Print panel and bin-selection diagnostics from an analysis result dict.
        Requires extract_source_data_from_fits(..., sanity_check=True).
        """

        sanity = results.get("sanity")
        if not sanity:
            print("No sanity-check results. Re-run with sanity_check=True.")
            return

        panels = sanity.get("panels") or list((results.get("panel_snr") or {}).keys())
        best_panel = sanity.get("best_panel", results.get("best_panel"))
        seed_panel = sanity.get("seed_panel", results.get("seed_panel"))
        best_snr = sanity.get("best_snr", results.get("best_snr"))
        panel_snr = sanity.get("panel_snr") or results.get("panel_snr") or {}
        snr_sel = sanity.get("snr_bin_selection") or results.get("snr_bin_selection") or {}
        peak = sanity.get("peak_bin") or {}
        t90_window = sanity.get("t90_window")
        selected_bins = np.asarray(sanity.get("selected_bins", []), dtype=int)

        print("\n" + "=" * 80)
        print("SANITY CHECK: panel selection")
        print("=" * 80)
        print(f"Seed panel (max rate, used for first T90): {seed_panel}")
        print(f"Selected panel (max peak SNR):             {best_panel}")
        print(f"Selection criterion: {sanity.get('selection_criterion', '')}")
        if t90_window is not None:
            print(
                f"T90 window: [{t90_window[0]:.6f}, {t90_window[1]:.6f}] s "
                f"(duration={t90_window[1] - t90_window[0]:.6f} s)"
            )
        print(
            f"Selected peak bin: index={peak.get('ipeak', 'n/a')} | "
            f"[{peak.get('t_lo', np.nan):.6f}, {peak.get('t_hi', np.nan):.6f}] s | "
            f"center={peak.get('t_center', np.nan):.6f} s"
        )
        print(
            f"Selected peak counts: d={peak.get('d', np.nan):.4f} | "
            f"b={peak.get('b', np.nan):.4f} | "
            f"SNR={best_snr:.4f}"
        )
        if sanity.get("bb_recomputed"):
            print(
                f"Bayesian blocks were recomputed on {best_panel} "
                f"(different from seed {seed_panel})"
            )
        else:
            print(f"Bayesian blocks kept on seed panel {seed_panel}")

        print("-" * 80)
        print("SNR bin selection on the best panel (guide for all panels)")
        print(
            f"Selected bins: {selected_bins.size} | "
            f"greedy order={list(snr_sel.get('indices', []))} | "
            f"time order={list(selected_bins)}"
        )
        print(
            f"Cumulative optimal SNR={snr_sel.get('snr_optimal', np.nan):.4f} | "
            f"d_sum={snr_sel.get('d_sum', np.nan):.3f} | "
            f"b_sum={snr_sel.get('b_sum', np.nan):.3f}"
        )
        selected_time_range = sanity.get("selected_time_range")
        if selected_time_range is not None:
            print(
                f"Time coverage of selected bins: "
                f"[{selected_time_range[0]:.6f}, {selected_time_range[1]:.6f}] s"
            )

        print("-" * 80)
        print(
            f"{'panel':<8} {'sel':<5} {'ipeak':>7} {'t_center':>12} "
            f"{'d':>10} {'b':>10} {'SNR':>8}"
        )
        for p in panels:
            info = panel_snr.get(p, {})
            mark = "<--" if p == best_panel else ""
            print(
                f"{p:<8} {mark:<5} {info.get('ipeak', -1):>7d} "
                f"{info.get('t_center', np.nan):>12.4f} "
                f"{info.get('d', np.nan):>10.3f} "
                f"{info.get('b', np.nan):>10.3f} "
                f"{info.get('snr', np.nan):>8.4f}"
            )

        if "s_counts" in sanity:
            print("-" * 80)
            print("Signal counts per panel:", sanity["s_counts"])
            print("Background counts per panel:", sanity["b_counts"])
        print("=" * 80 + "\n")

    def plot_panel_snr_sanity_check(
        self,
        results,
        save=False,
        prefix=""
    ):
        """
        Plot panel and bin-selection diagnostics from an analysis result dict.
        Requires extract_source_data_from_fits(..., sanity_check=True).
        """

        sanity = results.get("sanity")
        if not sanity:
            print("No sanity-check results. Re-run with sanity_check=True.")
            return

        acs_lc = results["lc"]
        bkg_fits = results.get("bkg_fits") or {}
        panel_snr = sanity.get("panel_snr") or results.get("panel_snr") or {}
        panels = sanity.get("panels") or list(acs_lc.keys())
        best_panel = sanity.get("best_panel", results.get("best_panel"))
        seed_panel = sanity.get("seed_panel", results.get("seed_panel"))
        signal_range = sanity.get("t90_window")
        if signal_range is None and results.get("signal_tstart") is not None:
            signal_range = (results["signal_tstart"], results["signal_tstop"])
        snr_bin_selection = (
            sanity.get("snr_bin_selection")
            or results.get("snr_bin_selection")
        )

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()

        for ax, panel in zip(axes, panels):
            lc = acs_lc[panel]
            info = panel_snr.get(panel, {})
            res = bkg_fits.get(panel)
            selected = panel == best_panel

            ax.step(
                lc.centroids,
                lc.rates,
                where="mid",
                color="#1f77b4",
                label="Observed rate"
            )

            if res is not None:
                ax.plot(
                    lc.centroids,
                    res["bkg_rate"],
                    color="red",
                    label="Fitted background"
                )

            if signal_range is not None:
                ax.axvspan(
                    signal_range[0],
                    signal_range[1],
                    color="olive",
                    alpha=0.12,
                    label="T90 window"
                )

            if snr_bin_selection is not None:
                selected_bins = snr_bin_selection.get("indices_time", [])
                for k, i in enumerate(selected_bins):
                    ax.axvspan(
                        lc.lo_edges[i],
                        lc.hi_edges[i],
                        color="green",
                        alpha=0.28,
                        label="SNR-selected bins" if k == 0 else None
                    )

            if "t_lo" in info:
                ax.axvspan(
                    info["t_lo"],
                    info["t_hi"],
                    color="orange",
                    alpha=0.45,
                    label="Peak bin"
                )
                ax.plot(
                    info["t_center"],
                    info["rate_peak"],
                    "o",
                    color="orange",
                    markersize=7,
                    zorder=5
                )

            snr = info.get("snr", np.nan)
            ipeak = info.get("ipeak", -1)
            title = f"{panel} | SNR={snr:.3f} | peak bin={ipeak}"
            if selected:
                title = f"SELECTED  {title}"
                for spine in ax.spines.values():
                    spine.set_color("darkorange")
                    spine.set_linewidth(2.5)

            ax.set_title(title)
            ax.set_ylabel("Counts / s")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time [s]")
        axes[-2].set_xlabel("Time [s]")
        fig.suptitle(
            f"Panel SNR sanity check | selected={best_panel} | seed={seed_panel}",
            fontsize=13
        )
        plt.tight_layout()

        if save:
            filename = f"{prefix}_panel_snr_sanity_check.png"
            plt.savefig(self.output_dir + "/" + filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {filename}")
        else:
            plt.show()

        if snr_bin_selection is not None:
            hist = snr_bin_selection.get("snr_cumulative", np.array([]))
            if hist.size > 0:
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.plot(
                    np.arange(1, hist.size + 1),
                    hist,
                    marker="o",
                    color="#1f77b4"
                )
                ax2.axhline(
                    snr_bin_selection["snr_optimal"],
                    color="green",
                    ls="--",
                    label=f"Optimal SNR={snr_bin_selection['snr_optimal']:.4f}"
                )
                ax2.set_xlabel("Bins added (decreasing per-bin SNR)")
                ax2.set_ylabel("Cumulative SNR")
                ax2.set_title(
                    f"Greedy SNR selection on {best_panel} "
                    f"({hist.size} bins)"
                )
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                fig2.tight_layout()

                if save:
                    filename = f"{prefix}_snr_cumulative_sanity_check.png"
                    plt.savefig(
                        self.output_dir + "/" + filename,
                        dpi=300,
                        bbox_inches="tight"
                    )
                    plt.close(fig2)
                    print(f"Saved: {filename}")
                else:
                    plt.show()

    def extract_source_data_from_fits(self,fits_path,plot=False,save_plot=False,prefix="",panels = ['z1','z0','x1','x0','y1','y0'],p0=0.05,sanity_check=False):
        
        t_min,t_max,event_panels = self.open_fits_file(fits_path)
        
        light_curve = {"t_min":t_min,"t_max":t_max,"panels":event_panels}
        
        # compute the TimeBins LC, time_start, time_stop, t90 and Li&Ma signfiicance
        bblocks_analysis_results = self.analyze_lc_with_bblocks(light_curve, p0=p0, isRate=False, panels=panels, sanity_check=sanity_check)
        
        acs_lc = bblocks_analysis_results['lc']
        event_time_start = bblocks_analysis_results['signal_tstart']
        lc_sel = bblocks_analysis_results['lc_sel']
        bb_lc = bblocks_analysis_results['bb_lc']
        print(bblocks_analysis_results['signal_tstart'])
        t90_tstart = bblocks_analysis_results['signal_tstart']
        t90_tstop = bblocks_analysis_results['signal_tstop']
        
        if bb_lc is None:
            return -1,-1,-1,-1
        signal_range = (t90_tstart,t90_tstop)
        
        #convert event time start from TT to Unix time stamp
        #mjd_ref_timestamp = 1735689669.184
        mjd_ref_timestamp = 1735689600.184
        event_time_start_unix = mjd_ref_timestamp + event_time_start
        
        if plot:
            self.plot_lc(lc_sel,bb_lc,signal_range,save=save_plot,prefix=prefix)

        results = []
        s_counts = []
        b_counts = []


        tstart = t90_tstart
        tstop = t90_tstop
        duration = tstop-tstart
        bkg_fits = bblocks_analysis_results["bkg_fits"]
        snr_sel = bblocks_analysis_results.get("snr_bin_selection") or {}
        selected_indices = np.asarray(snr_sel.get("indices_time", []), dtype=int)

        for p in panels:
        
            lc = acs_lc[p]
            res = bkg_fits.get(p)
            if res is None:
                res = self.fit_background(lc, signal_range, buffer=1.0, order=2)

            results.append(res)
            
            # --------------------------------------------------
            # SAME BINS FOR SOURCE AND BACKGROUND
            # High-SNR bins chosen on the best panel, applied to every panel.
            # --------------------------------------------------
            if selected_indices.size > 0:
                mask_sig = np.zeros(lc.counts.size, dtype=bool)
                mask_sig[selected_indices] = True
            else:
                mask_sig = (
                    (lc.hi_edges > tstart) &
                    (lc.lo_edges < tstop)
                )

            bin_indices = np.where(mask_sig)[0]

            # print("\n" + "=" * 80)
            # print(f"PANEL: {p}")
            # print(f"Signal window requested: [{tstart:.12f}, {tstop:.12f}]")
            # print(f"Signal window duration : {tstop - tstart:.12f} s")
            # print(f"Number of selected bins: {len(selected_indices)}")
            # print("-" * 80)

            for i in bin_indices:

                lo = lc.lo_edges[i]
                hi = lc.hi_edges[i]
                exp = lc.exposure[i]

                src_counts_bin = lc.counts[i]
                bkg_counts_bin = res["bkg_counts"][i]

            #     print(
            #         f"bin {i:5d} | "
            #         f"[{lo:.12f}, {hi:.12f}] | "
            #         f"width={hi-lo:.12f} | "
            #         f"exp={exp:.12f} | "
            #         f"source={src_counts_bin:.6f} | "
            #         f"background={bkg_counts_bin:.6f}"
            #     )

            # print("-" * 80)

            signal_counts = np.sum(lc.counts[mask_sig])
            background_counts = np.sum(res["bkg_counts"][mask_sig])

            # print(f"TOTAL source counts     = {signal_counts:.6f}")
            # print(f"TOTAL background counts = {background_counts:.6f}")

            # if len(selected_indices) > 0:
            #     first_bin = selected_indices[0]
            #     last_bin = selected_indices[-1]

            #     print(
            #         f"Actual selected window  = "
            #         f"[{lc.lo_edges[first_bin]:.12f}, "
            #         f"{lc.hi_edges[last_bin]:.12f}]"
            #     )

            #     print(
            #         f"Actual selected duration = "
            #         f"{lc.hi_edges[last_bin] - lc.lo_edges[first_bin]:.12f} s"
            #     )

            # print("=" * 80)
            s_counts.append(signal_counts)
            b_counts.append(background_counts)
            
            if plot:
                self.plot_background(
                    lc,
                    res,
                    signal_range,
                    signal_counts,
                    background_counts,
                    p,
                    save=save_plot,
                    prefix=prefix,
                    selected_indices=bin_indices
                )
               
                
        # obtain numpy arrays
        s_counts = np.array(s_counts)
        b_counts = np.array(b_counts)

        if sanity_check:
            sanity = bblocks_analysis_results.get("sanity") or {}
            sanity["s_counts"] = s_counts
            sanity["b_counts"] = b_counts
            bblocks_analysis_results["sanity"] = sanity
    
        return s_counts,b_counts,event_time_start_unix,bblocks_analysis_results

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from astropy.io import fits
from gdt.core.background.binned import Polynomial
from gdt.core.data_primitives import TimeBins
from bctools.analysis import BayesianBlocksLightcurve
import numpy as np

# Spacecraft-frame panel name -> FITS count column.
# SCBA is z, SCBB is y, SCBC is x. A0/A1 are the two faces.
_PANEL_COLUMNS = {
    "z1": "SCBA_A0",
    "z0": "SCBA_A1",
    "y1": "SCBB_A0",
    "y0": "SCBB_A1",
    "x1": "SCBC_A0",
    "x0": "SCBC_A1",
}


def _bins_overlapping(lc, tstart, tstop):
    """Light-curve bins whose interval overlaps [tstart, tstop]."""
    return (lc.hi_edges > tstart) & (lc.lo_edges < tstop)


def _empty_rank():
    """Ranking record used when a panel has no SNR selection."""
    indices = np.array([], dtype=int)
    return {
        "snr": 0.0,
        "d": np.nan,
        "b": np.nan,
        "n_bins": 0,
        "indices": indices,
        "indices_time": indices,
        "t90_tstart": np.nan,
        "t90_tstop": np.nan,
        "t90": np.nan,
        "bkg_rate": None,
    }


def _panel_rank(selection, tstart, tstop, t90, snr, bkg_rate=None):
    """Dictionary with one panel's greedy SNR, counts, and its own T90."""
    if selection is None:
        info = _empty_rank()
        info["t90_tstart"] = float(tstart)
        info["t90_tstop"] = float(tstop)
        info["t90"] = float(t90)
    else:
        info = {
            "snr": snr,
            "d": float(selection["d_sum"]),
            "b": float(selection["b_sum"]),
            "n_bins": int(selection["indices"].size),
            "indices": np.asarray(selection["indices"], dtype=int),
            "indices_time": np.asarray(selection["indices_time"], dtype=int),
            "t90_tstart": float(tstart),
            "t90_tstop": float(tstop),
            "t90": float(t90),
            "bkg_rate": None,
        }
    if bkg_rate is not None:
        info["bkg_rate"] = np.asarray(bkg_rate, dtype=float)
    return info


def _failed_analysis(lc_fallback):
    """Sentinel result when no panel produces a T90."""
    return {
        "lc_sel": lc_fallback,
        "bb_lc": None,
        "lc": None,
        "best_panel": None,
        "best_snr": -9999,
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
        "t_max": None,
    }


# Print size for a double-column figure. Fonts stay readable after the
# journal scales the file to the text width.
_PAPER_FIGSIZE = (15, 12)
_PAPER_BLUE = "#1f77b4"
_PAPER_RED = "#d62728"
_PAPER_T90 = "#b7e4c7"
_PAPER_BINS = "#1b7f3b"


def _draw_bin_bars(ax, lc, indices, t0, color, alpha=0.45):
    """Full-height bars whose width is exactly one light-curve bin."""
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return False
    lo = lc.lo_edges[indices] - t0
    width = lc.hi_edges[indices] - lc.lo_edges[indices]
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.bar(
        lo,
        np.ones(lo.size),
        width=width,
        bottom=0.0,
        align="edge",
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        transform=transform,
        zorder=1,
        clip_on=True,
    )
    return True


def _style_paper_ax(ax):
    """Tick and spine sizes for a figure that will be printed small."""
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=7,
        length=2.5,
        width=0.5,
        direction="in",
    )
    ax.grid(True, alpha=0.25, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_ylabel("Counts / s", fontsize=8)


def _save_paper_figure(fig, output_dir, stem):
    """Write a 300 dpi PNG and a vector PDF for the paper."""
    png = f"{stem}.png"
    pdf = f"{stem}.pdf"
    fig.savefig(output_dir + "/" + png, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir + "/" + pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def _finish_figure(fig, plot):
    """Show the figure, or close it when it stays off screen."""
    if plot:
        plt.show()
    else:
        plt.close(fig)


def _li_ma_significance(signal_lc, bkg_before, bkg_after):
    """Li & Ma significance of the on-burst window against the off-burst exposure."""
    t_on = np.sum(signal_lc.exposure)
    t_off = np.sum(bkg_before.exposure) + np.sum(bkg_after.exposure)
    n_on = np.sum(signal_lc.rates * signal_lc.exposure)
    n_off = (
        np.sum(bkg_before.rates * bkg_before.exposure)
        + np.sum(bkg_after.rates * bkg_after.exposure)
    )
    if t_off <= 0 or n_on <= 0 or n_off <= 0:
        return 0.0

    alpha = t_on / t_off
    return np.sqrt(2) * (
        n_on * np.log(((1 + alpha) / alpha) * (n_on / (n_on + n_off)))
        + n_off * np.log((1 + alpha) * (n_off / (n_on + n_off)))
    ) ** 0.5

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

    def __init__(self, output_dir=None):
        """Directory for saved plots. Unused when plots are only shown."""
        self.output_dir = output_dir
        
    def open_fits_file(self, fits_path):
        """Read bin edges and the six panel count columns from an ACS FITS file."""
        hdul = fits.open(fits_path)
        # Touch DATE-OBS so a file without the observation header fails here.
        hdul[0].header["DATE-OBS"]
        data = hdul[1].data

        t_min = data["TIME"]
        t_max = t_min + data["TIMEDEL"][:, 0]
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
                    "best_panel": panel with the highest greedy SNR inside its own T90,
                    "best_snr": optimal cumulative SNR inside that panel's own T90,
                    "bkg_fits": polynomial background fits on the selected T90,
                    "panel_snr": per-panel greedy SNR selection inside that panel's own T90,
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

        # Bin widths are not uniform: 1 s, 250 ms and 50 ms.
        t_min = np.asarray(lightcurve["t_min"])
        t_max = np.asarray(lightcurve["t_max"])
        exposure = t_max - t_min

        raw = lightcurve["panels"]
        counts = {
            name: np.asarray(raw[column])
            for name, column in _PANEL_COLUMNS.items()
        }
        if isRate:
            counts = {name: values * exposure for name, values in counts.items()}

        lc = {
            panel: TimeBins(counts[panel], t_min, t_max, exposure)
            for panel in panels
        }

        # Each panel is ranked inside its own T90. The winner's T90
        # becomes the common window used for the final counts.
        def t90_of(panel_lc):
            bb = BayesianBlocksLightcurve(panel_lc)
            bb.compute_bayesian_blocks(p0=p0)
            t90_val = bb.duration(quantile=.9)
            t90_err = bb.duration_error(.9, nsamples=100)
            tstart, tstop = bb.quantile_range(quantile=.9)
            return bb, t90_val, t90_err, tstart, tstop

        bb_by_panel = {}
        bb_panels = {}
        bkg_fits = {}
        panel_snr = {}
        best_panel = None
        best_snr = float("-inf")

        for panel in panels:
            panel_lc = lc[panel]
            try:
                bb_panel, t90_panel, t90_err_panel, tstart, tstop = t90_of(panel_lc)
            except Exception as exc:
                print(exc)
                print(f"WARNING: Bayesian blocks failed on {panel}")
                bb_by_panel[panel] = None
                bb_panels[panel] = None
                bkg_fits[panel] = None
                panel_snr[panel] = _empty_rank()
                continue

            bb_by_panel[panel] = (bb_panel, t90_panel, t90_err_panel, tstart, tstop)
            bb_panels[panel] = bb_panel
            in_t90 = _bins_overlapping(panel_lc, tstart, tstop)
            selection = None
            snr = 0.0
            ranking_rate = None

            try:
                fit = self.fit_background(
                    panel_lc,
                    (tstart, tstop),
                    buffer=bkg_buffer,
                    order=bkg_order,
                )
            except RuntimeError as exc:
                print(exc)
                bkg_fits[panel] = None
            else:
                bkg_fits[panel] = fit
                ranking_rate = fit["bkg_rate"]
                if np.any(in_t90):
                    selection = self.select_optimal_snr_bins(
                        panel_lc.counts,
                        fit["bkg_counts"],
                        mask=in_t90,
                    )
                    snr = selection["snr_optimal"]

            panel_snr[panel] = _panel_rank(
                selection, tstart, tstop, t90_panel, snr, ranking_rate
            )
            if selection is not None and snr > best_snr:
                best_snr = snr
                best_panel = panel

        if best_panel is None:
            print("WARNING: Bayesian blocks failed on every panel")
            fallback = lc[panels[0]] if panels else None
            return _failed_analysis(fallback)

        bb_lc, t90, t90_error, t90_tstart, t90_tstop = bb_by_panel[best_panel]
        lc_sel = lc[best_panel]
        signal_range = (t90_tstart, t90_tstop)

        # Ranking fits excluded each panel's own window. Refit every panel
        # on the single winning T90 before summing counts.
        for panel in panels:
            try:
                bkg_fits[panel] = self.fit_background(
                    lc[panel],
                    signal_range,
                    buffer=bkg_buffer,
                    order=bkg_order,
                )
            except RuntimeError as exc:
                print(exc)
                bkg_fits[panel] = None

        # Final bins are chosen only inside the common T90, on the winning
        # panel. The same indices are then applied to every panel.
        snr_bin_selection = None
        best_fit = bkg_fits.get(best_panel)
        if best_fit is not None:
            snr_bin_selection = self.select_optimal_snr_bins(
                lc_sel.counts,
                best_fit["bkg_counts"],
                mask=_bins_overlapping(lc_sel, t90_tstart, t90_tstop),
            )

        signal_lc = lc_sel.slice(t90_tstart, t90_tstop)
        bkg_before = lc_sel.slice(lc_sel.centroids[0], t90_tstart)
        bkg_after = lc_sel.slice(t90_tstop, lc_sel.centroids[-1])
        significance = _li_ma_significance(signal_lc, bkg_before, bkg_after)

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
                "selection_criterion": "greedy cumulative SNR inside each panel's own T90",
                "t90_window": (float(t90_tstart), float(t90_tstop)),
                "peak_bin": panel_snr.get(best_panel, {}),
                "panel_snr": panel_snr,
                "snr_bin_selection": snr_bin_selection,
                "selected_bins": selected_bins,
                "selected_time_range": selected_time_range,
                "panels": list(panels),
            }

        return {
            "lc_sel": lc_sel,
            "bb_lc": bb_lc,
            "bb_panels": bb_panels,
            "lc": lc,
            "best_panel": best_panel,
            "best_snr": best_snr,
            "bkg_fits": bkg_fits,
            "panel_snr": panel_snr,
            "snr_bin_selection": snr_bin_selection,
            "sanity": sanity,
            "signal_tstart":t90_tstart,
            "signal_tstop": t90_tstop,
            "t90": t90,
            "t90_err_low": t90_error[0],
            "t90_err_high": t90_error[1],
            "significance": significance,
            "significance_peak": -1,
            "t_min": t_min,
            "t_max": t_max
        }
    
    @staticmethod
    def counts_snr(d, b):
        """
        Poisson SNR for observed counts d and fitted background b.

        SNR = (d - b) / sqrt(b)
        """
        d_arr = np.atleast_1d(np.asarray(d, dtype=float))
        b_arr = np.atleast_1d(np.asarray(b, dtype=float))
        snr = np.zeros(d_arr.shape, dtype=float)
        ok = b_arr > 0
        snr[ok] = (d_arr[ok] - b_arr[ok]) / np.sqrt(b_arr[ok])
        if np.ndim(d) == 0:
            return float(snr[0])
        return snr

    def select_optimal_snr_bins(self, counts, bkg_counts, mask=None):
        """
        Select the bin set that maximises cumulative SNR on one light curve.

        1. Compute per-bin SNR = (d - b) / sqrt(b)
        2. Sort bins by decreasing per-bin SNR
        3. Add bins in that order, accumulating observed and fitted background counts
        4. Stop when the cumulative SNR decreases

        Parameters
        ----------
        counts : array
            Observed counts per bin.
        bkg_counts : array
            Fitted background counts per bin.
        mask : array of bool, optional
            Bins allowed to enter the selection. Indices stay in the frame of
            ``counts``. The default uses every bin.

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

        bin_snr = self.counts_snr(counts, bkg_counts)
        n = counts.size
        if mask is None:
            eligible = np.ones(n, dtype=bool)
        else:
            eligible = np.asarray(mask, dtype=bool)
            if eligible.shape != (n,):
                raise ValueError("mask length must match counts")

        order = np.flatnonzero(eligible)
        if order.size:
            order = order[np.argsort(-bin_snr[order])]
            order = order[bin_snr[order] > 0]

        selected = []
        d_sum = 0.0
        b_sum = 0.0
        snr_cum = -np.inf
        snr_history = []

        # Recompute SNR on the summed counts. A positive bin can still
        # lower it, because the background it adds grows the denominator.
        for i in order:
            d_new = d_sum + counts[i]
            b_new = b_sum + bkg_counts[i]
            snr_new = self.counts_snr(d_new, b_new)

            if snr_new < snr_cum:
                break

            selected.append(int(i))
            d_sum = d_new
            b_sum = b_new
            snr_cum = snr_new
            snr_history.append(snr_new)

        if not selected and np.any(eligible):
            eligible_idx = np.flatnonzero(eligible)
            ipeak = int(eligible_idx[np.argmax(counts[eligible_idx])])
            selected = [ipeak]
            d_sum = float(counts[ipeak])
            b_sum = float(bkg_counts[ipeak])
            snr_cum = self.counts_snr(d_sum, b_sum)
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

    def fit_background(self, lc, signal_range, buffer=0.0, order=2):
        """
        Polynomial background fit on a TimeBins light curve.

        Bins inside ``signal_range``, expanded by ``buffer`` on both sides,
        are left out of the fit. The model is then evaluated on every bin.

        Returns
        -------
        dict
            ``model`` is the fitted Polynomial.
            ``mask_bkg`` marks bins used in the fit.
            ``bkg_rate`` and ``bkg_counts`` are the model on every bin.
            ``interpolate`` returns a rate, so counts are rate times exposure.
            ``net_counts`` and ``net_rate`` are observed minus background.
        """

        tstart, tstop = signal_range
        excl_start = tstart - buffer
        excl_stop = tstop + buffer

        # A bin is background only when it lies fully outside the excluded window.
        mask_bkg = (lc.hi_edges <= excl_start) | (lc.lo_edges >= excl_stop)

        n_bkg_bins = np.sum(mask_bkg)
        if n_bkg_bins < (order + 2):
            raise RuntimeError(
                f"Too few background bins ({n_bkg_bins}) "
                f"for a polynomial of order {order}"
            )

        bkg_model = Polynomial(
            counts=lc.counts[mask_bkg][:, np.newaxis],
            tstart=lc.lo_edges[mask_bkg],
            tstop=lc.hi_edges[mask_bkg],
            exposure=lc.exposure[mask_bkg],
        )
        bkg_model.fit(order=order)

        # interpolate() returns rate, not counts, with shape (N, 1).
        bkg_rate, bkg_rate_err = bkg_model.interpolate(
            tstart=lc.lo_edges,
            tstop=lc.hi_edges,
        )
        bkg_rate = np.squeeze(bkg_rate)
        bkg_rate_err = np.squeeze(bkg_rate_err)

        bkg_counts = bkg_rate * lc.exposure
        bkg_counts_err = bkg_rate_err * lc.exposure
        obs_rate = lc.counts / lc.exposure
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
        best_snr = sanity.get("best_snr", results.get("best_snr"))
        panel_snr = sanity.get("panel_snr") or results.get("panel_snr") or {}
        snr_sel = sanity.get("snr_bin_selection") or results.get("snr_bin_selection") or {}
        ranking = panel_snr.get(best_panel, {})
        t90_window = sanity.get("t90_window")
        selected_bins = np.asarray(sanity.get("selected_bins", []), dtype=int)
        ranking_bins = np.asarray(ranking.get("indices_time", []), dtype=int)

        print("\n" + "=" * 80)
        print("SANITY CHECK: panel selection")
        print("=" * 80)
        print(f"Selected panel (max greedy SNR in own T90): {best_panel}")
        print(f"Selection criterion: {sanity.get('selection_criterion', '')}")
        if t90_window is not None:
            print(
                f"Common T90 window: [{t90_window[0]:.6f}, {t90_window[1]:.6f}] s "
                f"(duration={t90_window[1] - t90_window[0]:.6f} s)"
            )
        print(
            f"Ranking set on {best_panel}: {ranking_bins.size} bins | "
            f"time order={list(ranking_bins)}"
        )
        print(
            f"Ranking counts: d={ranking.get('d', np.nan):.4f} | "
            f"b={ranking.get('b', np.nan):.4f} | "
            f"SNR={best_snr:.4f}"
        )

        print("-" * 80)
        print("Final SNR bin selection inside the common T90 (guide for all panels)")
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
            f"{'panel':<8} {'sel':<5} {'nbins':>7} {'t90_0':>12} {'t90_1':>12} "
            f"{'d_sum':>10} {'b_sum':>10} {'SNR':>8}"
        )
        for p in panels:
            info = panel_snr.get(p, {})
            mark = "<--" if p == best_panel else ""
            print(
                f"{p:<8} {mark:<5} {int(info.get('n_bins', 0)):>7d} "
                f"{info.get('t90_tstart', np.nan):>12.4f} "
                f"{info.get('t90_tstop', np.nan):>12.4f} "
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
        plot=True,
        prefix=""
    ):
        """
        Plot panel and bin-selection diagnostics from an analysis result dict.
        Requires extract_source_data_from_fits(..., sanity_check=True).

        ``save`` writes the figures. ``plot`` shows them. The two are independent.
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
        signal_range = sanity.get("t90_window")
        if signal_range is None and results.get("signal_tstart") is not None:
            signal_range = (results["signal_tstart"], results["signal_tstop"])
        snr_bin_selection = (
            sanity.get("snr_bin_selection")
            or results.get("snr_bin_selection")
        )

        if signal_range is not None:
            t0 = 0.5 * (signal_range[0] + signal_range[1])
        elif sanity.get("selected_time_range"):
            t_lo, t_hi = sanity["selected_time_range"]
            t0 = 0.5 * (t_lo + t_hi)
        else:
            t0 = 0.0

        zoom = 5.0

        fig, axes = plt.subplots(
            3, 2,
            figsize=_PAPER_FIGSIZE,
            sharex=True,
            gridspec_kw={"hspace": 0.38, "wspace": 0.16},
        )
        axes = np.atleast_1d(axes).flatten()

        drew_background = False
        drew_final_bins = False
        selected_bins = np.asarray(
            (snr_bin_selection or {}).get("indices_time", []),
            dtype=int,
        )

        for ax, panel in zip(axes, panels):
            lc = acs_lc[panel]
            res = bkg_fits.get(panel)
            is_best = panel == best_panel
            t = lc.centroids - t0
            t_edges = np.empty(2 * lc.lo_edges.size)
            t_edges[0::2] = lc.lo_edges - t0
            t_edges[1::2] = lc.hi_edges - t0

            if _draw_bin_bars(ax, lc, selected_bins, t0, _PAPER_BINS):
                drew_final_bins = True

            ax.plot(
                t_edges,
                np.repeat(lc.rates, 2),
                color=_PAPER_BLUE,
                lw=0.8,
                zorder=3,
            )

            if res is not None:
                ax.plot(
                    t,
                    res["bkg_rate"],
                    color=_PAPER_RED,
                    lw=0.9,
                    zorder=3,
                )
                drew_background = True

            title = f"{panel} · best panel" if is_best else panel

            in_view = (t >= -zoom) & (t <= zoom)
            if np.any(in_view):
                y_max = np.nanmax(lc.rates[in_view])
                if np.isfinite(y_max) and y_max > 0:
                    ax.set_ylim(0, 1.15 * y_max)

            ax.set_xlim(-zoom, zoom)
            ax.set_title(
                title,
                fontsize=7.5,
                fontweight="bold" if is_best else "regular",
                pad=3,
            )
            _style_paper_ax(ax)

        for ax in axes[len(list(panels)):]:
            ax.axis("off")

        legend_handles = [Line2D([0], [0], color=_PAPER_BLUE, lw=1.2)]
        legend_labels = ["Observed"]
        if drew_background:
            legend_handles.append(Line2D([0], [0], color=_PAPER_RED, lw=1.2))
            legend_labels.append("Background")
        if drew_final_bins:
            legend_handles.append(
                Patch(facecolor=_PAPER_BINS, alpha=0.45, edgecolor="none")
            )
            legend_labels.append("Selected bins")
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
            fontsize=7,
            handlelength=1.6,
            columnspacing=1.1,
        )

        n_shown = len(list(panels))
        bottom = max(0, ((n_shown - 1) // 2) * 2)
        for ax in axes[bottom:n_shown]:
            ax.set_xlabel("Time - T90 center [s]", fontsize=8)
        fig.suptitle("Panel SNR sanity check", fontsize=9, y=0.985)
        fig.subplots_adjust(
            left=0.09, right=0.985, bottom=0.07, top=0.88,
            hspace=0.48, wspace=0.22,
        )

        if save:
            _save_paper_figure(
                fig,
                self.output_dir,
                f"{prefix}_panel_snr_sanity_check",
            )
        _finish_figure(fig, plot)

        self._plot_greedy_snr_by_panel(
            acs_lc,
            panel_snr,
            panels,
            best_panel,
            t0,
            save=save,
            plot=plot,
            prefix=prefix,
        )

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
                    print(f"Saved: {filename}")
                _finish_figure(fig2, plot)

        self._plot_bayesian_blocks_sanity_check(
            results,
            panels,
            best_panel,
            panel_snr,
            t0,
            save=save,
            plot=plot,
            prefix=prefix,
        )

    def _plot_greedy_snr_by_panel(
        self,
        acs_lc,
        panel_snr,
        panels,
        best_panel,
        t0,
        save=False,
        plot=True,
        prefix="",
    ):
        """Greedy SNR of each panel inside that panel's own T90."""

        fig, axes = plt.subplots(
            3, 2,
            figsize=_PAPER_FIGSIZE,
            sharex=True,
            gridspec_kw={"hspace": 0.38, "wspace": 0.16},
        )
        axes = np.atleast_1d(axes).flatten()
        zoom = 5.0
        drew_background = False
        drew_t90 = False
        drew_greedy = False

        for ax, panel in zip(axes, panels):
            lc = acs_lc[panel]
            info = panel_snr.get(panel, {})
            is_best = panel == best_panel
            t90_start = info.get("t90_tstart", np.nan)
            t90_stop = info.get("t90_tstop", np.nan)
            if np.isfinite(t90_start) and np.isfinite(t90_stop):
                ax.axvspan(
                    t90_start - t0,
                    t90_stop - t0,
                    facecolor=_PAPER_T90,
                    edgecolor="none",
                    zorder=0,
                )
                drew_t90 = True

            t = lc.centroids - t0
            if _draw_bin_bars(
                ax, lc, info.get("indices_time", []), t0, _PAPER_BINS
            ):
                drew_greedy = True

            t_edges = np.empty(2 * lc.lo_edges.size)
            t_edges[0::2] = lc.lo_edges - t0
            t_edges[1::2] = lc.hi_edges - t0
            ax.plot(
                t_edges,
                np.repeat(lc.rates, 2),
                color=_PAPER_BLUE,
                lw=0.8,
                zorder=3,
            )

            bkg_rate = info.get("bkg_rate")
            if bkg_rate is not None:
                ax.plot(t, bkg_rate, color=_PAPER_RED, lw=0.9, zorder=3)
                drew_background = True

            snr = info.get("snr", np.nan)
            n_bins = int(info.get("n_bins", 0))
            if n_bins == 0:
                bin_text = "no greedy bins"
            elif n_bins == 1:
                bin_text = "1 greedy bin"
            else:
                bin_text = f"{n_bins} greedy bins"
            if is_best:
                title = f"{panel} · best panel · SNR {snr:.3f} · {bin_text}"
            else:
                title = f"{panel} · SNR {snr:.3f} · {bin_text}"

            in_view = (t >= -zoom) & (t <= zoom)
            if np.any(in_view):
                y_max = np.nanmax(lc.rates[in_view])
                if np.isfinite(y_max) and y_max > 0:
                    ax.set_ylim(0, 1.15 * y_max)

            ax.set_xlim(-zoom, zoom)
            ax.set_title(
                title,
                fontsize=7.5,
                fontweight="bold" if is_best else "regular",
                pad=3,
            )
            _style_paper_ax(ax)

        for ax in axes[len(list(panels)):]:
            ax.axis("off")

        legend_handles = [Line2D([0], [0], color=_PAPER_BLUE, lw=1.2)]
        legend_labels = ["Observed"]
        if drew_background:
            legend_handles.append(Line2D([0], [0], color=_PAPER_RED, lw=1.2))
            legend_labels.append("Background")
        if drew_t90:
            legend_handles.append(Patch(facecolor=_PAPER_T90, edgecolor="none"))
            legend_labels.append("T90")
        if drew_greedy:
            legend_handles.append(
                Patch(facecolor=_PAPER_BINS, alpha=0.45, edgecolor="none")
            )
            legend_labels.append("Greedy bins")
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
            fontsize=7,
            handlelength=1.6,
            columnspacing=1.1,
        )

        n_shown = len(list(panels))
        bottom = max(0, ((n_shown - 1) // 2) * 2)
        for ax in axes[bottom:n_shown]:
            ax.set_xlabel("Time - T90 center [s]", fontsize=8)
        fig.suptitle("Greedy SNR in each panel T90", fontsize=9, y=0.985)
        fig.subplots_adjust(
            left=0.09, right=0.985, bottom=0.07, top=0.88,
            hspace=0.48, wspace=0.22,
        )

        if save:
            _save_paper_figure(
                fig,
                self.output_dir,
                f"{prefix}_greedy_snr_by_panel",
            )
        _finish_figure(fig, plot)

    def _plot_bayesian_blocks_sanity_check(
        self,
        results,
        panels,
        best_panel,
        panel_snr,
        t0,
        save=False,
        plot=True,
        prefix="",
    ):
        """Six-panel figure of each panel's Bayesian-blocks light curve."""

        bb_panels = results.get("bb_panels") or {}
        fig, axes = plt.subplots(
            3, 2,
            figsize=_PAPER_FIGSIZE,
            sharex=True,
            gridspec_kw={"hspace": 0.38, "wspace": 0.16},
        )
        axes = np.atleast_1d(axes).flatten()
        window = 10.0

        for ax, panel in zip(axes, panels):
            lc = results["lc"][panel]
            info = panel_snr.get(panel, {})
            bb = bb_panels.get(panel)
            is_best = panel == best_panel
            t = lc.centroids - t0

            t90_start = info.get("t90_tstart", np.nan)
            t90_stop = info.get("t90_tstop", np.nan)
            if np.isfinite(t90_start) and np.isfinite(t90_stop):
                ax.axvspan(
                    t90_start - t0,
                    t90_stop - t0,
                    facecolor=_PAPER_T90,
                    edgecolor="none",
                    zorder=0,
                )

            t_edges = np.empty(2 * lc.lo_edges.size)
            t_edges[0::2] = lc.lo_edges - t0
            t_edges[1::2] = lc.hi_edges - t0
            ax.plot(
                t_edges,
                np.repeat(lc.rates, 2),
                color=_PAPER_BLUE,
                lw=0.7,
                zorder=2,
            )

            if bb is not None:
                blocks = bb.bb_lightcurve
                ax.plot(
                    np.append(blocks.lo_edges, blocks.hi_edges[-1]) - t0,
                    np.append(blocks.rates, blocks.rates[-1]),
                    drawstyle="steps-post",
                    color=_PAPER_RED,
                    lw=0.75,
                    zorder=3,
                )

            if bb is None:
                title = f"{panel} · Bayesian blocks failed"
            elif is_best:
                title = f"{panel} · best panel"
            else:
                title = panel

            in_view = (t >= -window) & (t <= window)
            if np.any(in_view):
                y_max = np.nanmax(lc.rates[in_view])
                if np.isfinite(y_max) and y_max > 0:
                    ax.set_ylim(0, 1.15 * y_max)

            ax.set_xlim(-window, window)
            ax.set_title(
                title,
                fontsize=8,
                fontweight="bold" if is_best else "regular",
                pad=3,
            )
            _style_paper_ax(ax)

        for ax in axes[len(list(panels)):]:
            ax.axis("off")

        fig.legend(
            [
                Line2D([0], [0], color=_PAPER_BLUE, lw=1.2),
                Line2D([0], [0], color=_PAPER_RED, lw=1.2),
                Patch(facecolor=_PAPER_T90, edgecolor="none"),
            ],
            ["Observed", "Bayesian blocks", "T90"],
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
            fontsize=7,
            handlelength=1.6,
            columnspacing=1.1,
        )

        n_shown = len(list(panels))
        bottom = max(0, ((n_shown - 1) // 2) * 2)
        for ax in axes[bottom:n_shown]:
            ax.set_xlabel("Time - common T90 center [s]", fontsize=8)
        fig.suptitle("Bayesian blocks by panel", fontsize=9, y=0.985)
        fig.subplots_adjust(
            left=0.09, right=0.985, bottom=0.07, top=0.88,
            hspace=0.48, wspace=0.22,
        )

        if save:
            _save_paper_figure(
                fig,
                self.output_dir,
                f"{prefix}_bayesian_blocks_sanity_check",
            )
        _finish_figure(fig, plot)

    def extract_source_data_from_fits(self, fits_path, plot=False, save_plot=False, prefix="", panels=('z1', 'z0', 'x1', 'x0', 'y1', 'y0'), p0=0.05, sanity_check=False):
        """
        Read one GRB FITS file and return source and background counts.

        The same bin indices, chosen on the winning panel inside its T90,
        are summed on every panel.
        """

        t_min, t_max, event_panels = self.open_fits_file(fits_path)
        light_curve = {"t_min": t_min, "t_max": t_max, "panels": event_panels}
        analysis = self.analyze_lc_with_bblocks(
            light_curve,
            p0=p0,
            isRate=False,
            panels=panels,
            sanity_check=sanity_check,
        )

        if analysis["bb_lc"] is None:
            return -1, -1, -1, -1

        t90_tstart = analysis["signal_tstart"]
        t90_tstop = analysis["signal_tstop"]
        signal_range = (t90_tstart, t90_tstop)

        # TT seconds since 2025-01-01 00:00:00.184 UTC.
        event_time_start_unix = 1735689600.184 + t90_tstart

        if plot:
            self.plot_lc(
                analysis["lc_sel"],
                analysis["bb_lc"],
                signal_range,
                save=save_plot,
                prefix=prefix,
            )

        bkg_fits = analysis["bkg_fits"]
        snr_sel = analysis.get("snr_bin_selection") or {}
        selected_indices = np.asarray(snr_sel.get("indices_time", []), dtype=int)

        s_counts = []
        b_counts = []
        for panel in panels:
            lc = analysis["lc"][panel]
            fit = bkg_fits.get(panel)
            if fit is None:
                fit = self.fit_background(lc, signal_range, buffer=1.0, order=2)

            # One mask for source and background. Fall back to the whole T90
            # when the greedy selection is empty.
            if selected_indices.size > 0:
                mask_sig = np.zeros(lc.counts.size, dtype=bool)
                mask_sig[selected_indices] = True
            else:
                mask_sig = _bins_overlapping(lc, t90_tstart, t90_tstop)

            s_counts.append(np.sum(lc.counts[mask_sig]))
            b_counts.append(np.sum(fit["bkg_counts"][mask_sig]))

            if plot:
                self.plot_background(
                    lc,
                    fit,
                    signal_range,
                    s_counts[-1],
                    b_counts[-1],
                    panel,
                    save=save_plot,
                    prefix=prefix,
                    selected_indices=np.flatnonzero(mask_sig),
                )

        s_counts = np.array(s_counts)
        b_counts = np.array(b_counts)

        if sanity_check:
            sanity = analysis.get("sanity") or {}
            sanity["s_counts"] = s_counts
            sanity["b_counts"] = b_counts
            analysis["sanity"] = sanity

        return s_counts, b_counts, event_time_start_unix, analysis

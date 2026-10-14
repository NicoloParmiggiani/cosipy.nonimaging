"""
Source and background counts of a transient seen by the six ACS (BGO) panels.

Pipeline of ``ACSDataAnalyzer.extract_source_data_from_fits``:

1. Read the FITS file: one light curve per panel, and their sum.
2. Bayesian Blocks on the summed light curve: signal window and T90.
3. Polynomial background on every panel, fitted outside the signal window.
4. Choose the bins on the summed light curve (``bin_method``):
     t90            every bin inside the T90
     snr_threshold  bins inside the signal window with SNR > ``snr_threshold``
     greedy         signal-window bins added by decreasing SNR while the
                    cumulative SNR does not decrease
5. On every panel, sum observed (S+B) and background (B) counts of those bins.

With ``debug=True`` every step prints a short summary and draws one figure.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from bctools.analysis import BayesianBlocksLightcurve
from gdt.core.background.binned import Polynomial
from gdt.core.data_primitives import TimeBins

# Column of the FITS COUNT array for each panel.
# SCBA is z, SCBB is y, SCBC is x. Face A0 is "1", face A1 is "0".
_PANEL_INDEX = {"z1": 0, "z0": 1, "y1": 2, "y0": 3, "x1": 4, "x0": 5}
_DEFAULT_PANELS = ("z1", "z0", "x1", "x0", "y1", "y0")

# TT seconds of 2025-01-01 00:00:00.184 UTC. ACS TIME is seconds since then.
_ACS_TIME_UNIX_EPOCH = 1735689600.184

BIN_METHODS = ("t90", "snr_threshold", "greedy")

# Style of the debug figures.
_LC_STYLE = {"color": "blue", "lw": 2}
_BKG_STYLE = {"color": "red", "lw": 2}
_FONT_SIZE = 15


def counts_snr(d, b):
    """(d - b) / sqrt(b) for observed counts d and background b. Zero where b <= 0."""
    d = np.asarray(d, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, (d - b) / np.sqrt(b), 0.0)


def _bins_overlapping(lc, window):
    """Bins whose interval overlaps [tstart, tstop]."""
    return (lc.hi_edges > window[0]) & (lc.lo_edges < window[1])


def _select_greedy(d, b, in_window):
    """Add window bins by decreasing SNR, stop when the cumulative SNR drops.

    A positive-SNR bin can still lower the cumulative SNR, because the
    background it adds grows the denominator. If no bin is kept, the
    brightest bin of the window is used.
    """
    snr = counts_snr(d, b)
    order = np.flatnonzero(in_window)
    order = order[np.argsort(-snr[order])]
    order = order[snr[order] > 0]

    selected = []
    d_sum = b_sum = 0.0
    snr_sum = -np.inf
    for i in order:
        snr_new = counts_snr(d_sum + d[i], b_sum + b[i])
        if snr_new < snr_sum:
            break
        selected.append(i)
        d_sum += d[i]
        b_sum += b[i]
        snr_sum = snr_new

    if not selected and np.any(in_window):
        window = np.flatnonzero(in_window)
        selected = [window[np.argmax(d[window])]]
    return np.sort(np.asarray(selected, dtype=int))


def _li_ma_significance(lc, t90_window):
    """Li & Ma significance of the T90 against the rest of the light curve."""
    tstart, tstop = t90_window
    on = lc.slice(tstart, tstop)
    off = [lc.slice(lc.centroids[0], tstart), lc.slice(tstop, lc.centroids[-1])]
    t_on = np.sum(on.exposure)
    t_off = sum(np.sum(x.exposure) for x in off)
    n_on = np.sum(on.rates * on.exposure)
    n_off = sum(np.sum(x.rates * x.exposure) for x in off)
    if t_off <= 0 or n_on <= 0 or n_off <= 0:
        return 0.0
    alpha = t_on / t_off
    n_tot = n_on + n_off
    return np.sqrt(2) * (
        n_on * np.log((1 + alpha) / alpha * n_on / n_tot)
        + n_off * np.log((1 + alpha) * n_off / n_tot)
    ) ** 0.5


class ACSDataAnalyzer:
    """Run the pipeline described in the module docstring."""

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def extract_source_data_from_fits(
        self,
        fits_path,
        bin_method="t90",
        snr_threshold=3.5,
        panels=_DEFAULT_PANELS,
        p0=0.05,
        bkg_buffer=1.0,
        bkg_order=2,
        debug=False,
    ):
        """Run the whole pipeline on one FITS file.

        Returns ``s_counts, b_counts, t90_start_unix, results``, with the
        counts in the order of ``panels``. Returns ``-1, -1, -1, -1`` when
        the Bayesian Blocks or a background fit fail.
        """
        light_curves = self.read_fits(fits_path, panels)
        results = self.analyze(
            light_curves, bin_method, snr_threshold, p0,
            bkg_buffer, bkg_order, debug,
        )
        if results is None:
            return -1, -1, -1, -1
        t90_start_unix = _ACS_TIME_UNIX_EPOCH + results["t90_window"][0]
        return results["s_counts"], results["b_counts"], t90_start_unix, results

    def read_fits(self, fits_path, panels=_DEFAULT_PANELS):
        """Step 1. One TimeBins light curve per panel, all on the same time grid."""
        with fits.open(fits_path) as hdul:
            # A file without the observation header must fail here.
            hdul[0].header["DATE-OBS"]
            data = hdul[1].data
            t_min = np.asarray(data["TIME"])
            exposure = np.asarray(data["TIMEDEL"][:, 0])
            counts = np.asarray(data["COUNT"])
        t_max = t_min + exposure
        return {
            p: TimeBins(counts[:, _PANEL_INDEX[p]], t_min, t_max, exposure)
            for p in panels
        }

    def analyze(
        self,
        light_curves,
        bin_method="t90",
        snr_threshold=3.5,
        p0=0.05,
        bkg_buffer=1.0,
        bkg_order=2,
        debug=False,
    ):
        """Steps 1-5 on a dict ``panel -> TimeBins``. Returns None on failure."""
        if bin_method not in BIN_METHODS:
            raise ValueError(f"bin_method must be one of: {', '.join(BIN_METHODS)}")
        panels = list(light_curves)

        # 1. Summed light curve: d is the observed counts per bin.
        first = light_curves[panels[0]]
        d = sum(np.asarray(lc.counts, dtype=float) for lc in light_curves.values())
        lc_sum = TimeBins(d, first.lo_edges, first.hi_edges, first.exposure)
        if debug:
            self._debug_light_curves(light_curves)

        # 2. Bayesian Blocks on the sum: signal window, and the T90 inside it.
        try:
            bb = BayesianBlocksLightcurve(lc_sum)
            bb.compute_bayesian_blocks(p0=p0)
            t90 = bb.duration(quantile=0.9)
            t90_err = bb.duration_error(0.9, nsamples=100)
            t90_window = tuple(float(t) for t in bb.quantile_range(quantile=0.9))
            signal_window = (float(bb.signal_range.tstart), float(bb.signal_range.tstop))
        except Exception as exc:
            print(f"WARNING: Bayesian blocks failed on the summed light curve: {exc}")
            return None
        if debug:
            self._debug_bayesian_blocks(lc_sum, bb, t90, t90_err, t90_window, signal_window)

        # 3. Background of every panel. The whole signal window is excluded,
        #    not only the T90, so the burst tails do not enter the fit.
        try:
            bkg_fits = {
                p: self.fit_background(lc, signal_window, bkg_buffer, bkg_order)
                for p, lc in light_curves.items()
            }
        except RuntimeError as exc:
            print(f"WARNING: background fit failed: {exc}")
            return None
        b = sum(fit["bkg_counts"] for fit in bkg_fits.values())
        if debug:
            self._debug_background(light_curves, bkg_fits, signal_window, bkg_buffer)

        # 4. Bins chosen once, on the summed d and b.
        if bin_method == "t90":
            selected = np.flatnonzero(_bins_overlapping(lc_sum, t90_window))
        elif bin_method == "snr_threshold":
            in_signal = _bins_overlapping(lc_sum, signal_window)
            selected = np.flatnonzero(in_signal & (counts_snr(d, b) > snr_threshold))
            if selected.size == 0:
                print(f"WARNING: no bin has SNR > {snr_threshold}. Panel counts will be zero.")
        else:
            selected = _select_greedy(d, b, _bins_overlapping(lc_sum, signal_window))
        snr = float(counts_snr(d[selected].sum(), b[selected].sum()))
        significance = _li_ma_significance(lc_sum, t90_window)
        if debug:
            self._debug_selection(
                lc_sum, b, selected, snr, significance, bin_method, snr_threshold,
                t90_window,
            )

        # 5. The same bins on every panel.
        s_counts = np.array([light_curves[p].counts[selected].sum() for p in panels])
        b_counts = np.array([bkg_fits[p]["bkg_counts"][selected].sum() for p in panels])
        if debug:
            self._debug_panel_counts(panels, s_counts, b_counts)

        return {
            "panels": panels,
            "lc": light_curves,
            "lc_sum": lc_sum,
            "bb": bb,
            "signal_window": signal_window,
            "t90_window": t90_window,
            "t90": t90,
            "t90_err": t90_err,
            "bkg_fits": bkg_fits,
            "d": d,
            "b": b,
            "bin_method": bin_method,
            "snr_threshold": snr_threshold if bin_method == "snr_threshold" else None,
            "selected_bins": selected,
            "snr": snr,
            "significance": significance,
            "s_counts": s_counts,
            "b_counts": b_counts,
        }

    def fit_background(self, lc, signal_window, buffer=0.0, order=2):
        """Polynomial fit of the bins fully outside the signal window +/- buffer.

        ``bkg_rate`` and ``bkg_counts`` are the model evaluated on every bin,
        ``mask_bkg`` marks the bins used in the fit.
        """
        excl_start = signal_window[0] - buffer
        excl_stop = signal_window[1] + buffer
        mask_bkg = (lc.hi_edges <= excl_start) | (lc.lo_edges >= excl_stop)
        if mask_bkg.sum() < order + 2:
            raise RuntimeError(
                f"Too few background bins ({mask_bkg.sum()}) for a polynomial of order {order}"
            )

        model = Polynomial(
            counts=lc.counts[mask_bkg][:, np.newaxis],
            tstart=lc.lo_edges[mask_bkg],
            tstop=lc.hi_edges[mask_bkg],
            exposure=lc.exposure[mask_bkg],
        )
        model.fit(order=order)
        # interpolate() returns a rate of shape (N, 1).
        bkg_rate, _ = model.interpolate(tstart=lc.lo_edges, tstop=lc.hi_edges)
        bkg_rate = np.squeeze(bkg_rate)
        return {
            "model": model,
            "mask_bkg": mask_bkg,
            "bkg_rate": bkg_rate,
            "bkg_counts": bkg_rate * lc.exposure,
        }

    # ------------------------------------------------------------------
    # Debug output: one print and one figure per pipeline step
    # ------------------------------------------------------------------

    def _debug_light_curves(self, light_curves):
        lc = next(iter(light_curves.values()))
        print("[1] Light curves")
        print(f"    {lc.size} bins, width {np.median(lc.exposure):.4g} s, "
              f"from {lc.lo_edges[0]:.3f} to {lc.hi_edges[-1]:.3f} s")
        for p, panel_lc in light_curves.items():
            print(f"    {p}: {panel_lc.counts.sum():.0f} counts")

        fig, axes, t0 = self._panel_grid("Step 1: panel light curves", lc)
        for ax, (p, panel_lc) in zip(axes, light_curves.items()):
            _plot_rates(ax, panel_lc, t0, **_LC_STYLE)
            ax.set_title(p)
        _show(fig)

    def _debug_bayesian_blocks(self, lc_sum, bb, t90, t90_err, t90_window, signal_window):
        print("[2] Bayesian Blocks on the summed light curve")
        print(f"    {bb.bb_lightcurve.size} blocks")
        print(f"    signal window [{signal_window[0]:.3f}, {signal_window[1]:.3f}] s")
        print(f"    T90 window    [{t90_window[0]:.3f}, {t90_window[1]:.3f}] s")
        print(f"    T90 = {t90:.3f} s  (-{t90_err[0]:.3f} / +{t90_err[1]:.3f})")

        fig, ax, t0 = _t90_axes(t90_window)
        _plot_rates(ax, lc_sum, t0, **_LC_STYLE, label="Summed counts")
        _plot_rates(ax, bb.bb_lightcurve, t0, color="red", lw=2, label="Bayesian blocks")
        ax.set_title("Step 2: Bayesian Blocks, T90 and signal window")
        ax.legend()
        _show(fig)

    def _debug_background(self, light_curves, bkg_fits, signal_window, buffer):
        print("[3] Background fit (bins outside the signal window +/- buffer)")
        for p, fit in bkg_fits.items():
            print(f"    {p}: {fit['mask_bkg'].sum()} bins in the fit, "
                  f"mean rate {fit['bkg_rate'].mean():.1f} counts/s")

        first = next(iter(light_curves.values()))
        fig, axes, t0 = self._panel_grid("Step 3: background fit (grey = excluded)", first)
        for ax, (p, lc) in zip(axes, light_curves.items()):
            _plot_rates(ax, lc, t0, **_LC_STYLE)
            ax.plot(lc.centroids - t0, bkg_fits[p]["bkg_rate"], **_BKG_STYLE)
            ax.axvspan(signal_window[0] - buffer - t0, signal_window[1] + buffer - t0,
                       color="0.85", zorder=0)
            ax.set_title(p)
        _show(fig)

    def _debug_selection(self, lc_sum, b, selected, snr, significance, bin_method,
                         snr_threshold, t90_window):
        rule = f"{bin_method} (SNR > {snr_threshold})" if bin_method == "snr_threshold" else bin_method
        d_sel = lc_sum.counts[selected].sum()
        b_sel = b[selected].sum()
        print(f"[4] Bin selection on the summed light curve: {rule}")
        print(f"    {selected.size} bins, indices {selected.tolist()}")
        print(f"    d = {d_sel:.1f}, b = {b_sel:.1f}, SNR = {snr:.2f}")
        print(f"    Li & Ma significance in the T90: {significance:.2f}")

        fig, ax, t0 = _t90_axes(t90_window)
        _shade_bins(ax, lc_sum, selected, t0)
        _plot_rates(ax, lc_sum, t0, **_LC_STYLE, label="Summed counts")
        ax.plot(lc_sum.centroids - t0, b / lc_sum.exposure, **_BKG_STYLE, label="Summed background")
        ax.set_title(f"Step 4: {rule}, {selected.size} bins, SNR = {snr:.2f}")
        ax.legend()
        _show(fig)

    def _debug_panel_counts(self, panels, s_counts, b_counts):
        print("[5] Counts in the selected bins")
        print(f"    {'panel':<6}{'S+B':>10}{'B':>10}{'S':>10}")
        for p, s, b in zip(panels, s_counts, b_counts):
            print(f"    {p:<6}{s:>10.1f}{b:>10.1f}{s - b:>10.1f}")

        x = np.arange(len(panels))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.2, s_counts, width=0.4, label="S+B")
        ax.bar(x + 0.2, b_counts, width=0.4, label="B")
        ax.set_xticks(x, panels)
        ax.set_ylabel("Counts")
        ax.set_title("Step 5: counts per panel in the selected bins")
        ax.legend()
        _show(fig)

    def _panel_grid(self, title, lc):
        """3 x 2 panel figure, time measured from the center of the light curve."""
        t0 = 0.5 * (lc.lo_edges[0] + lc.hi_edges[-1])
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        fig.suptitle(title)
        axes[0, 0].set_xlim(-30, 30)
        for ax in axes[-1]:
            ax.set_xlabel("Time - window center [s]")
        for ax in axes[:, 0]:
            ax.set_ylabel("Counts / s")
        return fig, axes.flatten(), t0


def _show(fig):
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(_FONT_SIZE)
        ax.yaxis.label.set_fontsize(_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=_FONT_SIZE)
        if ax.get_legend() is not None:
            ax.legend(fontsize=_FONT_SIZE)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _plot_rates(ax, lc, t0, **kwargs):
    """Step plot of the rates of a TimeBins, with time shifted by t0."""
    edges = np.append(lc.lo_edges, lc.hi_edges[-1]) - t0
    ax.stairs(lc.rates, edges, baseline=None, **kwargs)


def _t90_axes(t90_window):
    """Figure centered on the T90 (+/- 5 s), with the T90 as a light green band."""
    t0 = 0.5 * (t90_window[0] + t90_window[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(t90_window[0] - t0, t90_window[1] - t0, color="#b7e4c7", zorder=0, label="T90")
    ax.set_xlim(-5, 5)
    ax.set_xlabel("Time - T90 center [s]")
    ax.set_ylabel("Counts / s")
    return fig, ax, t0


def _shade_bins(ax, lc, indices, t0):
    """Dark green full-height band over each run of consecutive selected bins."""
    label = "Selected bins"
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        if run.size:
            ax.axvspan(lc.lo_edges[run[0]] - t0, lc.hi_edges[run[-1]] - t0,
                       color="#1b7f3b", alpha=0.6, zorder=1, label=label)
            label = None

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
    
    def analyze_lc_with_bblocks(self,lightcurve, p0=0.05, isRate=False,panels=['z1', 'z0', 'x1', 'x0', 'y1', 'y0']):
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

        Output:
            dictionary containing:
                {
                    "lc_sel": selected lightcurve,
                    "bb_lc": BayesianBlocksLightcurve object,
                    "lc": dictionary of all lightcurves,
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

        # se sono rate → converto a counts
        if isRate:
            for k in signal:
                signal[k] = signal[k] * exposure

        # =========================
        # COSTRUZIONE LIGHT CURVES
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
        # BEST PANEL SELECTION
        # =========================
        best_panel = None
        best_value = float('-inf')

        for panel in panels:
            value = np.max(lc[panel].counts)

            if value > best_value:
                best_value = value
                best_panel = panel

        lc_sel = lc[best_panel]

        # =========================
        # BAYESIAN BLOCKS
        # =========================
        try:
            bb_lc = BayesianBlocksLightcurve(lc_sel)
            bb_lc.compute_bayesian_blocks(p0=p0)

            signal_range = bb_lc.signal_range

            t90 = bb_lc.duration(quantile=.9)
            t90_error = bb_lc.duration_error(.9, nsamples=100)

        except Exception as e:
            print(e)
            print("WARNING")
            return {
                "lc_sel": lc_sel,
                "bb_lc": None,
                "lc": None,
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

        # =========================
        # SIGNAL + BACKGROUND
        # =========================
        signal_lc = lc_sel.slice(signal_range.tstart, signal_range.tstop)

        # background prima e dopo (FIX rispetto al tuo codice originale)
        bkg_lc1 = lc_sel.slice(lc_sel.centroids[0], signal_range.tstart)
        bkg_lc2 = lc_sel.slice(signal_range.tstop, lc_sel.centroids[-1])

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

        # =========================
        # PEAK SIGNIFICANCE
        # =========================
        significance = []

        for rate, exp in zip(signal_lc.rates, signal_lc.exposure):
            N_on_bin = rate * exp
            alpha_bin = exp / t_off if t_off > 0 else 0

            if alpha_bin > 0 and N_on_bin > 0 and N_off > 0:
                S_bin = np.sqrt(2) * (
                    N_on_bin * np.log(((1 + alpha_bin) / alpha_bin) *
                                    (N_on_bin / (N_on_bin + N_off))) +
                    N_off * np.log((1 + alpha_bin) *
                                (N_off / (N_on_bin + N_off)))
                )**0.5
            else:
                S_bin = 0

            significance.append(S_bin)

        significance = np.array(significance)
        S_peak = np.max(significance)

        # =========================
        # OUTPUT
        # =========================
        return {
            "lc_sel": lc_sel,
            "bb_lc": bb_lc,
            "lc": lc,
            "signal_tstart": signal_range.tstart,
            "signal_tstop": signal_range.tstop,
            "t90": t90,
            "t90_err_low": t90_error[0],
            "t90_err_high": t90_error[1],
            "significance": S,
            "significance_peak": S_peak,
            "t_min": t_min,
            "t_max": t_max
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

        tstart_sig = signal_range.tstart
        tstop_sig = signal_range.tstop
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

        # ATTENZIONE: interpolate() restituisce RATE, non counts
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

    def plot_lc(self, lc_sel, bb_lc, save=False,prefix=""):

        signal_range = bb_lc.signal_range

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

        plt.plot(
            np.append(lc_bayes.lo_edges, lc_bayes.hi_edges[-1]),
            np.append(lc_bayes.rates, lc_bayes.rates[-1]),
            drawstyle='steps-post',
            label='Bayesian blocks',
            color="#1f77b4"
        )

        # Vertical lines showing the start and stop of the identified signal
        plt.axvline(
            bb_lc.signal_range.tstart,
            ls="--",
            color='olive',
            label="Signal start/stop"
        )

        plt.axvline(
            bb_lc.signal_range.tstop,
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
        prefix=""
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

        plt.axvspan(
            signal_range.tstart,
            signal_range.tstop,
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

    def extract_source_data_from_fits(self,fits_path,plot=False,save_plot=False,prefix="",panels = ['z1','z0','x1','x0','y1','y0']):
        
        t_min,t_max,event_panels = self.open_fits_file(fits_path)
        
        light_curve = {"t_min":t_min,"t_max":t_max,"panels":event_panels}
        
        # compute the TimeBins LC, time_start, time_stop, t90 and Li&Ma signfiicance
        bblocks_analysis_results = self.analyze_lc_with_bblocks(light_curve, p0=10e-5, isRate=False, panels=panels)
        
        acs_lc = bblocks_analysis_results['lc']
        event_time_start = bblocks_analysis_results['signal_tstart']
        lc_sel = bblocks_analysis_results['lc_sel']
        bb_lc = bblocks_analysis_results['bb_lc']
        
        if bb_lc is None:
            return -1,-1,-1,-1
        signal_range = bb_lc.signal_range
        
        #convert event time start from TT to Unix time stamp
        mjd_ref_timestamp = 1735689669.184
        event_time_start_unix = mjd_ref_timestamp + event_time_start
        
        if plot:
            self.plot_lc(lc_sel,bb_lc,save=save_plot,prefix=prefix)

        results = []
        s_counts = []
        b_counts = []


        tstart = signal_range.tstart
        tstop = signal_range.tstop
        duration = tstop-tstart
        for p in panels:
        
            lc = acs_lc[p]
            res = self.fit_background(lc, signal_range, buffer=1.0, order=2)

            results.append(res)
            
            # maschera della finestra del segnale
            mask_sig = (lc.lo_edges >= tstart) & (lc.hi_edges <= tstop)

            # somme nella finestra [tstart, tstop]
            signal_counts = np.sum(lc.counts[mask_sig])
            background_counts = np.sum(res["bkg_counts"][mask_sig])
           
            s_counts.append(signal_counts)
            b_counts.append(round(background_counts,3))
            
            if plot:
                self.plot_background(lc,res,signal_range,signal_counts,background_counts,p,save=save_plot,prefix=prefix)
               
                
        # obtain numpy arrays
        s_counts = np.array(s_counts)
        b_counts = np.array(b_counts)
       
        print("Signal counts per panel:", s_counts)
        print("Background counts per panel:", b_counts)
        
    
        return s_counts,b_counts,event_time_start_unix,bblocks_analysis_results

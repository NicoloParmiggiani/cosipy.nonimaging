import pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic, ICRS
from bctools.loc import TSMap, NormLocLike
from scoords import Attitude, SpacecraftFrame
import healpy as hp
from mhealpy.containers.healpix_map import HealpixMap
from matplotlib.lines import Line2D
from astropy.io import fits
from gdt.core.background.binned import Polynomial
from gdt.core.data_primitives import TimeBins
from bctools.analysis import BayesianBlocksLightcurve

class BGOLocalizerBCT:
    """
    Simple class to localize GRBs using three BGO Look-Up Tables (soft/medium/hard).

    Parameters
    ----------
    soft_loctable_path : str
        Path to the pickled LUT for the 'soft' spectrum.
    medium_loctable_path : str
        Path to the pickled LUT for the 'medium' spectrum.
    hard_loctable_path : str
        Path to the pickled LUT for the 'hard' spectrum.
    nside : int
        HEALPix Nside used to compute TS maps.

    Notes
    -----
    The input counts (s_counts and b_counts) must follow the correct order
    of BGO detector panels to match the LUT definition:

        
        ['BGO_Z1', 'BGO_Z0', 'BGO_X1', 'BGO_X0', 'BGO_Y1', 'BGO_Y0']
    """

    def __init__(self, soft_loctable_path, medium_loctable_path, hard_loctable_path, output_dir, nside=64 ):
        # Store HEALPix Nside
        self.nside = nside
        self.output_dir = output_dir
        # Load all three IRFs at initialization
        # The IRFs have been generated using the bc-tools package:
        #soft_spectrum = BandFunction._from_megalib(['BandFunction',10,10000,-1.9,-3.7,230],"10.0")
        #medium_spectrum = BandFunction._from_megalib(['BandFunction',10,10000,-1,-2.3,699.9],"10.0")
        #hard_spectrum = Comptonized._from_megalib(['Comptonized',10,10000,-0.5,1500],"10.0")
        #soft_local_loctable = LocalLocTable.from_irf(irf, soft_spectrum,energy_channels = 1) # [80,2000]
        #medium_local_loctable = LocalLocTable.from_irf(irf, medium_spectrum,energy_channels = 1)
        #hard_local_loctable = LocalLocTable.from_irf(irf, hard_spectrum,energy_channels = 1)
        
             
        self.loctables = {
            "soft": self._load_pickle(soft_loctable_path),
            "medium": self._load_pickle(medium_loctable_path),
            "hard": self._load_pickle(hard_loctable_path),
        }
    
        
    def open_fits_file(self,fits_path):
        
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

    def localize(self, s_counts, b_counts, attitude=None, conf_level=0.9, duration=1):
        """
        Run localization using all LUTs and return the one with the highest TS.

        Parameters
        ----------
        s_counts : list or np.ndarray
            Source counts.
        b_counts : list or np.ndarray
            Background counts.
        attitude : Attitude or None
            Spacecraft attitude. If provided, output is in Galactic (l, b).
            If None, output is in instrument coordinates (theta, phi).
        conf_level : float, optional
            Confidence level for error region (default = 0.9).

        Returns
        -------
        dict
            Best localization result.
        """
        
        #s_counts = [100000,2000,2000,2000,2000,2000]
        #b_counts = [1000,1000,1000,1000,1000,1000]
        
        #bgo_z1[keV] 1591
        #bgo_z0[keV] 605
        #bgo_x1[keV] 16835
        #bgo_x0[keV] 257
        #bgo_y1[keV] 1291
        #bgo_y0[keV] 797
                
        #remap the counts following BGO IRF ['BGO_X0', 'BGO_X1', 'BGO_Y0', 'BGO_Y1', 'BGO_Z0', 'BGO_Z1']

    
        #s_counts = np.array([257,16835,797,1491,605,1591])+np.random.poisson([100,100,100,100,100,100])
        #b_counts = [100,100,100,100,100,100]
        s_counts = s_counts[[3, 2, 5, 4, 1, 0]]
        b_counts = b_counts[[3, 2, 5, 4, 1, 0]]

        results = []
    
        if attitude is None:
            soft_sky_loctable = self.loctables['soft'].to_skyloctable(
                attitude=[0, 0, 0, 1],
                duration=duration,
                frame='icrs'
            )
            medium_sky_loctable = self.loctables['medium'].to_skyloctable(
                attitude=[0, 0, 0, 1],
                duration=duration,
                frame='icrs'
            )
            hard_sky_loctable = self.loctables['hard'].to_skyloctable(
                attitude=[0, 0, 0, 1],
                duration=duration,
                frame='icrs'
            )
            coordsys = "icrs"
        else:
            q = attitude.as_quat()

            soft_sky_loctable = self.loctables['soft'].to_skyloctable(
                attitude=q,
                duration=duration,
                frame='galactic'
            )
            medium_sky_loctable = self.loctables['medium'].to_skyloctable(
                attitude=q,
                duration=duration,
                frame='galactic'
            )
            hard_sky_loctable = self.loctables['hard'].to_skyloctable(
                attitude=q,
                duration=duration,
                frame='galactic'
            )
            coordsys = "galactic"

        self.luts = {
            "soft": soft_sky_loctable,
            "medium": medium_sky_loctable,
            "hard": hard_sky_loctable,
        }
        
        for label, lut in self.luts.items():
            lut.set_background(b_counts)
            lut.set_data(s_counts)

            ts_map = TSMap(nside=self.nside, coordsys=coordsys)
            likelihood = NormLocLike(lut)
            ts_map.compute(likelihood)

            ts_value = float(np.max(ts_map))
            best = ts_map.best_loc()

            cont_area = ts_map.error_area(cont=conf_level).to(u.deg**2)
            eq_radius = np.sqrt(cont_area / np.pi).to(u.deg)

            theta_out = -1.0
            phi_out = -1.0
            l_out = -1.0
            b_out = -1.0

            if attitude is None:
                phi_out = float(best.spherical.lon.deg)
                theta_out = float(90.0 - best.spherical.lat.deg)
            else:
                l_out = float(best.l.deg)
                b_out = float(best.b.deg)
                phi_out = float(best.l.deg)
                theta_out = float(90.0 - best.b.deg)

            results.append({
                "theta_out": theta_out,
                "phi_out": phi_out,
                "l": l_out,
                "b": b_out,
                "label": label,
                "ts_map": ts_map,
                "ts_value": ts_value,
                "sqrt_ts": float(np.sqrt(ts_value)),
                "cont_area_deg2": float(cont_area.value),
                "eq_radius_deg": float(eq_radius.value),
            })

        return max(results, key=lambda r: r["ts_value"])

    def print_loc_table_names(self):
        
        print(f"Soft Look-up tables: {self.luts['soft'].labels}")
        print(f"Medium Look-up tables: {self.luts['medium'].labels}")
        print(f"Hard Look-up tables: {self.luts['hard'].labels}")

    def plot_loc_table(self,panel_name):
        
        sky_loctable = self.luts['soft']
        sky_loctable.get_expectation_map(panel_name).plot()
        sky_loctable = self.luts['medium']
        sky_loctable.get_expectation_map(panel_name).plot()
        sky_loctable = self.luts['hard']
        sky_loctable.get_expectation_map(panel_name).plot()


    def localize_old(self, s_counts, b_counts, attitude=None, conf_level=0.9,duration=1):
        """
        Run localization using all LUTs and return the one with the highest TS.

        Parameters
        ----------
        s_counts : list or np.ndarray
            Source counts.
        b_counts : list or np.ndarray
            Background counts.
        attitude : Attitude or None
            Spacecraft attitude. If provided, output is in Galactic (l, b).
            If None, output is in instrument coordinates (theta, phi).
        conf_level : float, optional
            Confidence level for error region (default = 0.9).

        Returns
        -------
        dict
            Best localization result.
        """
        #['BGO_Z1', 'BGO_Z0', 'BGO_X1', 'BGO_X0', 'BGO_Y1', 'BGO_Y0']
        #remap the counts following BGO IRF ['BGO_X0', 'BGO_X1', 'BGO_Y0', 'BGO_Y1', 'BGO_Z0', 'BGO_Z1']
        
        s_counts = [2000,4000,2000,2000,2000]
        b_counts = [1000,1000,1000,1000,1000]
        
        #s_counts = s_counts[[3, 2, 5, 4, 1, 0]]
        #b_counts = b_counts[[3, 2, 5, 4, 1, 0]]
        
        print(s_counts)
        print(b_counts)

        results = []
        
        # The local_loctable contains the expected rates in spacecraft coordinates
        # We now need to use this to estimate the total expected counts in sky coordinate for
        # the full duration of an event. 
        # In this case we simply have a 1 second event and specifying the attitude by a quaternion
        # ([0,0,0,1] corresponds to the identity rotation). You can have multiple attitude-duration
        # pairs to correctly model long duration events.
        
        print(attitude.as_quat())

        soft_sky_loctable = self.loctables['soft'].to_skyloctable(attitude = attitude.as_quat(), duration = duration,frame='galactic')
        medium_sky_loctable = self.loctables['medium'].to_skyloctable(attitude = attitude.as_quat(), duration = duration,frame='galactic')
        hard_sky_loctable = self.loctables['hard'].to_skyloctable(attitude = attitude.as_quat(), duration = duration,frame='galactic')

        #soft_sky_loctable = self.loctables['soft'].to_skyloctable(attitude = [0,0,0,1], duration = duration)
        #medium_sky_loctable = self.loctables['medium'].to_skyloctable(attitude = [0,0,0,1], duration = duration)
        #hard_sky_loctable = self.loctables['hard'].to_skyloctable(attitude = [0,0,0,1], duration = duration)
      
        self.luts = {
            "soft": soft_sky_loctable,
            "medium": medium_sky_loctable,
            "hard": hard_sky_loctable,
        }

        for label, lut in self.luts.items():

            lut.set_background(b_counts)
            lut.set_data(s_counts)

            ts_map = TSMap(nside=self.nside, coordsys="galactic")
            likelihood = NormLocLike(lut)
            ts_map.compute(likelihood)

            ts_value = float(np.max(ts_map))
            best = ts_map.best_loc()
            
            print(best)

            cont_area = ts_map.error_area(cont=conf_level).to(u.deg**2)
            eq_radius = np.sqrt(cont_area / np.pi).to(u.deg)
            
            
            # Default outputs
            theta_out = -1.0
            phi_out = -1.0
            l_out = -1.0
            b_out = -1.0
            
            #best_gal = best.transform_to(Galactic())
            best_gal = best
            l_out = float(best_gal.l.deg)
            b_out = float(best_gal.b.deg)

            """ if attitude is None:
                # best is assumed to be in local/spacecraft-like coordinates
                phi = best.spherical.lon.to(u.rad).value
                lat = best.spherical.lat.to(u.rad).value

                theta_out = np.pi / 2.0 - lat   # zenith
                phi_out = phi                   # azimuth

                l_out = -1
                b_out = -1

                out_map = ts_map.data

            else:
                # interpret best as local coordinates and convert to galactic
                phi = best.spherical.lon
                lat = best.spherical.lat

                position_sc = SkyCoord(
                    lon=phi,
                    lat=lat,
                    frame=SpacecraftFrame(attitude=attitude)
                )

                position_gal = position_sc.transform_to(Galactic())
                
                
                print("position_sc =", position_sc)
                print("position_sc frame =", position_sc.frame)
                print("position_sc frame name =", position_sc.frame.name)

                position_gal = position_sc.transform_to(Galactic())

                print("position_gal =", position_gal)
                print("position_gal frame =", position_gal.frame)
                print("position_gal frame name =", position_gal.frame.name)
                print("position_gal class =", type(position_gal))


                theta_out = -1
                phi_out = -1
                
                print(position_gal)

                l_out = float(position_gal.l.deg)
                b_out = float(position_gal.b.deg)

                ts_map_rot = self.rotate_tsmap(ts_map, attitude)

                out_map = HealpixMap(
                    ts_map_rot,
                    nside=ts_map.nside,
                    coordsys=ts_map.coordsys
                )
                 """
                
                
            results.append({
                "theta_out":theta_out,
                "phi_out":phi_out,
                "l":l_out,
                "b":b_out,
                "label": label,
                "ts_map": ts_map,
                "ts_value": ts_value,
                "sqrt_ts": float(np.sqrt(ts_value)),
                "cont_area_deg2": float(cont_area.value),
                "eq_radius_deg": float(eq_radius.value),
            })

        return max(results, key=lambda r: r["ts_value"])

    def plot(self, result, true_coord=None, show=True, save_path=None):
        """
        Plot the TS map from a localization result.

        Parameters
        ----------
        result : dict
            Output from `localize()`. Must contain a 'ts_map' (TSMap) and
            summary fields like 'label', 'sqrt_ts', 'ra_deg', 'dec_deg',
            'eq_radius_deg'.
        true_coord : astropy.coordinates.SkyCoord, optional
            True source position to overlay on the map (assumed ICRS RA/Dec).
        show : bool
            If True, display the plot with plt.show() and return None.
            If False, return (img, moll) as given by TSMap.plot().
        save_path : str or None, optional
            If provided, save the plot to the given path as a PNG file.

        Returns
        -------
        tuple or None
            (img, moll) when show=False, otherwise None.
        """
 

        ts_map = result["ts_map"]
        img, ax = ts_map.plot()
        ax.grid(alpha=0.5)
        
        if true_coord is not None:
            # Actual location of simulated source
            ax.scatter(
                true_coord.icrs.ra.to(u.deg).value,
                true_coord.icrs.dec.to(u.deg).value,
                color="red",
                transform=ax.get_transform("world"),
                s=2,
                label="True source"
            )
        best_loc = SkyCoord(l=result['l']*u.deg, b=result['b']*u.deg, frame="galactic")
        ax.scatter(
                
                best_loc.icrs.ra.to(u.deg).value,
                best_loc.icrs.dec.to(u.deg).value,
                color="#1f77b4",
                transform=ax.get_transform("world"),
                s=2,
                label="Best localization"
            )
        
        # Add legend 
        ax.legend(loc="upper right", frameon=True)

        # Compose a title with key localization numbers.
        title = (
            f"{result['label']}  "
            f"sqrt(TS)={result['sqrt_ts']:.2f}  "
            f"l={result['l']:.2f}  "
            f"b={result['b']:.2f}  "
            f"area={result['cont_area_deg2']:.2f} deg^2"
        )
        ax.set_title(title)

        # Save to PNG if path provided
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot salvato in: {save_path}")

        if show:
            plt.show()
            return None

        return img, ax


    @staticmethod
    def _load_pickle(path):
        """Load a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

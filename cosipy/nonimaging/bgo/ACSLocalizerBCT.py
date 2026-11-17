import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic, ICRS
from scipy.optimize import NonlinearConstraint, differential_evolution
from scipy.special import logsumexp
from scipy.stats import binom, kstest, norm
from mhealpy import HealpixMap
from bctools.loc import TSMap, SkyMap, NormLocLike
from scoords import Attitude, SpacecraftFrame

from cosipy.nonimaging.bgo.panels import to_array

# Panel order of the detectors in the bctools IRF
IRF_PANEL_ORDER = (
    "ACS-C-X0",
    "ACS-C-X1",
    "ACS-B-Y0",
    "ACS-B-Y1",
    "ACS-A-Z0",
    "ACS-A-Z1",
)


class ACSLocalizerBCT:
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
    The input counts (s_counts and b_counts) are dicts panel name -> value.
    They are reordered internally to the IRF panel order, ``IRF_PANEL_ORDER``.
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
    
    
    def localize(self, s_counts, b_counts, attitude=None, conf_level=0.9, duration=1):
        """
        Run localization using all LUTs and return the one with the highest TS.

        Parameters
        ----------
        s_counts : dict
            Source counts, panel name -> value.
        b_counts : dict
            Background counts, panel name -> value.
        attitude : Attitude or None
            Spacecraft attitude. If provided, `l`/`b` are Galactic and
            `theta_out`/`phi_out` are converted back to spacecraft
            coordinates. If None, `theta_out`/`phi_out` are already
            spacecraft coordinates.
        conf_level : float, optional
            Confidence level for error region (default = 0.9).

        Returns
        -------
        dict
            Best localization result.
        """
        
        s_counts = to_array(s_counts, IRF_PANEL_ORDER)
        b_counts = to_array(b_counts, IRF_PANEL_ORDER)

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


            likelihood = NormLocLike(lut)
            ts_map = TSMap(nside=self.nside, coordsys=coordsys)
            ts_map.compute(likelihood)
            
            sky_map = SkyMap(nside=self.nside, coordsys=coordsys)
            sky_map.compute(likelihood)
            
            print(sky_map)

            ts_value = float(np.max(ts_map))
            best = ts_map.best_loc()
                
            cont_area = ts_map.error_area(cont=conf_level).to(u.deg**2)
            eq_radius = np.sqrt(cont_area / np.pi).to(u.deg)

            theta_out = -1.0
            phi_out = -1.0
            l_out = -1.0
            b_out = -1.0

            if attitude is None:
                # Identity attitude: ICRS is aligned with the spacecraft frame.
                phi_out = float(best.spherical.lon.deg)
                theta_out = float(90.0 - best.spherical.lat.deg)
            else:
                l_out = float(best.l.deg)
                b_out = float(best.b.deg)
                best_sc = best.transform_to(SpacecraftFrame(attitude=attitude))
                phi_out = float(best_sc.spherical.lon.deg)
                theta_out = float(90.0 - best_sc.spherical.lat.deg)

            results.append({
                "theta_out": theta_out,
                "phi_out": phi_out,
                "l": l_out,
                "b": b_out,
                "label": label,
                "ts_map": ts_map,
                "sky_map": sky_map,
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

    def plot_loc_table(self, panel_name):
        """
        Plot expected counts for one panel from the soft/medium/hard sky LUTs.

        Call ``localize()`` first: this method uses ``self.luts``, which is
        created there from the attitude of that call.
        """

        if not hasattr(self, "luts") or self.luts is None:
            raise RuntimeError("Call localize() before plot_loc_table().")

        for label in ("soft", "medium", "hard"):
            sky_loctable = self.luts[label]
            sky_loctable.get_expectation_map(panel_name).plot()
            plt.title(f"{label} | {panel_name}")
            plt.show()


    
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

    def plot_skymap(
        self,
        values,
        true_l,
        true_b,
        loc_l,
        loc_b,
        conf_level=0.9,
        scheme=None,
        coordsys="galactic",
        show=True,
        save_path=None,
        title=None,
    ):
        """
        Plot the X% containment region of a probability HEALPix map.

        Pixels are ranked by probability and kept until their sum reaches
        ``conf_level`` (same definition as ``smooth_skymap_area``). The
        rest of the sky is set to 0 and drawn with ``HealpixMap.plot``.

        Parameters
        ----------
        values : array or map object
            Probability map. An array, or an object with ``_data``
            (and optionally ``scheme``).
        true_l, true_b : float
            True Galactic coordinates in degrees.
        loc_l, loc_b : float
            Reconstructed Galactic coordinates in degrees.
        conf_level : float
            Containment fraction, default 0.9.
        scheme : {'ring', 'nested'} or None
            HEALPix ordering of ``values``. If None, use the map object's
            ``scheme`` when available, otherwise ``ring``.
            ``smooth_skymap_area`` returns RING maps.
        coordsys : str
            Coordinate system of the map, default ``galactic``.
        show : bool
            If True, display the figure.
        save_path : str or None
            Optional PNG path.
        title : str or None
            Optional axes title. Defaults to the containment area.

        Returns
        -------
        tuple or None
            ``(img, ax)`` when ``show=False``, otherwise None.
        """
        if scheme is None:
            scheme = str(getattr(values, "scheme", "ring"))
        scheme = scheme.lower()
        scheme = "nested" if "nest" in scheme else "ring"

        if hasattr(values, "_data"):
            values = values._data
        prob = np.asarray(values, dtype=float).copy()
        prob = np.clip(prob, 0.0, None)
        total = np.sum(prob)
        if (not np.isfinite(total)) or total <= 0:
            raise ValueError("Probability map is empty or invalid.")
        prob /= total

        order = np.argsort(prob)[::-1]
        n_pix = int(np.searchsorted(np.cumsum(prob[order]), conf_level) + 1)
        n_pix = min(max(n_pix, 1), prob.size)

        region = np.zeros(prob.size)
        region[order[:n_pix]] = prob[order[:n_pix]]

        nside = hp.get_nside(prob)
        hmap = HealpixMap(
            nside=nside, scheme=scheme, coordsys=coordsys, density=False
        )
        hmap[:] = region
        img, ax = hmap.plot()
        ax.grid(color="white", alpha=0.6)

        true_coord = SkyCoord(l=true_l * u.deg, b=true_b * u.deg, frame="galactic")
        ax.scatter(
            true_coord.l.to(u.deg).value,
            true_coord.b.to(u.deg).value,
            color="red",
            transform=ax.get_transform("world"),
            s=4,
            label="True source",
        )

        best_loc = SkyCoord(l=loc_l * u.deg, b=loc_b * u.deg, frame="galactic")
        ax.scatter(
            best_loc.l.to(u.deg).value,
            best_loc.b.to(u.deg).value,
            color="blue",
            transform=ax.get_transform("world"),
            s=4,
            label="Best localization",
        )

        ax.legend(loc="upper right", frameon=True)
        area = n_pix * hp.nside2pixarea(nside, degrees=True)
        if title is None:
            title = f"{100 * conf_level:.0f}% region  area={area:.2f} deg^2"
        ax.set_title(title)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
            return None
        return img, ax


    def prepare_skymaps(self, results):
        """
        Prepare probability maps once so harmonic transforms can be reused.

        ``map2alm`` is expensive. This method normalizes each probability
        ``sky_map``, converts it to RING ordering, finds the HEALPix pixel
        of the true source, and stores the spherical-harmonic coefficients.
        Call this once, then reuse the output for every systematic sigma.

        Parameters
        ----------
        results : list of dict
            Each dict must contain:
            - ``sky_map``: probability HEALPix map (uses ``_data``)
            - ``source_galactic_l``, ``source_galactic_b``: true Galactic
              coordinates in degrees (not the reconstructed location)

        Returns
        -------
        list of dict
            Prepared items with ``prob``, ``alm``, ``ipix_true``,
            ``nside`` and ``lmax``. Failed maps are skipped.
        """
        prepared = []
        invalid = []
        for i, result in enumerate(results):
            sky_map = result["sky_map"]
            prob = np.asarray(sky_map._data, dtype=float).copy()

            # Skip failed / NaN maps
            if not np.all(np.isfinite(prob)):
                invalid.append(i)
                continue
            total = np.sum(prob)
            if (not np.isfinite(total)) or total <= 0:
                invalid.append(i)
                continue

            prob /= total

            # Harmonic transforms require RING ordering
            if "nest" in str(getattr(sky_map, "scheme", "ring")).lower():
                prob = hp.reorder(prob, n2r=True)

            nside = hp.get_nside(prob)

            # HEALPix: theta = colatitude, phi = longitude
            l_true = result["source_galactic_l"]
            b_true = result["source_galactic_b"]
            ipix_true = hp.ang2pix(
                nside,
                np.deg2rad(90.0 - b_true),
                np.deg2rad(l_true),
                nest=False,
            )

            lmax = 3 * nside - 1
            prepared.append({
                "prob": prob,
                "alm": hp.map2alm(prob, lmax=lmax, iter=0),
                "ipix_true": ipix_true,
                "nside": nside,
                "lmax": lmax,
            })

        print("Total / valid / excluded GRBs:", len(results), len(prepared), len(invalid))
        if invalid:
            print("Excluded indices:", invalid)
        return prepared

    @staticmethod
    def _gaussian_smoothed_map(item, sigma_deg):
        """Unnormalized map convolved with a Gaussian beam (``sigma_deg = 0`` returns ``prob``)."""
        if sigma_deg == 0:
            return item["prob"]
        sigma_rad = np.deg2rad(sigma_deg)
        ell = np.arange(item["lmax"] + 1)
        beam = np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)
        return hp.alm2map(
            hp.almxfl(item["alm"], beam),
            nside=item["nside"],
            lmax=item["lmax"],
        )

    @staticmethod
    def _normalize_smoothed(prob):
        """Clip the ~-1e-15 values of the inverse transform and renormalize. None if invalid."""
        prob = np.clip(prob, 0.0, None)
        total = prob.sum()
        if total <= 0 or not np.isfinite(total):
            return None
        return prob / total

    @staticmethod
    def _true_pixel_cl(prob, ipix_true):
        """Confidence level of the true position: sum of the pixels with prob >= prob[ipix_true]."""
        return np.sum(prob[prob >= prob[ipix_true]])

    @staticmethod
    def smooth_prepared_map(item, sigma_deg, sigma_tail_deg=0.0, tail_frac=0.0):
        """
        Apply a Gaussian (or core+tail) systematic kernel to a prepared probability map.

        ``sigma_deg`` is the Gaussian standard deviation in degrees,
        not the FWHM. ``sigma_deg = 0`` returns the unsmoothed map.

        The beam in harmonic space is
        ``B_l = (1 - f) exp[-0.5 l(l+1) sigma^2] + f exp[-0.5 l(l+1) sigma_tail^2]``
        with ``f = tail_frac``. ``tail_frac = 0`` is the single Gaussian.
        Tiny negative values from the inverse transform are clipped,
        then the map is renormalized.

        Parameters
        ----------
        item : dict
            Output of ``prepare_skymaps`` (must include ``prob``,
            ``alm``, ``nside``, ``lmax``).
        sigma_deg : float
            Gaussian sigma of the core in degrees.
        sigma_tail_deg : float
            Gaussian sigma of the tail in degrees. Used only if ``tail_frac > 0``.
        tail_frac : float
            Fraction of the kernel in the tail, between 0 and 1.

        Returns
        -------
        ndarray or None
            Smoothed probability map, or None if the result is invalid.
        """
        if not 0.0 <= tail_frac <= 1.0:
            raise ValueError(f"tail_frac must be in [0, 1], got {tail_frac}")
        if sigma_deg == 0 and tail_frac == 0:
            return item["prob"].copy()

        prob = (1.0 - tail_frac) * ACSLocalizerBCT._gaussian_smoothed_map(item, sigma_deg)
        if tail_frac > 0:
            prob = prob + tail_frac * ACSLocalizerBCT._gaussian_smoothed_map(item, sigma_tail_deg)
        return ACSLocalizerBCT._normalize_smoothed(prob)

    def pp_curve_smoothed(self, prepared, values, sigma_deg, sigma_tail_deg=0.0, tail_frac=0.0):
        """
        Build the P-P curve at a given systematic kernel (Gaussian or core+tail).

        For each GRB the confidence level of the true position is the
        sum of all pixels with probability >= that of the true pixel.
        The P-P fraction at containment C is the fraction of GRBs
        with that confidence level <= C.

        Parameters
        ----------
        prepared : list of dict
            Output of ``prepare_skymaps``.
        values : array
            Confidence levels at which to evaluate the curve (e.g. 0 to 1
            in steps of 0.01).
        sigma_deg : float
            Systematic Gaussian sigma of the core in degrees (0 = no smoothing).
        sigma_tail_deg, tail_frac : float
            Tail of the kernel, see ``smooth_prepared_map``.

        Returns
        -------
        confidence_levels : ndarray
            True-position confidence level of each GRB.
        fraction : ndarray
            Contained fraction at each value in ``values``.
        """
        cls = []
        for item in prepared:
            prob = self.smooth_prepared_map(item, sigma_deg, sigma_tail_deg, tail_frac)
            if prob is None:
                continue
            cls.append(self._true_pixel_cl(prob, item["ipix_true"]))
        cls = np.asarray(cls)
        fraction = np.searchsorted(np.sort(cls), values, side="right") / len(cls)
        return cls, fraction

    @staticmethod
    def pp_binomial_bands(values, N):
        """
        1, 2 and 3 sigma binomial bands around the P-P diagonal.

        For a perfectly calibrated sample of size N, the contained
        fraction at each confidence level is a binomial draw. The bands
        are the central interval of that distribution, with
        ``alpha = 2 * (1 - Phi(n_sigma))``.

        Parameters
        ----------
        values : array
            Confidence levels (nominal coverage).
        N : int
            Number of events used in the P-P curve.

        Returns
        -------
        dict
            ``bands[nsigma] = (lower, upper)`` for nsigma in {1, 2, 3}.
        """
        bands = {}
        for nsigma in (1, 2, 3):
            alpha = 2.0 * norm.sf(nsigma)
            bands[nsigma] = (
                binom.ppf(alpha / 2.0, N, values) / N,
                binom.ppf(1.0 - alpha / 2.0, N, values) / N,
            )
        return bands

    def plot_pp_coverage(self, results, values=None, show=True):
        """
        Plot the P-P frequentist coverage curve with no systematic smoothing.

        For each confidence level C from 0 to 1 in 1% steps, plot the
        fraction of GRBs whose true position falls inside the C region.
        1, 2 and 3 sigma binomial bands around the diagonal are shown.

        Parameters
        ----------
        results : list of dict
            Same format as ``prepare_skymaps`` (``sky_map`` plus true
            Galactic ``source_galactic_l`` / ``source_galactic_b``).
        values : array, optional
            Confidence levels. Default is 0, 0.01, ..., 1.00.
        show : bool
            If True, display the figure.

        Returns
        -------
        dict
            Output of ``calibrate_pp_coverage`` with ``sigma_sys_deg=[0]``.
            ``pp_plot`` is the pickleable copy of the figure: confidence
            levels, the plotted fraction, and the binomial bands.
        """
        return self.calibrate_pp_coverage(
            results, sigma_sys_deg=[0.0], values=values, show=show
        )

    def calibrate_pp_coverage(
        self, results, sigma_sys_deg=(0.0, 1.0, 2.0, 3.0), values=None, show=True
    ):
        """
        Overlay P-P curves after Gaussian smoothing at several systematic sigmas.

        Harmonic coefficients are computed once in ``prepare_skymaps``.
        Each ``sigma_sys_deg`` is the Gaussian standard deviation in
        degrees (not FWHM). Pick the sigma whose curve lies on the
        diagonal (perfect calibration).

        Parameters
        ----------
        results : list of dict
            Same format as ``prepare_skymaps``.
        sigma_sys_deg : sequence of float
            Systematic Gaussian sigmas to test, in degrees.
        values : array, optional
            Confidence levels. Default is 0, 0.01, ..., 1.00.
        show : bool
            If True, display the figure.

        Returns
        -------
        dict
            ``values``, ``per_sigma`` (confidence levels, P-P fraction,
            90% coverage for each sigma), ``bands``, ``N``, and
            ``pp_plot``. ``pp_plot`` is enough to redraw the figure and
            can be written with ``pickle.dump``.
        """
        if values is None:
            values = np.linspace(0.0, 1.0, 101)
        prepared = self.prepare_skymaps(results)

        per_sigma = {}
        for sigma_deg in sigma_sys_deg:
            cl, fraction = self.pp_curve_smoothed(prepared, values, float(sigma_deg))
            cov90 = float(np.mean(cl <= 0.90))
            print(f"Coverage 90%, sigma={sigma_deg:g} deg: {cov90:.3f}  (N={len(cl)})")
            per_sigma[float(sigma_deg)] = {
                "confidence_levels": cl,
                "fraction": fraction,
                "coverage_90": cov90,
            }

        N = len(next(iter(per_sigma.values()))["confidence_levels"])
        bands = self.pp_binomial_bands(values, N)
        pp_plot = self._pp_coverage_record(values, per_sigma, bands, N)
        self.plot_pp_coverage_from_data(pp_plot, show=show)
        return {
            "values": values,
            "per_sigma": per_sigma,
            "bands": bands,
            "N": N,
            "pp_plot": pp_plot,
        }

    @staticmethod
    def _pp_coverage_record(values, per_sigma, bands, n):
        """Arrays of one ``calibrate_pp_coverage`` figure, in the form that is drawn."""
        curves = []
        for sigma_deg, item in per_sigma.items():
            sigma_deg = float(sigma_deg)
            curves.append({
                "sigma_sys_deg": sigma_deg,
                "label": rf"$\sigma_{{\rm sys}}={sigma_deg:g}^\circ$",
                "confidence_levels": np.asarray(item["confidence_levels"], dtype=float),
                "fraction": np.asarray(item["fraction"], dtype=float),
                "coverage_90": float(item["coverage_90"]),
            })
        return {
            "values": np.asarray(values, dtype=float),
            "n": int(n),
            "bands": [
                {
                    "nsigma": int(nsigma),
                    "low": np.asarray(edge[0], dtype=float),
                    "high": np.asarray(edge[1], dtype=float),
                }
                for nsigma, edge in bands.items()
            ],
            "curves": curves,
        }

    def plot_pp_coverage_from_data(self, pp_plot, show=True):
        """
        Redraw a ``plot_pp_coverage`` / ``calibrate_pp_coverage`` figure.

        ``pp_plot`` is the dict returned by those methods and the object
        to pass to ``pickle.dump``.
        """
        values = np.asarray(pp_plot["values"], dtype=float)
        bands = {
            int(item["nsigma"]): (
                np.asarray(item["low"], dtype=float),
                np.asarray(item["high"], dtype=float),
            )
            for item in pp_plot["bands"]
        }
        self._pp_figure(values, bands)
        for curve in pp_plot["curves"]:
            plt.plot(values, np.asarray(curve["fraction"], dtype=float), linewidth=2, label=curve["label"])
        self._pp_finish(values, show)
        return plt.gcf()

    def calibrate_pp_coverage_core_tail(
        self,
        results,
        sigma_core_deg=(0.0, 1.0, 2.0, 3.0),
        sigma_tail_deg=(5.0, 10.0, 15.0, 20.0, 30.0),
        tail_frac=(0.05, 0.1, 0.2, 0.3, 0.5),
        values=None,
        n_best=3,
        show=True,
    ):
        """
        Grid search of a core+tail systematic kernel on the P-P curve.

        The kernel is ``(1 - f) G(sigma_core) + f G(sigma_tail)``, with
        ``G`` a Gaussian of standard deviation sigma (degrees, not FWHM).
        Use it when a single sigma cannot put the P-P curve on the
        diagonal, e.g. a curve above the diagonal at low confidence and
        below it at high confidence: the core is already wide enough and
        what is missing is the tail (far outliers).

        Every combination with ``sigma_tail > sigma_core`` is tested. Each
        combination is scored with the maximum distance of the P-P curve
        from the diagonal (``ks``). The unsmoothed (original) curve is
        computed and drawn for reference. The Gaussian smoothed maps are
        computed once per GRB and sigma, then mixed for every ``f``.

        Parameters
        ----------
        results : list of dict
            Same format as ``prepare_skymaps``.
        sigma_core_deg, sigma_tail_deg : sequence of float
            Gaussian sigmas of the core and of the tail to test, in degrees.
        tail_frac : sequence of float
            Fractions of the kernel in the tail to test, in (0, 1].
        values : array, optional
            Confidence levels. Default is 0, 0.01, ..., 1.00.
        n_best : int
            Number of best core+tail kernels drawn in the figure, together
            with the original curve.
        show : bool
            If True, display the figure.

        Returns
        -------
        dict
            ``grid``: one dict per kernel (``sigma_core_deg``,
            ``sigma_tail_deg``, ``tail_frac``, ``ks``, ``coverage_50``,
            ``coverage_90``, ``N``, ``confidence_levels``, ``fraction``),
            sorted by ``ks``; ``best`` (best kernel), ``original`` (same
            keys, unsmoothed maps), ``values``, ``bands``, ``N``.
        """
        if values is None:
            values = np.linspace(0.0, 1.0, 101)
        prepared = self.prepare_skymaps(results)

        # first kernel: no smoothing, the original curve
        kernels = [(0.0, 0.0, 0.0)]
        kernels += [
            (float(c), float(t), float(f))
            for c in sigma_core_deg
            for t in sigma_tail_deg
            for f in tail_frac
            if t > c and 0.0 < f <= 1.0
        ]
        if len(kernels) == 1:
            raise ValueError("No core+tail kernel to test: need sigma_tail > sigma_core and 0 < tail_frac <= 1")
        sigmas = sorted({c for c, _, _ in kernels} | {t for _, t, f in kernels if f > 0})

        cls = np.full((len(kernels), len(prepared)), np.nan)
        for j, item in enumerate(prepared):
            maps = {s: self._gaussian_smoothed_map(item, s) for s in sigmas}
            for k, (c, t, f) in enumerate(kernels):
                prob = maps[c] if f == 0 else (1.0 - f) * maps[c] + f * maps[t]
                prob = self._normalize_smoothed(prob)
                if prob is not None:
                    cls[k, j] = self._true_pixel_cl(prob, item["ipix_true"])

        grid = []
        for (c, t, f), cl in zip(kernels, cls):
            cl = cl[np.isfinite(cl)]
            fraction = np.searchsorted(np.sort(cl), values, side="right") / len(cl)
            grid.append({
                "sigma_core_deg": c,
                "sigma_tail_deg": t,
                "tail_frac": f,
                "ks": float(np.max(np.abs(fraction - values))),
                "coverage_50": float(np.mean(cl <= 0.50)),
                "coverage_90": float(np.mean(cl <= 0.90)),
                "N": len(cl),
                "confidence_levels": cl,
                "fraction": fraction,
            })
        original, grid = grid[0], grid[1:]
        grid.sort(key=lambda g: g["ks"])
        best = grid[0]

        def describe(g):
            if g is original:
                return "original (no smoothing)"
            return (f"core {g['sigma_core_deg']:g} deg + tail {g['sigma_tail_deg']:g} deg "
                    f"({100 * g['tail_frac']:g}%)")

        print(f"{'kernel':<40}{'ks':>8}{'cov50':>8}{'cov90':>8}")
        for g in [original] + grid[:max(n_best, 10)]:
            print(f"{describe(g):<40}{g['ks']:>8.3f}{g['coverage_50']:>8.3f}{g['coverage_90']:>8.3f}")

        N = best["N"]
        bands = self.pp_binomial_bands(values, N)

        self._pp_figure(values, bands)
        plt.plot(values, original["fraction"], linewidth=2, color="0.3", ls=":",
                 label=f"{describe(original)} (ks={original['ks']:.3f})")
        for g in grid[:n_best]:
            plt.plot(values, g["fraction"], linewidth=2,
                     label=f"{describe(g)} (ks={g['ks']:.3f})")
        self._pp_finish(values, show)

        return {
            "grid": grid,
            "best": best,
            "original": original,
            "values": values,
            "bands": bands,
            "N": N,
        }

    @staticmethod
    def _pp_figure(values, bands):
        plt.figure(figsize=(8, 7))
        plt.fill_between(values, bands[3][0], bands[3][1], alpha=0.15, label=r"$3\sigma$ expected")
        plt.fill_between(values, bands[2][0], bands[2][1], alpha=0.20, label=r"$2\sigma$ expected")
        plt.fill_between(values, bands[1][0], bands[1][1], alpha=0.30, label=r"$1\sigma$ expected")

    @staticmethod
    def _pp_finish(values, show):
        plt.plot(values, values, "--", color="black", linewidth=1.5, label="Perfect calibration")
        plt.xlabel("Confidence Level", fontsize=15)
        plt.ylabel("Fraction < Confidence Level", fontsize=15)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(fontsize=10)
        plt.tick_params(axis='both', labelsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()

    def smooth_skymap_area(self, sky_map, sigma_deg, conf_level=0.9, sigma_tail_deg=0.0, tail_frac=0.0):
        """
        Smooth one probability sky map with a calibrated systematic kernel.

        Runtime helper: after choosing the kernel from
        ``calibrate_pp_coverage`` (single Gaussian) or
        ``calibrate_pp_coverage_core_tail`` (core+tail), apply it to a
        single map and return the new containment area at ``conf_level``.

        Parameters
        ----------
        sky_map : map object
            Probability HEALPix map (uses ``_data``).
        sigma_deg : float
            Gaussian systematic sigma (of the core) in degrees. 0 skips smoothing.
        conf_level : float
            Containment fraction, default 0.9.
        sigma_tail_deg, tail_frac : float
            Tail of the kernel, see ``smooth_prepared_map``. ``tail_frac = 0``
            is the single Gaussian.

        Returns
        -------
        dict
            Smoothed probability map, containment area in deg^2, and
            equivalent-disk radius in degrees.
        """
        prob = np.asarray(sky_map._data, dtype=float).copy()
        prob /= np.sum(prob)
        if "nest" in str(getattr(sky_map, "scheme", "ring")).lower():
            prob = hp.reorder(prob, n2r=True)
        nside = hp.get_nside(prob)
        item = {"prob": prob, "nside": nside, "lmax": 3 * nside - 1}
        if sigma_deg != 0 or tail_frac != 0:
            item["alm"] = hp.map2alm(prob, lmax=item["lmax"], iter=0)
        prob = self.smooth_prepared_map(item, sigma_deg, sigma_tail_deg, tail_frac)

        # Smallest set of pixels whose probability sums to conf_level
        order = np.argsort(prob)[::-1]
        n_pix = int(np.searchsorted(np.cumsum(prob[order]), conf_level) + 1)
        area = n_pix * hp.nside2pixarea(nside, degrees=True)
        return {
            "prob": prob,
            "cont_area_deg2": float(area),
            "eq_radius_deg": float(np.sqrt(area / np.pi)),
        }

    # ------------------------------------------------------------------
    # Core+tail systematic: maximum-likelihood fit
    # ------------------------------------------------------------------

    COVERAGE_LEVELS = (0.50, 0.68, 0.90, 0.95, 0.99)

    @staticmethod
    def _gaussian_beam(ell, sigma_deg):
        """Harmonic Gaussian beam ``exp[-0.5 l(l+1) sigma^2]``, the one of ``healpy.smoothing(sigma=...)``."""
        sigma_rad = np.deg2rad(sigma_deg)
        return np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)

    @staticmethod
    def true_position_projection(item):
        """
        Legendre projection of a prepared map on its true position.

        For any isotropic beam ``B_l``, the smoothed map at the true pixel,
        as a density per steradian, is ``sum_l B_l c_l`` with

            c_l = (2l + 1) / (4 pi) * sum_i p_i P_l(cos gamma_i)

        where ``p_i`` is the probability of pixel i and ``gamma_i`` its
        angle from the true pixel centre. By the addition theorem this is
        exactly ``healpy.smoothing(prob, sigma, iter=0, lmax=3 nside - 1)``
        evaluated at the true pixel, divided by the pixel area, so the
        likelihood never needs a full-sky transform. Since ``B_0 = 1`` the
        smoothed density integrates exactly to 1 for every sigma (its
        HEALPix pixel sum differs from 1 only by the pixel quadrature).

        Parameters
        ----------
        item : dict
            Output of ``prepare_skymaps``.

        Returns
        -------
        ndarray
            ``c_l`` for ``l = 0 .. lmax``, in sr^-1. ``c_0 = 1 / (4 pi)``.
        """
        nside, lmax, prob = item["nside"], item["lmax"], item["prob"]
        pix = np.flatnonzero(prob > 0)
        w = prob[pix]
        x = np.array(hp.pix2vec(nside, pix)).T @ np.array(hp.pix2vec(nside, item["ipix_true"]))
        x = np.clip(x, -1.0, 1.0)

        c = np.empty(lmax + 1)
        p_prev, p_cur = np.ones_like(x), x
        c[0] = w.sum()
        if lmax >= 1:
            c[1] = w @ x
        for ell in range(1, lmax):
            p_prev, p_cur = p_cur, ((2 * ell + 1) * x * p_cur - ell * p_prev) / (ell + 1)
            c[ell + 1] = w @ p_cur
        ell = np.arange(lmax + 1)
        return c * (2 * ell + 1) / (4.0 * np.pi)

    def _projection_matrix(self, prepared):
        """Stack the projections of all maps, zero-padded to the largest lmax."""
        projections = [self.true_position_projection(item) for item in prepared]
        n_ell = max(len(c) for c in projections)
        C = np.zeros((len(projections), n_ell))
        for j, c in enumerate(projections):
            C[j, :len(c)] = c
        return C, np.arange(n_ell)

    @classmethod
    def _mixture_density(cls, thetas, C, ell):
        """Core+tail density per steradian at each true position. Shape ``(n_params, n_grb)``."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        ell = np.asarray(ell, dtype=float)
        sigma = np.deg2rad(thetas[:, :2])[:, :, None]
        beam = np.exp(-0.5 * (ell * (ell + 1.0)) * sigma**2)
        weight = np.stack((1.0 - thetas[:, 2], thetas[:, 2]), axis=1)[:, :, None]
        return np.sum(weight * beam, axis=1) @ np.asarray(C, dtype=float).T

    @classmethod
    def _core_tail_log_likelihood_many(cls, thetas, C, ell, density_floor=1e-12):
        """Log-likelihood at many ``(sigma_core, sigma_tail, f)`` points. Same floor as the scalar call."""
        logl = np.sum(np.log(np.maximum(cls._mixture_density(thetas, C, ell), density_floor)), axis=1)
        logl[~np.isfinite(logl)] = -np.inf
        return logl

    @classmethod
    def core_tail_log_likelihood(cls, params, C, ell, density_floor=1e-12):
        """``sum log p_cal(true)`` for ``(sigma_core, sigma_tail, f)`` in degrees. ``f = 0`` is one Gaussian."""
        params = np.asarray(params, dtype=float).reshape(1, 3)
        return float(cls._core_tail_log_likelihood_many(params, C, ell, density_floor)[0])

    @classmethod
    def gaussian_log_likelihood(cls, sigma, C, ell, density_floor=1e-12):
        """Single-Gaussian systematic: the core+tail likelihood with ``f = 0``."""
        sigma = float(sigma)
        return cls.core_tail_log_likelihood((sigma, sigma, 0.0), C, ell, density_floor)

    @staticmethod
    def prior_transform_gaussian(u, sigma_bounds):
        """Map one unit-cube coordinate to ``sigma ~ Uniform(low, high)`` degrees."""
        lo, hi = sigma_bounds
        return np.array([lo + float(u[0]) * (hi - lo)])

    @staticmethod
    def prior_transform_core_tail(u, sigma_core_bounds, sigma_tail_bounds, f_bounds, min_separation_deg):
        """
        Map the unit cube to the core+tail prior.

        ``sigma_core`` and ``f`` are uniform on their bounds.
        ``sigma_tail | sigma_core`` is uniform from ``max(tail_low, sigma_core + eps)`` to ``tail_high``.
        That is a normalized conditional prior, not a uniform density on ``sigma_tail > sigma_core``.
        """
        c_lo, c_hi = sigma_core_bounds
        t_lo, t_hi = sigma_tail_bounds
        f_lo, f_hi = f_bounds
        sigma_c = c_lo + float(u[0]) * (c_hi - c_lo)
        low = max(t_lo, sigma_c + min_separation_deg)
        if low >= t_hi:
            raise ValueError("sigma_tail prior has zero width; raise its upper bound or lower sigma_core_bounds[1]")
        sigma_t = low + float(u[1]) * (t_hi - low)
        frac = f_lo + float(u[2]) * (f_hi - f_lo)
        return np.array([sigma_c, sigma_t, frac])

    def _fit_core_tail(self, C, ell, bounds, density_floor, de_kwargs, min_separation_deg):
        """Maximum likelihood of ``(sigma_core, sigma_tail, f)``, with ``sigma_tail > sigma_core``."""
        separation = NonlinearConstraint(lambda x: x[1] - x[0], min_separation_deg, np.inf)
        return differential_evolution(
            lambda x: -self.core_tail_log_likelihood(x, C, ell, density_floor),
            bounds=bounds, constraints=(separation,), **de_kwargs,
        )

    def _fit_gaussian(self, C, ell, sigma_bounds, density_floor, de_kwargs):
        """Maximum likelihood of the single systematic sigma, on the same projections."""
        return differential_evolution(
            lambda x: -self.gaussian_log_likelihood(float(x[0]), C, ell, density_floor),
            bounds=[tuple(sigma_bounds)], **de_kwargs,
        )

    def _bootstrap_core_tail(self, C, ell, bounds, density_floor, de_kwargs, min_separation_deg, n_bootstrap, rng):
        samples = np.empty((n_bootstrap, 3))
        for i in range(n_bootstrap):
            idx = rng.integers(0, len(C), len(C))
            kw = {**de_kwargs, "seed": int(rng.integers(2**32))}
            samples[i] = self._fit_core_tail(C[idx], ell, bounds, density_floor, kw, min_separation_deg).x
        low, high = np.percentile(samples, [16, 84], axis=0)
        names = ("sigma_core", "sigma_tail", "f")
        return {
            "n": n_bootstrap,
            "std": dict(zip(names, samples.std(axis=0, ddof=1).tolist())),
            "p16": dict(zip(names, low.tolist())),
            "p84": dict(zip(names, high.tolist())),
            "samples": samples,
        }

    def _split_calibration_sample(self, results, validation_results, validation_fraction, rng):
        """Training sample for the fit and the evidence, held-out sample for the P-P plot."""
        prepared = self.prepare_skymaps(results)
        if validation_results is not None:
            fit_sample, val_sample = prepared, self.prepare_skymaps(validation_results)
        elif validation_fraction > 0:
            order = rng.permutation(len(prepared))
            n_val = int(round(validation_fraction * len(prepared)))
            val_sample = [prepared[i] for i in order[:n_val]]
            fit_sample = [prepared[i] for i in order[n_val:]]
        else:
            fit_sample = val_sample = prepared
        if len(fit_sample) < 3 or len(val_sample) < 3:
            raise ValueError(f"Too few GRBs: {len(fit_sample)} to fit, {len(val_sample)} to validate")
        return fit_sample, val_sample

    def _log_predictive(self, thetas, logwt, C, ell, density_floor):
        """ln p(validation | training): one systematic kernel shared by every GRB."""
        logl = np.empty(len(thetas))
        for start in range(0, len(thetas), 1024):
            chunk = thetas[start:start + 1024]
            logl[start:start + len(chunk)] = self._core_tail_log_likelihood_many(chunk, C, ell, density_floor)
        return float(logsumexp(logwt + logl) - logsumexp(logwt))

    def _nested_evidence(
        self, label, loglike, prior_transform, ndim,
        C_val, ell_val, density_floor, nlive, dlogz, seed, n_repeat, print_nested,
    ):
        """dynesty evidence. The quoted value is the first run; extra runs only check that ln Z repeats."""
        try:
            import dynesty
        except ImportError as exc:
            raise ImportError("Bayesian evidence needs dynesty (`pip install dynesty`).") from exc
        if dlogz is None:
            dlogz = 1e-3 * (nlive - 1.0) + 0.01
        runs = []
        predictive = None
        for repeat in range(int(n_repeat)):
            run_seed = None if seed is None else int(seed) + 10007 * (repeat + 1) + ndim
            print(f"Nested sampling {label}: nlive={nlive}, dlogz={dlogz:.4g}, run {repeat + 1}/{int(n_repeat)}")
            sampler = dynesty.NestedSampler(
                loglike, prior_transform, ndim, nlive=int(nlive), rstate=np.random.default_rng(run_seed),
            )
            sampler.run_nested(dlogz=float(dlogz), print_progress=bool(print_nested))
            res = sampler.results
            runs.append({
                "lnZ": float(res.logz[-1]),
                "lnZ_err": float(res.logzerr[-1]),
                "niter": int(res.niter),
                "ncall": int(np.sum(res.ncall)),
                "logl_max": float(np.max(res.logl)),
            })
            print(f"    ln Z = {runs[-1]['lnZ']:.2f} ± {runs[-1]['lnZ_err']:.3f}")
            if repeat == 0:
                samples = np.asarray(res.samples, dtype=float)
                if ndim == 1:
                    sigma = samples.reshape(-1)
                    samples = np.column_stack((sigma, sigma, np.zeros(len(sigma))))
                predictive = self._log_predictive(samples, res.logwt, C_val, ell_val, density_floor)
        out = dict(runs[0])
        out["log10_Z"] = out["lnZ"] / np.log(10.0)
        out["ln_predictive"] = predictive
        out["lnZ_repeat_std"] = None if len(runs) == 1 else float(np.std([run["lnZ"] for run in runs], ddof=1))
        return out

    def _validation_confidence(self, sample, sigma, sigma_core, sigma_tail, frac):
        """True-position confidence level of the original, Gaussian and core+tail maps."""
        levels = []
        for item in sample:
            prob_g = self.smooth_prepared_map(item, sigma, 0.0, 0.0)
            prob_ct = self.smooth_prepared_map(item, sigma_core, sigma_tail, frac)
            if prob_g is None or prob_ct is None:
                continue
            ipix = item["ipix_true"]
            levels.append((
                self._true_pixel_cl(item["prob"], ipix),
                self._true_pixel_cl(prob_g, ipix),
                self._true_pixel_cl(prob_ct, ipix),
            ))
        if len(levels) < 3:
            raise ValueError(f"Too few validation maps could be smoothed ({len(levels)})")
        return np.asarray(levels).T

    def _coverage_at_levels(self, confidence):
        return {f"{level:.2f}": float(np.mean(confidence <= level)) for level in self.COVERAGE_LEVELS}

    def calibrate_core_tail_likelihood(
        self,
        results,
        validation_results=None,
        validation_fraction=0.3,
        sigma_bounds=(0.5, 60.0),
        sigma_core_bounds=(0.5, 20.0),
        sigma_tail_bounds=(1.0, 60.0),
        f_bounds=(0.0, 1.0),
        min_separation_deg=1e-3,
        n_bootstrap=100,
        density_floor=1e-12,
        seed=0,
        de_kwargs=None,
        compute_evidence=True,
        nlive=500,
        dlogz=None,
        n_repeat=1,
        print_nested=False,
        output_dir=None,
        prefix="core_tail_likelihood",
        show=True,
    ):
        """
        Compare a single Gaussian and a core+tail systematic on the same GRBs.

        Both use ``core_tail_log_likelihood`` on the HEALPix map at the true
        position. The Gaussian is that likelihood with ``f = 0``. The evidence
        comes from nested sampling (dynesty) and

            log10 B_21 = (ln Z_core+tail - ln Z_gaussian) / ln(10).

        Priors are uniform on the bounds below. The tail prior is conditional
        on ``sigma_core`` (see ``prior_transform_core_tail``), not uniform on
        the region ``sigma_tail > sigma_core``.

        The fit and the evidence use only the training sample. The held-out
        fraction is used afterwards for the P-P plot, the KS test and the
        predictive likelihood. ``log10 B_21 > 0`` favors core+tail given these
        priors. It is not a factor of localization precision, and ``2 Δln L``
        is not a chi-square test (at ``f = 0`` the tail width is unidentified).

        ``result["pp_plot"]`` redraws the figure with ``plot_pp_from_data``
        and is the object to pickle. The same dict is written to
        ``<prefix>_pp.pkl`` when an output directory is set.

        Parameters
        ----------
        results : list of dict
            Same format as ``prepare_skymaps``.
        validation_results : list of dict, optional
            Independent validation sample. If given, ``validation_fraction`` is ignored.
        validation_fraction : float
            Fraction held out when ``validation_results`` is None. ``0`` validates on the fit sample.
        sigma_bounds, sigma_core_bounds, sigma_tail_bounds, f_bounds : (float, float)
            Uniform prior and search range. Sigmas in degrees.
            Need ``sigma_tail_bounds[1] > sigma_core_bounds[1] + min_separation_deg``.
        min_separation_deg : float
            Minimum ``sigma_tail - sigma_core``, and the ``eps`` of the tail prior.
        n_bootstrap : int
            Bootstrap resamples for the core+tail uncertainties. ``0`` skips them.
        density_floor : float
            Smallest density, in sr^-1, entering the logarithm.
        nlive, dlogz, n_repeat : int, float, int
            Nested-sampling settings. ``dlogz=None`` uses dynesty's default.
            ``n_repeat > 1`` repeats the run and reports the scatter of ln Z.
        output_dir : str, optional
            Directory for the json, the table, the pickle and the figure. Default ``self.output_dir``.
        """
        sigma_bounds, core_bounds, tail_bounds, frac_bounds = _check_systematic_bounds(
            sigma_bounds, sigma_core_bounds, sigma_tail_bounds, f_bounds,
            min_separation_deg, nlive, n_repeat,
        )
        bounds = [core_bounds, tail_bounds, frac_bounds]
        rng = np.random.default_rng(seed)
        de_kwargs = {
            "seed": seed, "tol": 1e-8, "popsize": 20, "maxiter": 2000, "polish": True,
            **(de_kwargs or {}),
        }

        fit_sample, val_sample = self._split_calibration_sample(
            results, validation_results, validation_fraction, rng,
        )
        on_training = val_sample is fit_sample
        if on_training:
            print("WARNING: validation uses the training sample, so the KS test is not independent")

        # Maximum likelihood on the training projections. Both models see the same GRBs.
        C, ell = self._projection_matrix(fit_sample)
        core_fit = self._fit_core_tail(C, ell, bounds, density_floor, de_kwargs, min_separation_deg)
        sigma_core, sigma_tail, frac = (float(v) for v in core_fit.x)
        log_like = float(-core_fit.fun)
        gauss_fit = self._fit_gaussian(C, ell, sigma_bounds, density_floor, de_kwargs)
        sigma = float(gauss_fit.x[0])
        log_like_g = float(-gauss_fit.fun)
        bootstrap = None
        if n_bootstrap > 0:
            bootstrap = self._bootstrap_core_tail(
                C, ell, bounds, density_floor, de_kwargs, min_separation_deg, n_bootstrap, rng,
            )

        C_val, ell_val = self._projection_matrix(val_sample)
        lnL_val_g = self.gaussian_log_likelihood(sigma, C_val, ell_val, density_floor)
        lnL_val_ct = self.core_tail_log_likelihood((sigma_core, sigma_tail, frac), C_val, ell_val, density_floor)

        evidence_g = evidence_ct = None
        if compute_evidence:
            evidence_g = self._nested_evidence(
                "single Gaussian",
                lambda theta: self.gaussian_log_likelihood(theta[0], C, ell, density_floor),
                lambda u: self.prior_transform_gaussian(u, sigma_bounds),
                1, C_val, ell_val, density_floor, nlive, dlogz, seed, n_repeat, print_nested,
            )
            evidence_ct = self._nested_evidence(
                "core+tail",
                lambda theta: self.core_tail_log_likelihood(theta, C, ell, density_floor),
                lambda u: self.prior_transform_core_tail(u, core_bounds, tail_bounds, frac_bounds, min_separation_deg),
                3, C_val, ell_val, density_floor, nlive, dlogz, seed, n_repeat, print_nested,
            )

        cl_orig, cl_g, cl_cal = self._validation_confidence(val_sample, sigma, sigma_core, sigma_tail, frac)
        ks, ks_g, ks_orig = kstest(cl_cal, "uniform"), kstest(cl_g, "uniform"), kstest(cl_orig, "uniform")
        coverage = self._coverage_at_levels(cl_cal)
        coverage_g = self._coverage_at_levels(cl_g)
        coverage_orig = self._coverage_at_levels(cl_orig)

        gaussian = _systematic_model(1, {"sigma_deg": sigma}, log_like_g, lnL_val_g, ks_g, evidence_g)
        core = _systematic_model(
            3,
            {"sigma_core_deg": sigma_core, "sigma_tail_deg": sigma_tail, "f_tail": frac},
            log_like, lnL_val_ct, ks, evidence_ct,
        )
        comparison = _model_comparison(
            gaussian, core, len(fit_sample), len(cl_cal), bootstrap, coverage_g, coverage, coverage_orig,
        )
        warnings = _calibration_warnings(
            sigma, sigma_bounds, sigma_core, core_bounds, sigma_tail, tail_bounds, frac, frac_bounds,
            min_separation_deg, C, ell, density_floor, self, evidence_g, evidence_ct,
        )
        if warnings:
            comparison["table_text"] += "\n" + "\n".join(warnings)
        print(comparison["table_text"])

        out = {
            "sigma_best": sigma,
            "sigma_core_best": sigma_core,
            "sigma_tail_best": sigma_tail,
            "f_best": frac,
            "log_likelihood": log_like,
            "log_likelihood_gaussian": log_like_g,
            "optimizer": _optimizer_record(core_fit),
            "optimizer_gaussian": _optimizer_record(gauss_fit),
            "bootstrap": bootstrap,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "ks_gaussian": {"statistic": float(ks_g.statistic), "pvalue": float(ks_g.pvalue)},
            "coverage": coverage,
            "coverage_gaussian": coverage_g,
            "ks_original": {"statistic": float(ks_orig.statistic), "pvalue": float(ks_orig.pvalue)},
            "coverage_original": coverage_orig,
            "confidence_levels": cl_cal,
            "confidence_levels_gaussian": cl_g,
            "confidence_levels_original": cl_orig,
            "n_fit": len(fit_sample),
            "n_validation": len(cl_cal),
            "validation_on_fit_sample": on_training,
            "bounds": {"sigma": sigma_bounds, "sigma_core": core_bounds, "sigma_tail": tail_bounds, "f": frac_bounds},
            "comparison": comparison,
            "log10_bayes_factor_21": comparison["log10_bayes_factor_21"],
            "log10_bayes_factor_21_err": comparison["log10_bayes_factor_21_err"],
            "favored_model": comparison["favored_model"],
            "files": {},
        }
        out["pp_plot"] = self._make_pp_plot(cl_orig, cl_g, cl_cal, out)
        self._save_calibration(out, output_dir, prefix, show)
        return out

    def _make_pp_plot(self, cl_orig, cl_gauss, cl_core, out):
        """Pickleable copy of the validation P-P figure."""
        grid = np.linspace(0.0, 1.0, 201)
        bands = self.pp_binomial_bands(grid, len(cl_core))
        curves = [
            _pp_curve("original", "Original maps", cl_orig, {"color": "0.35", "ls": ":"}, grid,
                      ks_statistic=out["ks_original"]["statistic"], ks_pvalue=out["ks_original"]["pvalue"]),
            _pp_curve("single_gaussian", "Single Gaussian", cl_gauss, {"color": "C1"}, grid,
                      sigma_deg=out["sigma_best"],
                      ks_statistic=out["ks_gaussian"]["statistic"], ks_pvalue=out["ks_gaussian"]["pvalue"]),
            _pp_curve("core_tail", "Core+tail", cl_core, {"color": "C0"}, grid,
                      sigma_core_deg=out["sigma_core_best"], sigma_tail_deg=out["sigma_tail_best"],
                      f_tail=out["f_best"], ks_statistic=out["ks_statistic"], ks_pvalue=out["ks_pvalue"]),
        ]
        return {
            "grid": grid,
            "n": int(len(cl_core)),
            "n_fit": int(out["n_fit"]),
            "validation_on_fit_sample": bool(out["validation_on_fit_sample"]),
            "bands": [
                {"nsigma": int(k), "low": np.asarray(lo, dtype=float), "high": np.asarray(hi, dtype=float)}
                for k, (lo, hi) in bands.items()
            ],
            "curves": curves,
            "annotation": _pp_annotation(out),
        }

    def plot_pp_from_data(self, pp_plot, show=True):
        """Redraw the calibration P-P figure from the dict stored in ``pp_plot`` or in the pickle."""
        grid = np.asarray(pp_plot["grid"], dtype=float)
        bands = {
            int(item["nsigma"]): (np.asarray(item["low"], dtype=float), np.asarray(item["high"], dtype=float))
            for item in pp_plot["bands"]
        }
        self._pp_figure(grid, bands)
        fig, ax = plt.gcf(), plt.gca()
        for curve in pp_plot["curves"]:
            ax.step(curve["step_x"], curve["step_y"], where="post", linewidth=2,
                    label=curve["label"], **curve["style"])
        ax.text(0.04, 0.96, "\n".join(pp_plot["annotation"]), transform=ax.transAxes,
                va="top", fontsize=10, bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"})
        self._pp_finish(grid, show=False)
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def _save_calibration(self, out, output_dir, prefix, show):
        fig = self.plot_pp_from_data(out["pp_plot"], show=False)
        output_dir = output_dir or getattr(self, "output_dir", None)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.join(output_dir, prefix)
            for ext in ("png", "pdf"):
                fig.savefig(f"{base}_pp.{ext}", dpi=300, bbox_inches="tight")
                out["files"][f"figure_{ext}"] = f"{base}_pp.{ext}"
            out["files"]["json"] = f"{base}.json"
            with open(out["files"]["json"], "w") as fp:
                json.dump(_to_json(out), fp, indent=2)
            out["files"]["pp_pickle"] = f"{base}_pp.pkl"
            with open(out["files"]["pp_pickle"], "wb") as fp:
                pickle.dump(out["pp_plot"], fp, protocol=pickle.HIGHEST_PROTOCOL)
            out["files"]["comparison_txt"] = f"{base}_comparison.txt"
            with open(out["files"]["comparison_txt"], "w") as fp:
                fp.write(out["comparison"]["table_text"] + "\n")
        if show:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def _load_pickle(path):
        """Load a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


def _check_systematic_bounds(sigma_bounds, core_bounds, tail_bounds, frac_bounds, eps, nlive, n_repeat):
    sigma_bounds, core_bounds, tail_bounds, frac_bounds = (
        tuple(map(float, bounds)) for bounds in (sigma_bounds, core_bounds, tail_bounds, frac_bounds)
    )
    s_lo, s_hi = sigma_bounds
    c_lo, c_hi = core_bounds
    t_lo, t_hi = tail_bounds
    f_lo, f_hi = frac_bounds
    if not 0.0 < s_lo < s_hi:
        raise ValueError(f"sigma_bounds must satisfy 0 < low < high, got {sigma_bounds}")
    if not 0.0 < c_lo < c_hi:
        raise ValueError(f"sigma_core_bounds must satisfy 0 < low < high, got {core_bounds}")
    if not t_lo < t_hi or t_hi <= c_lo + eps:
        raise ValueError(f"sigma_tail_bounds {tail_bounds} leave no room for sigma_tail > sigma_core")
    if t_hi <= c_hi + eps:
        raise ValueError(
            "sigma_tail_bounds[1] must exceed sigma_core_bounds[1] + min_separation_deg, "
            f"got tail {tail_bounds} and core {core_bounds}"
        )
    if not 0.0 <= f_lo < f_hi <= 1.0:
        raise ValueError(f"f_bounds must satisfy 0 <= low < high <= 1, got {frac_bounds}")
    if int(nlive) < 2 or int(n_repeat) < 1:
        raise ValueError(f"nlive must be >= 2 and n_repeat >= 1, got nlive={nlive}, n_repeat={n_repeat}")
    return sigma_bounds, core_bounds, tail_bounds, frac_bounds


def _optimizer_record(result):
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
    }


def _systematic_model(n_parameters, parameters, lnL_train, lnL_val, ks, evidence):
    """One column of the comparison table. Missing evidence values stay None."""
    evidence = evidence or {}
    model = {
        "n_parameters": n_parameters,
        "sigma_deg": None,
        "sigma_core_deg": None,
        "sigma_tail_deg": None,
        "f_tail": None,
        "lnL_max_training": lnL_train,
        "lnL_validation": lnL_val,
        "lnZ": evidence.get("lnZ"),
        "lnZ_err": evidence.get("lnZ_err"),
        "log10_Z": evidence.get("log10_Z"),
        "ln_predictive": evidence.get("ln_predictive"),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }
    model.update(parameters)
    return model


def _near_edge(value, low, high):
    pad = 0.01 * (high - low)
    return value <= low + pad or value >= high - pad


def _calibration_warnings(
    sigma, sigma_bounds, sigma_core, core_bounds, sigma_tail, tail_bounds, frac, frac_bounds,
    eps, C, ell, density_floor, localizer, evidence_g, evidence_ct,
):
    """Short checks that change how the Bayes factor should be read."""
    messages = []
    n_floor_g = int(np.sum(localizer._mixture_density((sigma, sigma, 0.0), C, ell) <= density_floor * (1 + 1e-8)))
    n_floor_ct = int(np.sum(
        localizer._mixture_density((sigma_core, sigma_tail, frac), C, ell) <= density_floor * (1 + 1e-8)
    ))
    if n_floor_g or n_floor_ct:
        messages.append(
            f"WARNING: density_floor cuts {n_floor_g} Gaussian and {n_floor_ct} core+tail training GRBs"
        )
    tail_low = max(tail_bounds[0], sigma_core + eps)
    close = [
        name for name, value, low, high in (
            ("sigma", sigma, *sigma_bounds),
            ("sigma_core", sigma_core, *core_bounds),
            ("sigma_tail", sigma_tail, tail_low, tail_bounds[1]),
            ("f", frac, *frac_bounds),
        ) if _near_edge(value, low, high)
    ]
    if close:
        messages.append("WARNING: maximum likelihood near a prior edge: " + ", ".join(close))
    if evidence_g is not None and max(evidence_g["lnZ_err"], evidence_ct["lnZ_err"]) > 0.5:
        messages.append("WARNING: evidence uncertainty is large; increase nlive or lower dlogz")
    if evidence_g is not None and evidence_g["lnZ_repeat_std"] is not None:
        messages.append(
            "Run-to-run std of ln Z: "
            f"Gaussian {evidence_g['lnZ_repeat_std']:.3f}, core+tail {evidence_ct['lnZ_repeat_std']:.3f}"
        )
    return messages


_TABLE_ROWS = (
    ("n_parameters", "Number of parameters", "{:.0f}"),
    ("sigma_deg", "sigma [deg]", "{:.3f}"),
    ("sigma_core_deg", "sigma_core [deg]", "{:.3f}"),
    ("sigma_tail_deg", "sigma_tail [deg]", "{:.3f}"),
    ("f_tail", "f_tail", "{:.4f}"),
    ("lnL_max_training", "ln L_max training", "{:.2f}"),
    ("lnZ", "ln Z", "{:.2f}"),
    ("lnZ_err", "ln Z uncertainty", "{:.3f}"),
    ("log10_Z", "log10 Z", "{:.3f}"),
    ("ks_statistic", "KS validation", "{:.3f}"),
    ("ks_pvalue", "KS p-value", "{:.3g}"),
    ("lnL_validation", "ln L validation", "{:.2f}"),
    ("ln_predictive", "ln predictive validation", "{:.2f}"),
)


def _model_comparison(gaussian, core, n_fit, n_validation, bootstrap, coverage_g, coverage, coverage_orig):
    """The scientific table, plus log10 B21 when both evidences exist."""
    rows = [
        {"quantity": key, "label": label, "single_gaussian": gaussian[key], "core_tail": core[key]}
        for key, label, _fmt in _TABLE_ROWS
    ]
    log10_b = log10_b_err = favored = within = None
    if gaussian["lnZ"] is not None and core["lnZ"] is not None:
        log10_b = (core["lnZ"] - gaussian["lnZ"]) / np.log(10.0)
        log10_b_err = np.hypot(gaussian["lnZ_err"], core["lnZ_err"]) / np.log(10.0)
        if np.isfinite(log10_b):
            favored = "core+tail" if log10_b > 0 else "single gaussian" if log10_b < 0 else "no preference"
            within = bool(abs(log10_b) <= log10_b_err)

    def cell(value, fmt):
        return "—" if value is None else fmt.format(value)

    formats = {key: fmt for key, _label, fmt in _TABLE_ROWS}
    body = [
        (row["label"], cell(row["single_gaussian"], formats[row["quantity"]]), cell(row["core_tail"], formats[row["quantity"]]))
        for row in rows
    ]
    headers = ("Quantity", "Single Gaussian", "Core+tail")
    widths = [max(len(headers[i]), *(len(line[i]) for line in body)) for i in range(3)]

    def line(cols):
        return "  ".join(col.ljust(width) for col, width in zip(cols, widths)).rstrip()

    text = [
        f"Model comparison on {n_fit} training GRBs, {n_validation} validation GRBs",
        line(headers),
        line(tuple("-" * width for width in widths)),
        *(line(row) for row in body),
        "",
        f"Delta ln L_max (core+tail - single Gaussian, training) = {core['lnL_max_training'] - gaussian['lnL_max_training']:.2f}",
    ]
    if log10_b is None or not np.isfinite(log10_b):
        text.append("log10 B21 was not computed.")
    else:
        text.append(f"log10 B21 (core+tail / single Gaussian) = {log10_b:.3f} ± {log10_b_err:.3f}")
        text.append(f"Favored model: {favored}")
        if within:
            text.append("The two models are not separated by the numerical uncertainty on log10 B21.")
        else:
            text.append("The preference is larger than the numerical uncertainty on log10 B21.")
        text.append("Equal model priors: the posterior odds equal the Bayes factor.")
        text.append("log10 B21 is relative support for these priors, not a gain in localization precision.")
    text.append("2*Delta ln L is not a chi-square statistic: at f=0 the tail width is unidentified.")
    if bootstrap is not None:
        std = bootstrap["std"]
        text.append(
            f"bootstrap std: sigma_core={std['sigma_core']:.3f} deg, "
            f"sigma_tail={std['sigma_tail']:.3f} deg, f={std['f']:.3f}"
        )
    for label, cov in (("gaussian", coverage_g), ("core+tail", coverage), ("original", coverage_orig)):
        text.append("coverage " + label + ": " + "  ".join(f"{level}={value:.3f}" for level, value in cov.items()))
    return {
        "single_gaussian": gaussian,
        "core_tail": core,
        "rows": rows,
        "table_text": "\n".join(text),
        "log10_bayes_factor_21": None if log10_b is None else float(log10_b),
        "log10_bayes_factor_21_err": None if log10_b_err is None else float(log10_b_err),
        "favored_model": favored,
        "n_fit": int(n_fit),
        "n_validation": int(n_validation),
    }


def _pp_curve(name, label, confidence, style, grid, **extra):
    ordered = np.sort(np.asarray(confidence, dtype=float))
    n = len(ordered)
    curve = {
        "name": name,
        "label": label,
        "confidence_levels": np.asarray(confidence, dtype=float),
        "step_x": np.concatenate(([0.0], ordered, [1.0])),
        "step_y": np.concatenate(([0.0], np.arange(1, n + 1) / n, [1.0])),
        "fraction": np.searchsorted(ordered, grid, side="right") / n,
        "style": style,
    }
    curve.update(extra)
    return curve


def _pp_annotation(out):
    bootstrap = out["bootstrap"]
    err = (lambda key: rf" \pm {bootstrap['std'][key]:.2f}") if bootstrap else (lambda key: "")
    lines = [
        rf"$\sigma = {out['sigma_best']:.2f}^\circ$",
        rf"$\sigma_{{\rm core}} = {out['sigma_core_best']:.2f}{err('sigma_core')}^\circ$",
        rf"$\sigma_{{\rm tail}} = {out['sigma_tail_best']:.2f}{err('sigma_tail')}^\circ$",
        rf"$f = {out['f_best']:.3f}{err('f')}$",
        rf"KS$_G = {out['ks_gaussian']['statistic']:.3f}$, KS$_{{CT}} = {out['ks_statistic']:.3f}$",
        rf"$N = {out['n_validation']}$",
    ]
    bayes = out["comparison"]["log10_bayes_factor_21"]
    if bayes is not None:
        lines.append(rf"$\log_{{10}} B_{{21}} = {bayes:.2f} \pm {out['comparison']['log10_bayes_factor_21_err']:.2f}$")
    return lines


def _to_json(obj):
    """Recursively convert numpy arrays, scalars and tuples to JSON types."""
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

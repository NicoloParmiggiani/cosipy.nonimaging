import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic, ICRS
from scipy.optimize import NonlinearConstraint, differential_evolution
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
            90% coverage for each sigma), ``bands``, and ``N``.
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

        self._pp_figure(values, bands)
        for sigma_deg, out in per_sigma.items():
            plt.plot(
                values, out["fraction"], linewidth=2,
                label=rf"$\sigma_{{\rm sys}}={sigma_deg:g}^\circ$",
            )
        self._pp_finish(values, show)
        return {"values": values, "per_sigma": per_sigma, "bands": bands, "N": N}

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
    def core_tail_log_likelihood(cls, params, C, ell, density_floor=1e-12):
        """
        ``log L = sum_i log p_cal,i(true)`` for ``params = (sigma_core, sigma_tail, f)`` in degrees.

        ``p_cal = (1 - f) P_sigma_core + f P_sigma_tail`` is linear in the
        beam, so the two beams are mixed first and applied with one
        product. Densities are per steradian; values below
        ``density_floor`` (zeros, or ringing of the band-limited transform
        for very small sigma) are raised to it.
        """
        sigma_core, sigma_tail, f = params
        beam = (1.0 - f) * cls._gaussian_beam(ell, sigma_core) + f * cls._gaussian_beam(ell, sigma_tail)
        density = C @ beam
        return float(np.sum(np.log(np.maximum(density, density_floor))))

    def _fit_core_tail(self, C, ell, bounds, density_floor, de_kwargs, min_separation_deg):
        """Maximize the core+tail likelihood with differential evolution, ``sigma_tail > sigma_core``."""
        separation = NonlinearConstraint(lambda x: x[1] - x[0], min_separation_deg, np.inf)
        return differential_evolution(
            lambda x: -self.core_tail_log_likelihood(x, C, ell, density_floor),
            bounds=bounds,
            constraints=(separation,),
            **de_kwargs,
        )

    def calibrate_core_tail_likelihood(
        self,
        results,
        validation_results=None,
        validation_fraction=0.3,
        sigma_core_bounds=(0.5, 20.0),
        sigma_tail_bounds=(1.0, 60.0),
        f_bounds=(0.0, 1.0),
        min_separation_deg=1e-3,
        n_bootstrap=100,
        density_floor=1e-12,
        seed=0,
        de_kwargs=None,
        output_dir=None,
        prefix="core_tail_likelihood",
        show=True,
    ):
        """
        Maximum-likelihood fit of a core+tail systematic kernel, validated with a P-P curve.

        Model, inspired by the Fermi-GBM systematic of Connaughton et al.
        (2015, ApJS 216, 32): every probability map is convolved with

            P_cal = (1 - f) P_sigma_core + f P_sigma_tail

        where ``P_sigma`` is the map smoothed with a Gaussian of standard
        deviation sigma (degrees, not FWHM), with the beam of
        ``healpy.smoothing``. The three parameters maximize

            log L = sum_i log p_cal,i(true position of GRB i)

        with ``p_cal`` the density per steradian, under the constraints
        ``sigma_core > 0``, ``sigma_tail > sigma_core``, ``0 <= f <= 1``.
        GBM fits the systematic on the localization offsets with a
        Gaussian statistical error; here the likelihood is evaluated on
        the full HEALPix maps, so this is an adaptation, not the same fit.

        The P-P curve is a check, not part of the fit: it is computed on
        ``validation_results`` if given, otherwise on a random
        ``validation_fraction`` of ``results`` held out from the fit
        (``validation_fraction = 0`` validates on the fit sample).

        Parameters
        ----------
        results : list of dict
            Same format as ``prepare_skymaps``.
        validation_results : list of dict, optional
            Independent sample for the P-P validation.
        validation_fraction : float
            Fraction of ``results`` held out for validation when
            ``validation_results`` is None.
        sigma_core_bounds, sigma_tail_bounds, f_bounds : (float, float)
            Search ranges. Sigmas in degrees; sigmas much smaller than the
            pixel size are not resolved by the band-limited transform.
        min_separation_deg : float
            ``sigma_tail - sigma_core`` must be at least this value.
        n_bootstrap : int
            Bootstrap resamplings of the fit sample for the parameter
            uncertainties (0 disables them).
        density_floor : float
            Smallest density (sr^-1) entering the logarithm.
        seed : int
            Seed of the split, of the optimizer and of the bootstrap.
        de_kwargs : dict, optional
            Extra arguments of ``scipy.optimize.differential_evolution``.
        output_dir : str, optional
            Where to save ``<prefix>.json`` and the P-P figure
            (``.png`` and ``.pdf``). Default ``self.output_dir``;
            nothing is saved if both are None.
        prefix : str
            File name prefix.
        show : bool
            If True, display the figure.

        Returns
        -------
        dict
            ``sigma_core_best``, ``sigma_tail_best``, ``f_best``,
            ``log_likelihood``, ``optimizer`` (success, message, nit, nfev),
            ``bootstrap`` (std and 16/84 percentiles, or None),
            ``ks_statistic`` / ``ks_pvalue`` against the uniform
            distribution, ``coverage`` (empirical coverage at
            ``COVERAGE_LEVELS``), the same for the original maps
            (``ks_original``, ``coverage_original``),
            ``confidence_levels`` of the validation GRBs, ``n_fit``,
            ``n_validation``, ``validation_on_fit_sample`` and ``files``.
        """
        bounds = [tuple(map(float, b)) for b in (sigma_core_bounds, sigma_tail_bounds, f_bounds)]
        (c_lo, c_hi), (t_lo, t_hi), (f_lo, f_hi) = bounds
        if not 0.0 < c_lo < c_hi:
            raise ValueError(f"sigma_core_bounds must satisfy 0 < low < high, got {sigma_core_bounds}")
        if not t_lo < t_hi or t_hi <= c_lo + min_separation_deg:
            raise ValueError(f"sigma_tail_bounds {sigma_tail_bounds} leave no room for sigma_tail > sigma_core")
        if not 0.0 <= f_lo < f_hi <= 1.0:
            raise ValueError(f"f_bounds must satisfy 0 <= low < high <= 1, got {f_bounds}")

        rng = np.random.default_rng(seed)
        de_kwargs = {"seed": seed, "tol": 1e-8, "popsize": 20, "maxiter": 2000, "polish": True,
                     **(de_kwargs or {})}

        # ---- samples ----
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
        validation_on_fit_sample = val_sample is fit_sample
        if len(fit_sample) < 3 or len(val_sample) < 3:
            raise ValueError(f"Too few GRBs: {len(fit_sample)} to fit, {len(val_sample)} to validate")
        if validation_on_fit_sample:
            print("WARNING: P-P validation on the fit sample, it is not an independent check")

        # ---- fit ----
        C, ell = self._projection_matrix(fit_sample)
        res = self._fit_core_tail(C, ell, bounds, density_floor, de_kwargs, min_separation_deg)
        sigma_core, sigma_tail, f = (float(v) for v in res.x)
        log_like = float(-res.fun)
        optimizer = {
            "success": bool(res.success),
            "message": str(res.message),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
        }

        # ---- bootstrap ----
        bootstrap = None
        if n_bootstrap > 0:
            samples = np.empty((n_bootstrap, 3))
            for b in range(n_bootstrap):
                idx = rng.integers(0, len(C), len(C))
                kw = {**de_kwargs, "seed": int(rng.integers(2**32))}
                samples[b] = self._fit_core_tail(C[idx], ell, bounds, density_floor, kw, min_separation_deg).x
            p16, p84 = np.percentile(samples, [16, 84], axis=0)
            names = ("sigma_core", "sigma_tail", "f")
            bootstrap = {
                "n": n_bootstrap,
                "std": dict(zip(names, samples.std(axis=0, ddof=1).tolist())),
                "p16": dict(zip(names, p16.tolist())),
                "p84": dict(zip(names, p84.tolist())),
                "samples": samples,
            }

        # ---- P-P validation ----
        cl_cal, cl_orig = [], []
        for item in val_sample:
            prob = self.smooth_prepared_map(item, sigma_core, sigma_tail, f)
            if prob is None:
                continue
            cl_cal.append(self._true_pixel_cl(prob, item["ipix_true"]))
            cl_orig.append(self._true_pixel_cl(item["prob"], item["ipix_true"]))
        cl_cal, cl_orig = np.asarray(cl_cal), np.asarray(cl_orig)

        ks = kstest(cl_cal, "uniform")
        ks_orig = kstest(cl_orig, "uniform")
        coverage = {f"{c:.2f}": float(np.mean(cl_cal <= c)) for c in self.COVERAGE_LEVELS}
        coverage_orig = {f"{c:.2f}": float(np.mean(cl_orig <= c)) for c in self.COVERAGE_LEVELS}

        out = {
            "sigma_core_best": sigma_core,
            "sigma_tail_best": sigma_tail,
            "f_best": f,
            "log_likelihood": log_like,
            "optimizer": optimizer,
            "bootstrap": bootstrap,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "coverage": coverage,
            "ks_original": {"statistic": float(ks_orig.statistic), "pvalue": float(ks_orig.pvalue)},
            "coverage_original": coverage_orig,
            "confidence_levels": cl_cal,
            "n_fit": len(fit_sample),
            "n_validation": len(cl_cal),
            "validation_on_fit_sample": validation_on_fit_sample,
            "bounds": {"sigma_core": bounds[0], "sigma_tail": bounds[1], "f": bounds[2]},
            "files": {},
        }
        self._print_core_tail_summary(out)

        # ---- figure and files ----
        fig = self._plot_core_tail_pp(cl_cal, cl_orig, out)
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
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out

    @staticmethod
    def _print_core_tail_summary(out):
        bs = out["bootstrap"]

        def fmt(name, key, unit):
            value = out[key]
            if bs is None:
                return f"    {name:<11}= {value:.3f}{unit}"
            return (f"    {name:<11}= {value:.3f}{unit}  (bootstrap std {bs['std'][name]:.3f}, "
                    f"16-84%: {bs['p16'][name]:.3f} - {bs['p84'][name]:.3f})")

        opt = out["optimizer"]
        print(f"Core+tail ML fit on {out['n_fit']} GRBs: log L = {out['log_likelihood']:.2f}, "
              f"success = {opt['success']} ({opt['nit']} iterations, {opt['nfev']} evaluations)")
        print(fmt("sigma_core", "sigma_core_best", " deg"))
        print(fmt("sigma_tail", "sigma_tail_best", " deg"))
        print(fmt("f", "f_best", ""))
        source = "fit sample" if out["validation_on_fit_sample"] else "independent sample"
        print(f"P-P validation on {out['n_validation']} GRBs ({source}): "
              f"KS = {out['ks_statistic']:.3f}, p = {out['ks_pvalue']:.3g} "
              f"(original maps: KS = {out['ks_original']['statistic']:.3f}, "
              f"p = {out['ks_original']['pvalue']:.3g})")
        n = out["n_validation"]
        print(f"    {'level':>6}{'calibrated':>12}{'original':>10}{'binomial 1 sigma':>18}")
        for level, cov in out["coverage"].items():
            c = float(level)
            print(f"    {level:>6}{cov:>12.3f}{out['coverage_original'][level]:>10.3f}"
                  f"{np.sqrt(c * (1 - c) / n):>18.3f}")

    def _plot_core_tail_pp(self, cl_cal, cl_orig, out):
        """P-P curve of the calibrated and original maps, with binomial bands for the sample size."""
        n = len(cl_cal)
        grid = np.linspace(0.0, 1.0, 201)
        bands = self.pp_binomial_bands(grid, n)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for nsigma, alpha in ((3, 0.12), (2, 0.18), (1, 0.28)):
            ax.fill_between(grid, bands[nsigma][0], bands[nsigma][1], color="tab:blue",
                            alpha=alpha, lw=0, label=rf"${nsigma}\sigma$ binomial")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal calibration")

        for cl, style, label in (
            (cl_orig, {"color": "0.4", "ls": ":"}, "Original maps"),
            (cl_cal, {"color": "tab:red"}, "Core+tail calibrated"),
        ):
            x = np.concatenate([[0.0], np.sort(cl), [1.0]])
            y = np.concatenate([[0.0], np.arange(1, len(cl) + 1) / len(cl), [1.0]])
            ax.step(x, y, where="post", lw=1.8, label=label, **style)

        bs = out["bootstrap"]
        err = (lambda k: rf" \pm {bs['std'][k]:.2f}") if bs else (lambda k: "")
        text = "\n".join([
            rf"$\sigma_{{\rm core}} = {out['sigma_core_best']:.2f}{err('sigma_core')}^\circ$",
            rf"$\sigma_{{\rm tail}} = {out['sigma_tail_best']:.2f}{err('sigma_tail')}^\circ$",
            rf"$f = {out['f_best']:.3f}{err('f')}$",
            rf"KS $= {out['ks_statistic']:.3f}$ ($p = {out['ks_pvalue']:.2g}$)",
            rf"$N = {n}$",
        ])
        ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=10,
                bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"})

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Credible level", fontsize=12)
        ax.set_ylabel("Fraction of GRBs within credible level", fontsize=12)
        ax.tick_params(labelsize=11, direction="in", top=True, right=True)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        fig.tight_layout()
        return fig

    @staticmethod
    def _load_pickle(path):
        """Load a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


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

import pickle
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic, ICRS
from scipy.stats import binom, norm
from bctools.loc import TSMap, NormLocLike
from scoords import Attitude, SpacecraftFrame


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
        
        #s_counts = [100000,2000,2000,2000,2000,2000]
        #b_counts = [1000,1000,1000,1000,1000,1000]
        
        #bgo_z1[keV] 1591
        #bgo_z0[keV] 605
        #bgo_x1[keV] 16835
        #bgo_x0[keV] 257
        #bgo_y1[keV] 1291
        #bgo_y0[keV] 797
                
        #remap the counts following BGO IRF ['BGO_X0', 'BGO_X1', 'BGO_Y0', 'BGO_Y1', 'BGO_Z0', 'BGO_Z1']
        #change order of input to follow BGO IRF.  
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
    def smooth_prepared_map(item, sigma_deg):
        """
        Apply a Gaussian beam to a prepared probability map.

        ``sigma_deg`` is the Gaussian standard deviation in degrees,
        not the FWHM. ``sigma_deg = 0`` returns the unsmoothed map.

        The beam in harmonic space is
        ``B_l = exp[-0.5 l(l+1) sigma^2]``.
        Tiny negative values from the inverse transform are clipped,
        then the map is renormalized.

        Parameters
        ----------
        item : dict
            Output of ``prepare_skymaps`` (must include ``prob``,
            ``alm``, ``nside``, ``lmax``).
        sigma_deg : float
            Gaussian sigma in degrees.

        Returns
        -------
        ndarray or None
            Smoothed probability map, or None if the result is invalid.
        """
        if sigma_deg == 0:
            return item["prob"].copy()

        sigma_rad = np.deg2rad(sigma_deg)
        ell = np.arange(item["lmax"] + 1)
        beam = np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)
        prob = hp.alm2map(
            hp.almxfl(item["alm"], beam),
            nside=item["nside"],
            lmax=item["lmax"],
        )
        # Numerical inverse transforms can yield ~-1e-15
        prob = np.clip(prob, 0.0, None)
        total = prob.sum()
        if total <= 0 or not np.isfinite(total):
            return None
        return prob / total

    def pp_curve_smoothed(self, prepared, values, sigma_deg):
        """
        Build the P-P curve at a given systematic Gaussian sigma.

        For each GRB the credible level of the true position is the
        sum of all pixels with probability >= that of the true pixel.
        The P-P fraction at containment C is the fraction of GRBs
        with that credible level <= C.

        Parameters
        ----------
        prepared : list of dict
            Output of ``prepare_skymaps``.
        values : array
            Credible levels at which to evaluate the curve (e.g. 0 to 1
            in steps of 0.01).
        sigma_deg : float
            Systematic Gaussian sigma in degrees (0 = no smoothing).

        Returns
        -------
        credible_levels : ndarray
            True-position credible level of each GRB.
        fraction : ndarray
            Contained fraction at each value in ``values``.
        """
        cls = []
        for item in prepared:
            prob = self.smooth_prepared_map(item, sigma_deg)
            if prob is None:
                continue
            p_true = prob[item["ipix_true"]]
            cls.append(np.sum(prob[prob >= p_true]))
        cls = np.asarray(cls)
        fraction = np.searchsorted(np.sort(cls), values, side="right") / len(cls)
        return cls, fraction

    @staticmethod
    def pp_binomial_bands(values, N):
        """
        1, 2 and 3 sigma binomial bands around the P-P diagonal.

        For a perfectly calibrated sample of size N, the contained
        fraction at each credible level is a binomial draw. The bands
        are the central interval of that distribution, with
        ``alpha = 2 * (1 - Phi(n_sigma))``.

        Parameters
        ----------
        values : array
            Credible levels (nominal coverage).
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

        For each credible level C from 0 to 1 in 1% steps, plot the
        fraction of GRBs whose true position falls inside the C region.
        1, 2 and 3 sigma binomial bands around the diagonal are shown.

        Parameters
        ----------
        results : list of dict
            Same format as ``prepare_skymaps`` (``sky_map`` plus true
            Galactic ``source_galactic_l`` / ``source_galactic_b``).
        values : array, optional
            Credible levels. Default is 0, 0.01, ..., 1.00.
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
            Credible levels. Default is 0, 0.01, ..., 1.00.
        show : bool
            If True, display the figure.

        Returns
        -------
        dict
            ``values``, ``per_sigma`` (credible levels, P-P fraction,
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
                "credible_levels": cl,
                "fraction": fraction,
                "coverage_90": cov90,
            }

        N = len(next(iter(per_sigma.values()))["credible_levels"])
        bands = self.pp_binomial_bands(values, N)

        plt.figure(figsize=(8, 7))
        plt.fill_between(values, bands[3][0], bands[3][1], alpha=0.15, label=r"$3\sigma$ expected")
        plt.fill_between(values, bands[2][0], bands[2][1], alpha=0.20, label=r"$2\sigma$ expected")
        plt.fill_between(values, bands[1][0], bands[1][1], alpha=0.30, label=r"$1\sigma$ expected")
        for sigma_deg, out in per_sigma.items():
            plt.plot(
                values, out["fraction"], linewidth=2,
                label=rf"$\sigma_{{\rm sys}}={sigma_deg:g}^\circ$",
            )
        plt.plot(values, values, "--", color="black", linewidth=1.5, label="Perfect calibration")
        plt.xlabel("Credible Level")
        plt.ylabel("Fraction contained")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return {"values": values, "per_sigma": per_sigma, "bands": bands, "N": N}

    def smooth_skymap_area(self, sky_map, sigma_deg, conf_level=0.9):
        """
        Smooth one probability sky map with a calibrated systematic kernel.

        Runtime helper: after choosing ``sigma_deg`` from
        ``calibrate_pp_coverage``, apply that Gaussian to a single map
        and return the new containment area at ``conf_level``.

        Parameters
        ----------
        sky_map : map object
            Probability HEALPix map (uses ``_data``).
        sigma_deg : float
            Gaussian systematic sigma in degrees. 0 skips smoothing.
        conf_level : float
            Containment fraction, default 0.9.

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
        if sigma_deg != 0:
            item["alm"] = hp.map2alm(prob, lmax=item["lmax"], iter=0)
        prob = self.smooth_prepared_map(item, sigma_deg)

        # Smallest set of pixels whose probability sums to conf_level
        order = np.argsort(prob)[::-1]
        n_pix = int(np.searchsorted(np.cumsum(prob[order]), conf_level) + 1)
        area = n_pix * hp.nside2pixarea(nside, degrees=True)
        return {
            "prob": prob,
            "cont_area_deg2": float(area),
            "eq_radius_deg": float(np.sqrt(area / np.pi)),
        }

    @staticmethod
    def _load_pickle(path):
        """Load a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

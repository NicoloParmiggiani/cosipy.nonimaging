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

    def __init__(self, soft_loctable_path, medium_loctable_path, hard_loctable_path, nside=64):
        # Store HEALPix Nside
        self.nside = nside
        
        # Load all three IRFs at initialization
        # The IRFs have been generated using the bc-tools package:
        #soft_spectrum = BandFunction._from_megalib(['BandFunction',10,10000,-1.9,-3.7,230],"10.0")
        #medium_spectrum = BandFunction._from_megalib(['BandFunction',10,10000,-1,-2.3,699.9],"10.0")
        #hard_spectrum = Comptonized._from_megalib(['Comptonized',10,10000,-0.5,1500],"10.0")
        #soft_local_loctable = LocalLocTable.from_irf(irf, soft_spectrum,energy_channels = 1) # [80,2000]
        #medium_local_loctable = LocalLocTable.from_irf(irf, medium_spectrum,energy_channels = 1)
        #hard_local_loctable = LocalLocTable.from_irf(irf, hard_spectrum,energy_channels = 1)
             
        print(self._load_pickle(soft_loctable_path))
             
        self.loctables = {
            "soft": self._load_pickle(soft_loctable_path),
            "medium": self._load_pickle(medium_loctable_path),
            "hard": self._load_pickle(hard_loctable_path),
        }
        
    def rotate_tsmap(self,ts_map, attitude):
        """
        Rotate a HEALPix TS map using spacecraft attitude.

        Parameters
        ----------
        ts_map : TSMap
            TSMap object (must have .data attribute)
        attitude : Attitude
            Spacecraft attitude (created with Attitude.from_axes)

        Returns
        -------
        np.ndarray
            Rotated HEALPix map
        """

        data = ts_map.data
        npix = len(data)
        nside = hp.npix2nside(npix)

        # all pixel directions
        ipix = np.arange(npix)
        theta, phi = hp.pix2ang(nside, ipix)

        # build "fake ICRS" coordinates
        ra = phi * u.rad
        dec = (np.pi/2 - theta) * u.rad

        coords = SkyCoord(ra=ra, dec=dec, frame=ICRS())

        # go through spacecraft frame (this applies the attitude rotation)
        coords_sc = coords.transform_to(SpacecraftFrame(attitude=attitude))

        # back to real sky
        coords_real = coords_sc.transform_to(ICRS())

        # convert back to healpix angles
        theta_new = np.pi/2 - coords_real.dec.to(u.rad).value
        phi_new = coords_real.ra.to(u.rad).value

        # interpolate rotated map
        data_rot = hp.get_interp_val(data, theta_new, phi_new)

        return data_rot

    def localize(self, s_counts, b_counts, attitude=None, conf_level=0.9):
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
        
        s_counts = s_counts[[3, 2, 5, 4, 1, 0]]
        b_counts = b_counts[[3, 2, 5, 4, 1, 0]]
        
        print(s_counts)
        print(b_counts)

        results = []
        
        # The local_loctable contains the expected rates in spacecraft coordinates
        # We now need to use this to estimate the total expected counts in sky coordinate for
        # the full duration of an event. 
        # In this case we simply have a 1 second event and specifying the attitude by a quaternion
        # ([0,0,0,1] corresponds to the identity rotation). You can have multiple attitude-duration
        # pairs to correctly model long duration events.

        print(self.loctables['soft'])
   
        soft_sky_loctable = self.loctables['soft'].to_skyloctable(attitude = attitude, duration = 1)
        medium_sky_loctable = self.loctables['medium'].to_skyloctable(attitude = attitude, duration = 1)
        hard_sky_loctable = self.loctables['hard'].to_skyloctable(attitude = attitude, duration = 1)
        
        self.luts = {
            "soft": soft_sky_loctable,
            "medium": medium_sky_loctable,
            "hard": hard_sky_loctable,
        }

        for label, lut in self.luts.items():

            lut.set_background(b_counts)
            lut.set_data(s_counts)

            ts_map = TSMap(nside=self.nside, coordsys="icrs")
            likelihood = NormLocLike(lut)
            ts_map.compute(likelihood)

            ts_value = float(np.max(ts_map))
            best = ts_map.best_loc()

            cont_area = ts_map.error_area(cont=conf_level).to(u.deg**2)
            eq_radius = np.sqrt(cont_area / np.pi).to(u.deg)

            # Default outputs
            theta_out = -1.0
            phi_out = -1.0
            l_out = -1.0
            b_out = -1.0

            if attitude is None:
                # return theta/phi (local)
                ra = best.ra.to(u.rad).value
                dec = best.dec.to(u.rad).value

                theta_out = np.pi / 2.0 - dec   # zenith
                phi_out = ra                   # azimuth

                out_map = ts_map.data

            else:
                # convert to galactic
                ra = best.ra.to(u.rad).value
                dec = best.dec.to(u.rad).value

                theta = np.pi / 2.0 - dec
                phi = ra

                azimuth = phi * u.rad
                zenith = theta * u.rad

                position_sc = SkyCoord(
                    lon=azimuth,
                    lat=90.0 * u.deg - zenith.to(u.deg),
                    frame=SpacecraftFrame(attitude=attitude)
                )

                position_gal = position_sc.transform_to(Galactic())

                l_out = float(position_gal.l.deg)
                b_out = float(position_gal.b.deg)
                
                ts_map_rot = self.rotate_tsmap(ts_map, attitude)
                
                out_map = HealpixMap(
                    ts_map_rot,
                    nside=ts_map.nside,
                    coordsys=ts_map.coordsys
                )

            results.append({
                "label": label,
                "ts_map": out_map,
                "ts_value": ts_value,
                "sqrt_ts": float(np.sqrt(ts_value)),
                "theta": float(theta_out),
                "phi": float(phi_out),
                "l": float(l_out),
                "b": float(b_out),
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
                color="blue",
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

import pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from bctools.loc import TSMap, NormLocLike


class BGOLocalizerBCT:
    """
    Simple class to localize GRBs using three BGO Look-Up Tables (soft/medium/hard).

    Parameters
    ----------
    soft_lut_path : str
        Path to the pickled LUT for the 'soft' spectrum.
    medium_lut_path : str
        Path to the pickled LUT for the 'medium' spectrum.
    hard_lut_path : str
        Path to the pickled LUT for the 'hard' spectrum.
    nside : int
        HEALPix Nside used to compute TS maps.

    Notes
    -----
    The input counts (s_counts and b_counts) must follow the correct order
    of BGO detector panels to match the LUT definition:

        ['BGO_X0', 'BGO_X1', 'BGO_Y0', 'BGO_Y1', 'BGO_Z0', 'BGO_Z1']
    """

    def __init__(self, soft_lut_path, medium_lut_path, hard_lut_path, nside=64):
        # Store HEALPix Nside
        self.nside = nside
        # Load all three LUTs at initialization
        self.luts = {
            "soft": self._load_pickle(soft_lut_path),
            "medium": self._load_pickle(medium_lut_path),
            "hard": self._load_pickle(hard_lut_path),
        }

    def localize(self, s_counts, b_counts):
        """
        Run localization using all LUTs and return the one with the highest TS.

        Parameters
        ----------
        s_counts : list or np.ndarray
            Source counts in the following order:
            ['BGO_X0', 'BGO_X1', 'BGO_Y0', 'BGO_Y1', 'BGO_Z0', 'BGO_Z1']
        b_counts : list or np.ndarray
            Background counts in the same order.

        Returns
        -------
        dict
            Dictionary containing the best localization result.
        """
        results = []

        for label, lut in self.luts.items():
            # Update LUT with data and background
            lut.set_background(b_counts)
            lut.set_data(s_counts)

            # Compute the TS map for this LUT
            ts_map = TSMap(nside=self.nside, coordsys="icrs")
            likelihood = NormLocLike(lut)
            ts_map.compute(likelihood)

            # Extract localization statistics
            ts_value = float(np.max(ts_map))
            best = ts_map.best_loc()
            cont_area = ts_map.error_area(cont=0.9).to(u.deg**2)
            eq_radius = np.sqrt(cont_area / np.pi).to(u.deg)

            results.append({
                "label": label,
                "ts_map": ts_map,
                "ts_value": ts_value,
                "sqrt_ts": float(np.sqrt(ts_value)),
                "ra_deg": best.ra.deg,
                "dec_deg": best.dec.deg,
                "cont_area_deg2": float(cont_area.value),
                "eq_radius_deg": float(eq_radius.value),
            })

        # Return the result with the highest TS value
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
        import healpy as hp
        import matplotlib.pyplot as plt
        import astropy.units as u
        from matplotlib.lines import Line2D

        ts_map = result["ts_map"]
        img, ax = ts_map.plot()
        ax.grid(alpha=0.5)

        if true_coord is not None:
            # Actual location of simulated source
            ax.scatter(
                true_coord.ra.to(u.deg).value,
                true_coord.dec.to(u.deg).value,
                color="red",
                transform=ax.get_transform("world"),
                s=2,
                label="True source"
            )
            
        ax.scatter(
                ts_map.best_loc().ra.to(u.deg).value,
                ts_map.best_loc().dec.to(u.deg).value,
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
            f"RA={result['ra_deg']:.2f}  "
            f"Dec={result['dec_deg']:.2f}  "
            f"r90={result['eq_radius_deg']:.2f} deg"
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

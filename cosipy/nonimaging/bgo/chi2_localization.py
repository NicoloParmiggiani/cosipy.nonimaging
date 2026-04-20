import math
import numpy as np
import pickle
from scipy.stats import chi2
import healpy as hp
from astropy import units as u
from scoords import Attitude, SpacecraftFrame
from astropy.coordinates import SkyCoord, Galactic

class BGOLocalizerChi2:
    """
    Class for GRB localization using three BGO lookup tables
    corresponding to soft, medium, and hard spectra.

    Parameters
    ----------
    soft_lut_path : str
        Path to the lookup table for the soft spectrum.
    medium_lut_path : str
        Path to the lookup table for the medium spectrum.
    hard_lut_path : str
        Path to the lookup table for the hard spectrum.
    nside : int, optional
        HEALPix Nside used to compute TS maps. Default is 32.

    Notes
    -----
    The input count arrays must follow the detector order used by the LUTs:

        ['BGO_Z1', 'BGO_Z0', 'BGO_X1', 'BGO_X0', 'BGO_Y1', 'BGO_Y0']
    """
    
    def __init__(self, soft_lut_path, medium_lut_path, hard_lut_path, nside=32):
        
        # Store the HEALPix resolution parameter
        self.nside = nside

        # Load the precomputed lookup tables at initialization
        self.luts = {
            "soft": np.load(soft_lut_path),
            "medium": np.load(medium_lut_path),
            "hard": np.load(hard_lut_path),
        }
        
        
    def calculate_chi_squared_optimized(self,s, b, m):
        
        """
        Compute the chi-squared value for each position in the grid in an optimized way.
        
        Equation from: V. Connaughton et. al. 2015 https://iopscience.iop.org/article/10.1088/0067-0049/216/2/32
        
        Parameters
        ----------
        signal_counts : ndarray of shape (6,)
            Observed detector counts.
        background_counts : ndarray of shape (6,) or compatible
            Background detector counts.
        model_counts : ndarray
            Model counts for each sky position and detector.

        Returns
        -------
        chi_squared : ndarray
            Chi-squared value for each sky position.
        """
        
        # Use only the first 6 detectors
        indices = np.arange(6)
        
        # Extract model counts for these detectors
        m_subset = m[:, indices]  # Shape: (41168, 6)

        # Compute numerator and denominator for normalization factor f_i
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = np.sum(m_subset * (s[indices] - b[indices]) / s[indices], axis=1)
            denominator = np.sum((m_subset**2) / s[indices], axis=1)

            # Avoid division by zero
            f_i = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

            # Expand f_i for broadcasting
            f_i_expanded = f_i[:, np.newaxis]  # Shape: (41168, 1)

            # Compute chi-squared terms
            chi_numerator = (s[indices] - b[indices] - f_i_expanded * m_subset)**2
            chi_denominator = b[indices] + f_i_expanded * m_subset

            # Safe division for chi-squared elements
            chi_squared_elements = np.divide(
                chi_numerator, chi_denominator,
                out=np.full_like(chi_numerator, np.finfo(np.float64).max),
                where=chi_denominator != 0
            )

        # Sum over detectors to obtain chi^2 per direction
        chi_squared = np.sum(chi_squared_elements, axis=1)

        return chi_squared

    def localize(self, s_counts, b_counts, cont_area_level=0.9, attitude=None):
        
        """
        Localize a GRB by selecting the best spectral model
        (soft, medium, or hard) via chi-squared minimization.

        Parameters
        ----------
        s_counts : ndarray
            Observed detector counts.
        b_counts : ndarray
            Background detector counts.
        cont_area_level : float, optional
            Confidence level used to compute the containment area
            (default is 0.9, i.e., 90%).
        attitude : scoords.Attitude or None, optional
            Spacecraft attitude. If provided, the best-fit direction
            (theta, phi) is interpreted in the spacecraft frame and
            converted to Galactic coordinates (l, b).
            If None, the result is returned in instrument coordinates
            (theta, phi).

        Returns
        -------
        results : dict
            Dictionary containing:
                - theta, phi : best-fit direction in instrument coordinates
                (set to -1 if attitude is provided)
                - l, b : Galactic coordinates (set to -1 if no attitude)
                - min_chi2 : minimum chi-squared value
                - chi2_map : chi-squared map for the best spectral model
                - best_fit_spectrum : selected spectral model
                - cont_area : containment area in deg^2
        """

        spectrum_keys = ["soft", "medium", "hard"]
        
        chi2_by_spectrum = {}
        min_chi2_values = []

        # Compute chi-squared maps for each spectral model
        for spectrum_key in spectrum_keys:
            chi2_array = self.calculate_chi_squared_optimized(
                s_counts,
                b_counts,
                self.luts[spectrum_key],
            )
            chi2_by_spectrum[spectrum_key] = chi2_array
            min_chi2_values.append(np.min(chi2_array))
            
        # Select the spectral model with the global minimum chi-squared
        best_spectrum_index = int(np.argmin(min_chi2_values))
        best_spectrum = spectrum_keys[best_spectrum_index]
        chi2_map = chi2_by_spectrum[best_spectrum]
        best_lut = self.luts[best_spectrum]
        min_chi2 = min_chi2_values[best_spectrum_index]

        # Identify the best-fit sky position (minimum of chi2 map)
        best_position_index = int(np.argmin(chi2_map))
        theta_best = best_lut[best_position_index][6]
        phi_best = best_lut[best_position_index][7]
        
        # Compute chi-squared threshold for the desired confidence region
        delta_chi2 = chi2.ppf(cont_area_level, 2)
        limit = min_chi2 + delta_chi2

        # Healpix bookkeeping
        nside = hp.get_nside(chi2_map)
        pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)

        # Select pixels within the confidence region
        mask_inside = chi2_map < limit
        npix_in_region = np.sum(mask_inside)

        # Compute containment area in square degrees
        cont_area = npix_in_region * pix_area_deg2

        # Default outputs: return instrument coordinates
        theta_out = theta_best
        phi_out = phi_best
        l_out = -1.0
        b_out = -1.0

        # If attitude is available, convert to Galactic coordinates
        if attitude is not None:
            print("attitude")

            azimuth = phi_best * u.deg
            zenith = theta_best * u.deg

            # Build coordinate in spacecraft reference frame
            position_sc = SkyCoord(
                azimuth,
                90.0 * u.deg - zenith.to(u.deg),  # convert zenith -> elevation
                representation_type='spherical',
                frame=SpacecraftFrame(attitude=attitude)
            )

            # Transform spacecraft coordinates to Galactic frame
            position_gal = position_sc.transform_to(Galactic())

            # Store Galactic coordinates and invalidate theta/phi
            l_out = position_gal.l.to(u.deg).value
            b_out = position_gal.b.to(u.deg).value

            theta_out = -1.0
            phi_out = -1.0

        results = {
            "theta": theta_out,
            "phi": phi_out,
            "l": l_out,
            "b": b_out,
            "min_chi2": min_chi2,
            "chi2_map": chi2_map,
            "best_fit_spectrum": best_spectrum,
            "cont_area": cont_area
        }
        
        return results
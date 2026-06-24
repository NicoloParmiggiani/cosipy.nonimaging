import pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic, ICRS
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

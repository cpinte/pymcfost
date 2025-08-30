"""
Chi² computation functions for MCFOST model fitting.

This module contains functions for computing chi² values from different types of
observational data (ALMA interferometric data, SED data) and combining them
for multi-wavelength fitting.
"""

import numpy as np
import os
from astropy.io import fits
from scipy.interpolate import interp1d
from .SED import SED


class ObservedSED:
    """
    A class to handle observed SED data for chi² calculations.

    This class stores observed SED data in W/m² units and provides methods
    for data validation and manipulation.

    Attributes:
        wavelength (ndarray): Wavelength array in microns
        flux (ndarray): Flux array in W/m²
        error (ndarray): Flux error array in W/m²
        n_points (int): Number of data points
    """

    def __init__(self, wavelength=None, flux=None, error=None, sed_data_path=None):
        """
        Initialize ObservedSED object.

        Args:
            wavelength (ndarray, optional): Wavelength array in microns
            flux (ndarray, optional): Flux array in W/m²
            error (ndarray, optional): Flux error array in W/m²
            sed_data_path (str, optional): Path to observed SED data file

        Notes:
            Either provide (wavelength, flux, error) arrays OR sed_data_path.
            If sed_data_path is provided, the arrays will be loaded from the file.
        """
        if sed_data_path is not None:
            # Load from file
            loaded_data = self._read_from_file(sed_data_path)
            if loaded_data is None:
                raise ValueError(f"Failed to load SED data from {sed_data_path}")
            wavelength, flux, error = loaded_data

        if wavelength is None or flux is None or error is None:
            raise ValueError("Must provide either (wavelength, flux, error) arrays or sed_data_path")

        self.wavelength = np.asarray(wavelength)
        self.flux = np.asarray(flux)
        self.error = np.asarray(error)
        self.n_points = len(self.wavelength)

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate the SED data."""
        if len(self.wavelength) != len(self.flux) or len(self.wavelength) != len(self.error):
            raise ValueError("All arrays must have the same length")

        if np.any(self.wavelength <= 0):
            raise ValueError("Wavelength values must be positive")

        if np.any(self.error <= 0):
            raise ValueError("Error values must be positive")

        if np.any(~np.isfinite(self.wavelength)) or np.any(~np.isfinite(self.flux)) or np.any(~np.isfinite(self.error)):
            raise ValueError("All values must be finite")

    def _read_from_file(self, sed_data_path):
        """
        Read observed SED data from file and convert to W/m² units.

        Parameters
        ----------
        sed_data_path : str
            Path to the observed SED data file

        Returns
        -------
        tuple or None
            (wavelength, flux_Wm2, error_Wm2) arrays or None if failed

        Notes
        -----
        Expected SED data format (observed_sed.txt):
        wavelength_micron flux_jy flux_error_jy
        1.25 0.5 0.05
        2.2 1.2 0.12
        ...
        """
        try:
            if not os.path.exists(sed_data_path):
                raise FileNotFoundError(f"SED data file {sed_data_path} not found.")

            obs_data = np.loadtxt(sed_data_path)
            obs_wl = obs_data[:, 0]  # wavelength in microns
            obs_flux_jy = obs_data[:, 1]  # flux in Jy
            try:
                obs_error_jy = obs_data[:, 2]  # flux error in Jy
            except IndexError:
                obs_error_jy = 0.1 * obs_flux_jy

            # Convert from Jy to W/m²
            # F_ν (Jy) * ν (Hz) = F_λ (W/m²/Hz) * ν (Hz) = F_λ (W/m²)
            # ν = c/λ, where c = 3e14 μm/s
            obs_flux_Wm2 = obs_flux_jy * 1e-26 * (3e14 / obs_wl)  # Convert Jy to W/m²
            obs_error_Wm2 = obs_error_jy * 1e-26 * (3e14 / obs_wl)  # Convert Jy to W/m²

            print(f"Loaded observed SED with {len(obs_wl)} points from {sed_data_path}")
            return obs_wl, obs_flux_Wm2, obs_error_Wm2

        except Exception as e:
            print(f"Error reading observed SED: {e}")
            return None

    @classmethod
    def from_file(cls, sed_data_path):
        """
        Create ObservedSED object from file.

        Parameters
        ----------
        sed_data_path : str
            Path to the observed SED data file

        Returns
        -------
        ObservedSED
            ObservedSED object containing the data

        Notes
        -----
        Expected SED data format (observed_sed.txt):
        wavelength_micron flux_jy flux_error_jy
        1.25 0.5 0.05
        2.2 1.2 0.12
        ...
        """
        return cls(sed_data_path=sed_data_path)

    def filter_wavelength_range(self, wl_min, wl_max):
        """
        Filter data to a specific wavelength range.

        Args:
            wl_min (float): Minimum wavelength in microns
            wl_max (float): Maximum wavelength in microns

        Returns:
            ObservedSED: New ObservedSED object with filtered data
        """
        mask = (self.wavelength >= wl_min) & (self.wavelength <= wl_max)
        return ObservedSED(
            self.wavelength[mask],
            self.flux[mask],
            self.error[mask]
        )

    def get_valid_points(self):
        """
        Get data points with positive, finite flux values.

        Returns:
            ObservedSED: New ObservedSED object with valid data points
        """
        mask = (self.flux > 0) & np.isfinite(self.flux)
        return ObservedSED(
            self.wavelength[mask],
            self.flux[mask],
            self.error[mask]
        )

    def __len__(self):
        """Return the number of data points."""
        return self.n_points

    def __str__(self):
        """String representation of the ObservedSED object."""
        return f"ObservedSED with {self.n_points} points, wavelength range: {self.wavelength.min():.2f}-{self.wavelength.max():.2f} μm"



def compute_chi2_sed(mcfost_sed, observed_sed, i=0, iaz=0):
    """
    Compute chi² for SED fitting using MCFOST SED class.

    Parameters
    ----------
    mcfost_sed : SED or str
        MCFOST SED object or path to the model SED directory/fits file
    observed_sed : ObservedSED
        ObservedSED object containing the observed data
    i : int, optional
        Inclination index for the model SED. Defaults to 0.
    iaz : int, optional
        Azimuth angle index for the model SED. Defaults to 0.

    Returns
    -------
    float
        Normalized chi² value
    """
    try:
        # Load model SED data
        if isinstance(mcfost_sed, SED):
            # Use provided SED object
            sed_model = mcfost_sed
            model_wl = sed_model.wl  # wavelength in microns
            model_flux_Wm2 = sed_model.sed[0, iaz, i, :]  # flux in W/m²
        elif isinstance(mcfost_sed, str):
            # Load from path
            if not os.path.exists(mcfost_sed):
                print(f"Warning: Model SED path {mcfost_sed} not found. Skipping SED chi² calculation.")
                return 0.0

            # Check if mcfost_sed is a directory (MCFOST output) or a fits file
            if os.path.isdir(mcfost_sed):
                # Load from MCFOST output directory
                sed_model = SED(mcfost_sed)
                model_wl = sed_model.wl  # wavelength in microns
                model_flux_Wm2 = sed_model.sed[0, iaz, i, :]  # flux in W/m²
            else:
                # Load from fits file directly
                try:
                    hdu = fits.open(mcfost_sed)
                    if len(hdu) > 1:
                        model_wl = hdu[1].data  # wavelength extension
                        model_flux_Wm2 = hdu[0].data[0, iaz, i, :]  # flux extension (already in W/m²)
                    else:
                        # Single extension - try to get wavelength from header
                        model_flux_Wm2 = hdu[0].data
                        n_wl = hdu[0].header.get('NWL', len(model_flux_Wm2))
                        wl_min = hdu[0].header.get('WL_MIN', 0.1)
                        wl_max = hdu[0].header.get('WL_MAX', 1000.0)
                        model_wl = np.logspace(np.log10(wl_min), np.log10(wl_max), n_wl)
                    hdu.close()
                except Exception as e:
                    print(f"Error loading SED from fits file: {e}")
                    return 0.0
        else:
            raise ValueError("mcfost_sed must be an SED object or a string path")

        # Interpolate model to observed wavelengths
        # Handle cases where model wavelength range doesn't cover observed range
        wl_min = max(observed_sed.wavelength.min(), model_wl.min())
        wl_max = min(observed_sed.wavelength.max(), model_wl.max())

        # Filter data to common wavelength range
        mask_obs = (observed_sed.wavelength >= wl_min) & (observed_sed.wavelength <= wl_max)
        mask_model = (model_wl >= wl_min) & (model_wl <= wl_max)

        if np.sum(mask_obs) == 0 or np.sum(mask_model) == 0:
            print("Warning: No overlap in wavelength range between observed and model SEDs.")
            return 0.0

        obs_wl_filt = observed_sed.wavelength[mask_obs]
        obs_flux_filt = observed_sed.flux[mask_obs]
        obs_error_filt = observed_sed.error[mask_obs]
        model_wl_filt = model_wl[mask_model]
        model_flux_filt = model_flux_Wm2[mask_model]

        # Interpolate model to observed wavelengths
        try:
            interp_func = interp1d(model_wl_filt, model_flux_filt,
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
            model_flux_interp = interp_func(obs_wl_filt)
        except Exception as e:
            print(f"Warning: Interpolation failed for SED: {e}")
            return 0.0

        # Remove any negative or invalid values
        valid_mask = (model_flux_interp > 0) & np.isfinite(model_flux_interp)
        if np.sum(valid_mask) == 0:
            print("Warning: No valid model flux values after interpolation.")
            return 0.0

        obs_wl_final = obs_wl_filt[valid_mask]
        obs_flux_final = obs_flux_filt[valid_mask]
        obs_error_final = obs_error_filt[valid_mask]
        model_flux_final = model_flux_interp[valid_mask]

        # Compute chi²
        chi2 = np.sum(((obs_flux_final - model_flux_final) / obs_error_final) ** 2)

        # Normalize by number of data points
        chi2_normalized = chi2 / len(obs_flux_final)

        print(f"SED chi²: {chi2:.2f} (normalized: {chi2_normalized:.2f}) for {len(obs_flux_final)} points")

        return chi2_normalized

    except Exception as e:
        print(f"Error computing SED chi²: {e}")
        return 0.0


def compute_combined_chi2(model_fits_path, mcfost_sed, data_cube_path, observed_sed,
                         noise_std, use_alma=True, use_sed=True, i=0, iaz=0):
    """
    Compute combined chi² from both ALMA data and SED data.

    Parameters
    ----------
    model_fits_path : str or None
        Path to the model fits file (ALMA/line data)
    mcfost_sed : SED or str
        MCFOST SED object or path to the model SED directory/fits file
    data_cube_path : str
        Path to the observed ALMA data cube
    observed_sed : ObservedSED or str
        ObservedSED object or path to observed SED file
    noise_std : float
        Standard deviation of the noise for ALMA data
    use_alma : bool
        Whether to include ALMA data in chi² calculation
    use_sed : bool
        Whether to include SED data in chi² calculation
    i : int, optional
        Inclination index for the model SED. Defaults to 0.
    iaz : int, optional
        Azimuth angle index for the model SED. Defaults to 0.

    Returns
    -------
    float
        Combined chi² value (average across all components)
    """
    chi2_total = 0.0
    n_components = 0

    # ALMA chi²
    if use_alma and model_fits_path is not None and os.path.exists(data_cube_path):
        try:
            chi2_alma = compute_chi2_alma_cube(model_fits_path, data_cube_path, noise_std)
            chi2_total += chi2_alma
            n_components += 1
            print(f"ALMA chi²: {chi2_alma:.2f}")
        except Exception as e:
            print(f"Error computing ALMA chi²: {e}")

    # SED chi²
    if use_sed and mcfost_sed is not None:
        try:
            # Handle observed SED data
            if isinstance(observed_sed, ObservedSED):
                # Use provided ObservedSED object
                obs_sed_obj = observed_sed
            elif isinstance(observed_sed, str):
                # Read from file
                try:
                    obs_sed_obj = ObservedSED.from_file(observed_sed)
                except Exception as e:
                    raise ValueError(f"Failed to read observed SED data: {e}")
            else:
                raise ValueError("observed_sed must be an ObservedSED object or string path")

            chi2_sed = compute_chi2_sed(mcfost_sed, obs_sed_obj, i=i, iaz=iaz)
            chi2_total += chi2_sed
            n_components += 1
        except Exception as e:
            print(f"Error computing SED chi²: {e}")

    if n_components == 0:
        print("Warning: No valid chi² components computed.")
        return float("inf")

    # Return average chi²
    return chi2_total / n_components


def validate_parameters(params):
    """
    Validate that parameters are within reasonable bounds.

    Parameters
    ----------
    params : dict
        Dictionary of parameter values

    Returns
    -------
    bool
        True if parameters are valid

    Raises
    ------
    ValueError
        If parameters are invalid
    """
    # Add any additional validation logic here
    if 'rin' in params and 'rout' in params:
        if params['rin'] >= params['rout']:
            raise ValueError("Inner radius must be less than outer radius")

    if 'amin' in params and 'amax' in params:
        if params['amin'] >= params['amax']:
            raise ValueError("Minimum grain size must be less than maximum grain size")

    return True



def compute_chi2_alma_cube(model_fits_path, data_cube_path, noise_std):
    """
    Compute chi² for ALMA interferometric data.

    Parameters
    ----------
    model_fits_path : str
        Path to the model fits file (ALMA/line data)
    data_cube_path : str
        Path to the observed data cube
    noise_std : float
        Standard deviation of the noise

    Returns
    -------
    float
        Chi² value
    """
    model_cube = fits.getdata(model_fits_path)
    data_cube = fits.getdata(data_cube_path)
    chi2 = np.sum(((data_cube - model_cube) / noise_std) ** 2)
    return chi2

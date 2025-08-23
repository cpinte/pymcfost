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


def compute_chi2_sed(model_sed_path, sed_data_path, i=0, iaz=0):
    """
    Compute chi² for SED fitting using MCFOST SED class.

    Parameters
    ----------
    model_sed_path : str
        Path to the model SED directory or fits file
    sed_data_path : str
        Path to the observed SED data file
    i : int, optional
        Inclination index for the model SED. Defaults to 0.
    iaz : int, optional
        Azimuth angle index for the model SED. Defaults to 0.

    Returns
    -------
    float
        Normalized chi² value

    Notes
    -----
    Expected SED data format (observed_sed.txt):
    wavelength_micron flux_jy flux_error_jy
    1.25 0.5 0.05
    2.2 1.2 0.12
    ...
    """
    try:
        # Load observed SED data
        if not os.path.exists(sed_data_path):
            print(f"Warning: SED data file {sed_data_path} not found. Skipping SED chi² calculation.")
            return 0.0

        obs_data = np.loadtxt(sed_data_path)
        obs_wl = obs_data[:, 0]  # wavelength in microns
        obs_flux = obs_data[:, 1]  # flux in Jy
        obs_error = obs_data[:, 2]  # flux error in Jy

        # Load model SED data using SED class
        if not os.path.exists(model_sed_path):
            print(f"Warning: Model SED path {model_sed_path} not found. Skipping SED chi² calculation.")
            return 0.0

        # Check if model_sed_path is a directory (MCFOST output) or a fits file
        if os.path.isdir(model_sed_path):
            # Load from MCFOST output directory
            sed_model = SED(model_sed_path)
            model_wl = sed_model.wl  # wavelength in microns
            model_flux = sed_model.sed[0, iaz, i, :]  # flux in W.m-2
            # Convert from W.m-2 to Jy
            model_flux_jy = model_flux * 1e26 / (3e14 / model_wl)  # Convert to Jy
        else:
            # Load from fits file directly
            try:
                hdu = fits.open(model_sed_path)
                if len(hdu) > 1:
                    model_wl = hdu[1].data  # wavelength extension
                    model_flux = hdu[0].data[0, iaz, i, :]  # flux extension
                else:
                    # Single extension - try to get wavelength from header
                    model_flux = hdu[0].data
                    n_wl = hdu[0].header.get('NWL', len(model_flux))
                    wl_min = hdu[0].header.get('WL_MIN', 0.1)
                    wl_max = hdu[0].header.get('WL_MAX', 1000.0)
                    model_wl = np.logspace(np.log10(wl_min), np.log10(wl_max), n_wl)
                hdu.close()
                # Convert from W.m-2 to Jy if needed
                model_flux_jy = model_flux * 1e26 / (3e14 / model_wl)
            except Exception as e:
                print(f"Error loading SED from fits file: {e}")
                return 0.0

        # Interpolate model to observed wavelengths
        # Handle cases where model wavelength range doesn't cover observed range
        wl_min = max(obs_wl.min(), model_wl.min())
        wl_max = min(obs_wl.max(), model_wl.max())

        # Filter data to common wavelength range
        mask_obs = (obs_wl >= wl_min) & (obs_wl <= wl_max)
        mask_model = (model_wl >= wl_min) & (model_wl <= wl_max)

        if np.sum(mask_obs) == 0 or np.sum(mask_model) == 0:
            print("Warning: No overlap in wavelength range between observed and model SEDs.")
            return 0.0

        obs_wl_filt = obs_wl[mask_obs]
        obs_flux_filt = obs_flux[mask_obs]
        obs_error_filt = obs_error[mask_obs]
        model_wl_filt = model_wl[mask_model]
        model_flux_filt = model_flux_jy[mask_model]

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


def compute_combined_chi2(model_fits_path, model_sed_path, data_cube_path, sed_data_path,
                         noise_std, use_alma=True, use_sed=True, i=0, iaz=0):
    """
    Compute combined chi² from both ALMA data and SED data.

    Parameters
    ----------
    model_fits_path : str or None
        Path to the model fits file (ALMA/line data)
    model_sed_path : str or None
        Path to the model SED directory or fits file
    data_cube_path : str
        Path to the observed ALMA data cube
    sed_data_path : str
        Path to the observed SED data file
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
    if use_sed and model_sed_path is not None:
        try:
            chi2_sed = compute_chi2_sed(model_sed_path, sed_data_path, i=i, iaz=iaz)
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

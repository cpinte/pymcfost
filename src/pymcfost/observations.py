import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import Jy_to_Wm2


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
            wavelength (ndarray or str, optional): Wavelength array in microns OR path to SED data file
            flux (ndarray, optional): Flux array in W/m²
            error (ndarray, optional): Flux error array in W/m²
            sed_data_path (str, optional): Path to observed SED data file

        Notes:
            If wavelength is a string, it's treated as a file path and data is loaded from file.
            Otherwise, provide (wavelength, flux, error) arrays OR sed_data_path.
            If sed_data_path is provided, the arrays will be loaded from the file.
        """
        # Check if first argument is a string (file path)
        if isinstance(wavelength, str) and flux is None and error is None and sed_data_path is None:
            # Treat wavelength as file path
            sed_data_path = wavelength
            wavelength = None

        if sed_data_path is not None:
            # Load from file
            loaded_data = self._read_from_file(sed_data_path)
            if loaded_data is None:
                raise ValueError(f"Failed to load SED data from {sed_data_path}")
            wavelength, flux, error = loaded_data

        if wavelength is None or flux is None or error is None:
            raise ValueError("Must provide either (wavelength, flux, error) arrays or file path")

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

            # Convert from Jy to W/m² using the utility function
            # ν = c/λ, where c = 3e14 μm/s
            nu = 3e14 / obs_wl  # frequency in Hz
            obs_flux_Wm2 = Jy_to_Wm2(obs_flux_jy, nu)
            obs_error_Wm2 = Jy_to_Wm2(obs_error_jy, nu)

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

    def plot(self, color="black", marker="o", markersize=4, capsize=3, 
             show_errorbars=True, label=None, **kwargs):
        """
        Plot the observed SED with error bars.

        Args:
            color (str): Color for the data points and error bars
            marker (str): Marker style for data points
            markersize (float): Size of the markers
            capsize (float): Size of error bar caps
            show_errorbars (bool): Whether to show error bars
            label (str, optional): Label for the legend
            **kwargs: Additional arguments passed to plt.errorbar()

        Returns:
            matplotlib.lines.Line2D: The plotted line object
        """
        if show_errorbars:
            line = plt.errorbar(
                self.wavelength, 
                self.flux, 
                yerr=self.error,
                color=color, 
                marker=marker, 
                markersize=markersize,
                capsize=capsize,
                linestyle='None',
                label=label,
                **kwargs
            )
        else:
            line = plt.plot(
                self.wavelength, 
                self.flux, 
                color=color, 
                marker=marker, 
                markersize=markersize,
                linestyle='None',
                label=label,
                **kwargs
            )[0]

        plt.xlabel("$\lambda$ [$\mu$m]")
        plt.ylabel("$\lambda$.F$_{\lambda}$ [W.m$^{-2}$]")
        plt.loglog()

        return line

    def __str__(self):
        """String representation of the ObservedSED object."""
        return f"ObservedSED with {self.n_points} points, wavelength range: {self.wavelength.min():.2f}-{self.wavelength.max():.2f} μm"

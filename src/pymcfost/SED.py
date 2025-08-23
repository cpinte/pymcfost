import os, warnings
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    import mpl_scatter_density
except ImportError:
    warnings.warn("mpl_scatter_density is not present", UserWarning)

from .parameters import Params, find_parameter_file
from .disc_structure import _plot_cutz, check_grid
from .utils import DustExtinction
from .run import run


class SED:
    """
    A class to handle MCFOST spectral energy distributions and temperature structures.

    This class reads and processes MCFOST SED outputs, providing methods to plot SEDs
    and temperature distributions.

    Attributes:
        dir (str): Directory containing MCFOST output files
        basedir (str): Base directory without data_th suffix
        P (Params): MCFOST parameter object
        sed (numpy.ndarray): SED data from ray-tracing
        wl (numpy.ndarray): Wavelength array in microns
        T (numpy.ndarray): Temperature structure

    Example:
        >>> sed = SED(dir="path/to/mcfost/data")
        >>> sed.plot(i=0)  # Plot SED for first inclination
    """

    _sed_th_file = ".sed_th.fits.gz"
    _sed_mc_file = "sed_mc.fits.gz"
    _sed_rt_file = "sed_rt.fits.gz"
    _temperature_file = "Temperature.fits.gz"
    _nlte_temperature_file = "Temperature_nLTE.fits.gz"

    def __init__(self, dir=None, **kwargs):
        """
        Initialize an SED object.

        Args:
            dir (str): Path to directory containing MCFOST output
            **kwargs: Additional arguments passed to _read()
        """
        # Correct path if needed
        dir = os.path.normpath(os.path.expanduser(dir))
        if dir[-7:] != "data_th":
            dir = os.path.join(dir, "data_th")
        self.dir = dir
        self.basedir = dir[:-8]

        # Search for parameter file
        para_file = find_parameter_file(dir)

        # Read parameter file
        self.P = Params(para_file)

        # Read model results
        self._read(**kwargs)


    def _read(self):
        # Read SED files
        try:
            hdu = fits.open(self.dir + "/" + self._sed_th_file)
            self._sed_th = hdu[0].data
            self._wl_th = hdu[1].data
            hdu.close()
        except OSError:
            print('cannot open', self._sed_th_file)

        try:
            hdu = fits.open(self.dir + "/" + self._sed_mc_file)
            self._sed_mc = hdu[0].data
            hdu.close()
        except OSError:
            print('cannot open', self._sed_mc_file)

        try:
            hdu = fits.open(self.dir + "/" + self._sed_rt_file)
            self.sed = hdu[0].data
            self.wl = hdu[1].data
            hdu.close()
        except OSError:
            print('cannot open', self._sed_rt_file)

        # Read temperature file
        try:
            hdu = fits.open(self.dir + "/" + self._temperature_file)
            self.T = hdu[0].data
            hdu.close()
        except OSError:
            print('Warning: cannot open', self._temperature_file)
            try:
                hdu = fits.open(self.dir + "/" + self._nlte_temperature_file)
                self.T = hdu[0].data
                hdu.close()
            except OSError:
                print('Warning: cannot open', self._temperature_file)


    def plot(self, i, iaz=0, MC=False, contrib=False, Av=0, Rv=3.1, color="black", **kwargs):
        """
        Plot the spectral energy distribution.

        Args:
            i (int): Inclination index
            iaz (int): Azimuth angle index
            MC (bool): Whether to plot Monte Carlo results
            contrib (bool): Whether to plot individual contributions
            Av (float): Extinction in V band
            Rv (float): Total to selective extinction ratio
            color (str): Line color
            **kwargs: Additional arguments passed to plotting function
        """

        # extinction
        if Av > 0:
            ext = DustExtinction(Rv=Rv)
            redenning = ext.redenning(self.wl, Av)
        else:
            redenning = 1.0

        if MC:
            sed = self._sed_mc[:, iaz, i, :] * redenning
        else:
            sed = self.sed[:, iaz, i, :] * redenning

        plt.loglog(self.wl, sed[0, :], color=color, **kwargs)
        plt.xlabel("$\lambda$ [$\mu$m]")
        plt.ylabel("$\lambda$.F$_{\lambda}$ [W.m$^{-2}$]")

        if contrib:
            n_type_flux = sed.shape[0]
            if n_type_flux in [5, 8, 9]:
                if n_type_flux > 5:
                    n_pola = 4
                else:
                    n_pola = 1
                linewidth = 0.5
                plt.loglog(
                    self.wl, sed[n_pola, :], linewidth=linewidth, color="magenta"
                )
                plt.loglog(
                    self.wl, sed[n_pola + 1, :], linewidth=linewidth, color="blue"
                )
                plt.loglog(
                    self.wl, sed[n_pola + 2, :], linewidth=linewidth, color="red"
                )
                plt.loglog(
                    self.wl, sed[n_pola + 3, :], linewidth=linewidth, color="green"
                )
            else:
                ValueError("There is no contribution data")

    def verif(self):
        """
        Compare SEDs from step 1 and 2 to verify energy conservation.
        """
        # Compares the SED from step 1 and step 2 to check if energy is properly conserved
        current_fig = plt.gcf().number
        plt.figure(20)
        plt.loglog(self._wl_th, np.sum(self._sed_th[0, :, :], axis=0), color="black")
        plt.loglog(self.wl, np.sum(self._sed_mc[0, 0, :, :], axis=0), color="red")
        plt.figure(current_fig)

    def plot_T(self, iaz=0, log=False, Tmin=None, Tmax=None):
        """
        Plot the temperature structure.

        Args:
            iaz (int): Azimuth angle index
            log (bool): Whether to use logarithmic scale
            Tmin (float, optional): Minimum temperature to plot
            Tmax (float, optional): Maximum temperature to plot
        """
        # For a cylindrical or spherical grid only at the moment
        # Todo:
        #  - automatically compute call mcfost to compute the grid
        # as required
        #  - add functions for radial and vertical cuts
        # We test if the grid structure already exist, if not we try to read it

        grid = check_grid(self)

        plt.cla()

        if grid.ndim > 2:
            Voronoi = False

            r = grid[0, iaz, :, :]
            z = grid[1, iaz, :, :]
            if self.T.ndim > 2:
                T = self.T[iaz, :, :]
            else:
                T = self.T[:, :]
        else:
            Voronoi = True
            T = self.T
            r = np.sqrt(grid[0, :] ** 2 + grid[1, :] ** 2)
            ou = r > 1e-6  # Removing star
            T = T[ou]
            r = r[ou]
            z = grid[2, ou]

        if Tmin is None:
            Tmin = T.min()
        if Tmax is None:
            Tmax = T.max()

        if log:
            if Voronoi:
                # plt.scatter(r,z/r,c=T,s=0.1, norm=mcolors.LogNorm(vmin=Tmin, vmax=Tmax))
                fig = plt.gcf()
                ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
                density = ax.scatter_density(
                    r,
                    z / r,
                    c=T,
                    cmap=plt.cm.RdYlBu_r,
                    dpi=100,
                    norm=mcolors.LogNorm(vmin=Tmin, vmax=Tmax),
                )
                fig.colorbar(density, label="T [K]")
            else:
                plt.pcolormesh(r, z / r, T, norm=mcolors.LogNorm(vmin=Tmin, vmax=Tmax))
                cb = plt.colorbar()
                cb.set_label('T [K]')
            plt.xscale('log')
            plt.xlabel("r [au]")
            plt.ylabel("z/r")
        else:
            if Voronoi:
                # plt.scatter(r,z,c=T,s=0.1, norm=mcolors.LogNorm(vmin=Tmin, vmax=Tmax))
                fig = plt.gcf()
                ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
                density = ax.scatter_density(
                    r,
                    z,
                    c=T,
                    cmap=plt.cm.RdYlBu_r,
                    dpi=100,
                    norm=mcolors.LogNorm(vmin=Tmin, vmax=Tmax),
                )
                fig.colorbar(density, label="T [K]")
            else:
                plt.pcolormesh(r, z, T, norm=mcolors.LogNorm(vmin=Tmin, vmax=Tmax))
                cb = plt.colorbar()
                cb.set_label('T [K]')
            plt.xlabel("r [au]")
            plt.ylabel("z [au]")


    def plot_Tz(self, r=100.0, dr=5.0, log=False, **kwargs):
        """
        Plot vertical temperature profile at specified radius.

        Args:
            r (float): Radius in AU
            dr (float): Radial range to average over in AU
            log (bool): Whether to use logarithmic scale
            **kwargs: Additional arguments passed to plotting function
        """

        _plot_cutz(self,self.T,r=r,dr=dr,log=log, **kwargs)
        plt.ylabel("T [K]")


    def plot_Tr(self, h_r=0.05, iaz=0, log=True, **kwargs):
        """
        Plot radial temperature profile at specified height.

        Args:
            h_r (float): Height above midplane in scale heights
            iaz (int): Azimuth angle index
            log (bool): Whether to use logarithmic scale
            **kwargs: Additional arguments passed to plotting function

        Returns:
            tuple: (radius array, temperature array)
        """

        grid = check_grid(self)

        if grid.ndim > 2:

            if self.T.ndim > 2:
                r_mcfost = grid[0, 0, self.P.grid.nz, :]
                T = self.T[iaz, self.P.grid.nz, :]
            else:
                r_mcfost = grid[0, 0, 0, :]
                T = self.T[0, :]

            if log:
                plt.loglog(r_mcfost, T, **kwargs)
            else:
                plt.plot(r_mcfost, T, **kwargs)
        else:
            r_mcfost = np.sqrt(grid[0, :] ** 2 + grid[1, :] ** 2)
            ou = r_mcfost > 1e-6  # Removing star
            T = self.T[ou]
            r_mcfost = r_mcfost[ou]
            z_mcfost = grid[2, ou]

            # Selecting data points
            ou = abs(z_mcfost) / r_mcfost < h_r

            r_mcfost = r_mcfost[ou]
            T = T[ou]

            fig = plt.gcf()
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            density = ax.scatter_density(r_mcfost,T, **kwargs)

        plt.xlabel("r [au]")
        plt.ylabel("T [K]")


        return r_mcfost, T

    def spectral_index(self, wl1, wl2, i=0, iaz=0):
        """
        Calculate spectral index between two wavelengths.

        Args:
            wl1 (float): First wavelength in microns
            wl2 (float): Second wavelength in microns
            i (int): Inclination index
            iaz (int): Azimuth angle index

        Returns:
            float: Spectral index
        """
        pass

    def convert_to_text(self, sed_text_path, i=0, iaz=0):
        """
        Convert MCFOST SED fits file to text format.
        The created file can be used as a sample file for chi² calculation.

        Parameters
        ----------
        sed_text_path : str
            Path where the text file will be saved
        i : int, optional
            Inclination index. Defaults to 0.
        iaz : int, optional
            Azimuth angle index. Defaults to 0.

        Returns
        -------
        str or None
            Path to the created text file, or None if conversion failed
        """
        try:
            # Use the already loaded SED data
            if not hasattr(self, 'wl') or not hasattr(self, 'sed'):
                raise ValueError("SED data not available. Make sure SED files were loaded successfully.")

            # Get the wavelength and flux data
            wavelength = self.wl
            flux = self.sed[0, iaz, i, :]  # Get flux for specific inclination and azimuth

            # Convert from W.m-2 to Jy
            flux_jy = flux * 1e26 / (3e14 / wavelength)

            # Write to text file
            with open(sed_text_path, 'w') as f:
                f.write("# Wavelength(micron) Flux(Jy)\n")
                for wl, fl in zip(wavelength, flux_jy):
                    if fl > 0 and np.isfinite(fl):  # Only write positive, finite values
                        f.write(f"{wl:.6e} {fl:.6e}\n")

            print(f"Converted SED to text: {sed_text_path}")
            return sed_text_path

        except Exception as e:
            print(f"Error converting SED to text: {e}")
            return None




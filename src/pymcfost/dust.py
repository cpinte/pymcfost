import os
import astropy.io.fits as fits
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

from .parameters import Params, find_parameter_file
from .utils import *

class Dust_model:
    """
    A class to handle MCFOST dust opacity models.

    This class reads and processes MCFOST dust opacity data, providing methods
    to plot and analyze various dust properties.

    Attributes:
        dir (str): Directory containing MCFOST output files
        P (Params): MCFOST parameter object
        wl (numpy.ndarray): Wavelength grid in microns
        kappa (numpy.ndarray): Total opacity in cm²/g
        albedo (numpy.ndarray): Dust albedo
        kappa_abs (numpy.ndarray): Absorption opacity in cm²/g
        kappa_sca (numpy.ndarray): Scattering opacity in cm²/g
        phase_function (numpy.ndarray): Scattering phase function
        polarisability (numpy.ndarray): Dust polarisability

    Example:
        >>> dust = Dust_model(dir="path/to/mcfost/data")
        >>> dust.plot_kappa()  # Plot opacity
    """

    def __init__(self, dir=None, **kwargs):
        """
        Initialize a Dust_model object.

        Args:
            dir (str): Path to directory containing MCFOST output
            **kwargs: Additional arguments passed to _read()
        """
        # Correct path if needed
        dir = os.path.normpath(os.path.expanduser(dir))
        self.dir = dir

        # Search for parameter file
        para_file = find_parameter_file(dir)

        # Read parameter file
        self.P = Params(para_file)

        # Read model results
        self._read(**kwargs)


    def _read(self):

        try:
            self.wl = fits.getdata(self.dir+"/lambda.fits.gz")
            self.kappa = fits.getdata(self.dir+"/kappa.fits.gz")
            self.albedo = fits.getdata(self.dir+"/albedo.fits.gz")

            self.kappa_abs = self.kappa * (1-self.albedo)
            self.kappa_sca = self.kappa * self.albedo

            self.phase_function = fits.getdata(self.dir+"/phase_function.fits.gz")
            try:
                self.polarisability = fits.getdata(self.dir+"/polarizability.fits.gz")
            except:
                print("Missing polarisability")


        except OSError:
            print('cannot open dust model in', self.dir)

    def plot_kappa(self, abs=True, scatt=True, linewidth=1, **kwargs):
        """
        Plot dust opacity coefficients.

        Args:
            abs (bool): Whether to plot absorption opacity
            scatt (bool): Whether to plot scattering opacity
            linewidth (float): Line width for plotting
            **kwargs: Additional arguments passed to plotting function
        """
        plt.loglog(self.wl,self.kappa,linewidth=linewidth, color="black")
        L = [r"$\kappa$"]

        if abs:
            plt.loglog(self.wl,self.kappa_abs,linewidth=linewidth, color="red")
            L.append(r"$\kappa_\mathrm{abs}$")
        if scatt:
            plt.loglog(self.wl,self.kappa_sca,linewidth=linewidth, color="blue")
            L.append("$\kappa_\mathrm{sca}$")

        plt.legend(L)

        plt.xlabel("$\lambda$ [$\mu$m]")
        plt.ylabel("$\kappa$ [cm$^2$/g]")

    def print_kappa(self, file="opacities.txt"):
        """
        Save opacity data to a text file.

        Args:
            file (str): Output filename
        """
        np.savetxt(file, np.array([self.wl,self.kappa_abs,self.kappa_sca]).T ,header="# wl [micron]  kappa_abs [cm2/g] kappa_abs [cm2/g]")

    def plot_albedo(self, linewidth=1, **kwargs):
        """
        Plot dust albedo as function of wavelength.

        Args:
            linewidth (float): Line width for plotting
            **kwargs: Additional arguments passed to plotting function
        """
        plt.semilogx(self.wl,self.albedo,linewidth=linewidth)
        plt.xlabel("$\lambda$ [$\mu$m]")
        plt.ylabel("albedo")

    def plot_phase_function(self, k=0, linewidth=1, **kwargs):
        """
        Plot scattering phase function.

        Args:
            k (int): Wavelength index
            linewidth (float): Line width for plotting
            **kwargs: Additional arguments passed to plotting function
        """
        theta = np.arange(0,181)
        plt.semilogy(theta,self.phase_function[:,k])

    def plot_polarisability(self, k=0, linewidth=1, **kwargs):
        """
        Plot dust polarisability.

        Args:
            k (int): Wavelength index
            linewidth (float): Line width for plotting
            **kwargs: Additional arguments passed to plotting function
        """
        theta = np.arange(0,181)
        plt.plot(theta,self.polarisability[:,k])

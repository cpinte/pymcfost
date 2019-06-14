import os
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

try:
    import mpl_scatter_density
except ImportError:
    print('WARNING: mpl_scatter_density is not present')

from .parameters import Params, find_parameter_file
from .disc_structure import Disc
from .utils import DustExtinction


class SED:

    _sed_th_file = ".sed_th.fits.gz"
    _sed_mc_file = "sed_mc.fits.gz"
    _sed_rt_file = "sed_rt.fits.gz"
    _temperature_file = "Temperature.fits.gz"

    def __init__(self, dir=None, **kwargs):
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
            print('cannot open', self._temperature_file)

    def plot(
        self, i, iaz=0, MC=False, contrib=False, Av=0, Rv=3.1, color="black", **kwargs
    ):

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
        # Compares the SED from step 1 and step 2 to check if energy is properly conserved
        current_fig = plt.gcf().number
        plt.figure(20)
        plt.loglog(self._wl_th, np.sum(self._sed_th[0, :, :], axis=0), color="black")
        plt.loglog(self.wl, np.sum(self._sed_mc[0, 0, :, :], axis=0), color="red")
        plt.figure(current_fig)

    def plot_T(self, iaz=0, log=False, Tmin=None, Tmax=None):
        # For a cylindrical or spherical grid only at the moment
        # Todo:
        #  - automatically compute call mcfost to compute the grid
        # as required
        #  - add functions for radial and vertical cuts
        # We test if the grid structure already exist, if not we try to read it

        try:
            grid = self.disc.grid
        except:
            try:
                print("Trying to read grid structure")
                self.disc = Disc(self.basedir)
                grid = self.disc.grid
            except AttributeError:
                print("Cannot read grid in " + self.basedir)

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
                # plt.scatter(r,z/r,c=T,s=0.1, norm=colors.LogNorm(vmin=Tmin, vmax=Tmax))
                fig = plt.gcf()
                ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
                density = ax.scatter_density(
                    r,
                    z / r,
                    c=T,
                    cmap=plt.cm.RdYlBu_r,
                    dpi=100,
                    norm=colors.LogNorm(vmin=Tmin, vmax=Tmax),
                )
                fig.colorbar(density, label="T [K]")
            else:
                plt.pcolormesh(r, z / r, T, norm=colors.LogNorm(vmin=Tmin, vmax=Tmax))
                cb = plt.colorbar()
                cb.set_label('T [K]')
            plt.xscale('log')
            plt.xlabel("r [au]")
            plt.ylabel("z/r")
        else:
            if Voronoi:
                # plt.scatter(r,z,c=T,s=0.1, norm=colors.LogNorm(vmin=Tmin, vmax=Tmax))
                fig = plt.gcf()
                ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
                density = ax.scatter_density(
                    r,
                    z,
                    c=T,
                    cmap=plt.cm.RdYlBu_r,
                    dpi=100,
                    norm=colors.LogNorm(vmin=Tmin, vmax=Tmax),
                )
                fig.colorbar(density, label="T [K]")
            else:
                plt.pcolormesh(r, z, T, norm=colors.LogNorm(vmin=Tmin, vmax=Tmax))
                cb = plt.colorbar()
                cb.set_label('T [K]')
            plt.xlabel("r [au]")
            plt.ylabel("z [au]")

    def plot_Tz(self, r=100.0, dr=5.0, log=False, **kwargs):

        try:
            grid = self.disc.grid
        except:
            try:
                print("Trying to read grid structure")
                self.disc = Disc(self.basedir)
                grid = self.disc.grid
            except AttributeError:
                print("Cannot read grid in " + self.basedir)

        T = self.T
        if grid.ndim > 2:
            Voronoi = False
            r_mcfost = grid[0, 0, :, :]
            z_mcfost = grid[1, 0, :, :]
        else:
            Voronoi = True
            r_mcfost = np.sqrt(grid[0, :] ** 2 + grid[1, :] ** 2)
            ou = r_mcfost > 1e-6  # Removing star
            T = T[ou]
            r_mcfost = r_mcfost[ou]
            z_mcfost = grid[2, ou]

        # Selecting data points
        ou = (r_mcfost > r - dr) & (r_mcfost < r + dr)

        # If we have cylindrical grid, we search for closest grid point
        # Todo

        z_mcfost = z_mcfost[ou]
        T = T[ou]

        plt.plot(z_mcfost, T, "o", **kwargs)
        plt.xlabel("z [au]")
        plt.ylabel("T [K]")

    def plot_Tr(self, h_r=0.05, log=True, symbol=None, **kwargs):

        try:
            grid = self.disc.grid
        except:
            try:
                print("Trying to read grid structure")
                self.disc = Disc(self.basedir)
                grid = self.disc.grid
            except AttributeError:
                print("Cannot read grid in " + self.basedir)

        T = self.T
        if grid.ndim > 2:
            Voronoi = False
            r_mcfost = grid[0, 0, :, :]
            z_mcfost = grid[1, 0, :, :]
        else:
            Voronoi = True
            r_mcfost = np.sqrt(grid[0, :] ** 2 + grid[1, :] ** 2)
            ou = r_mcfost > 1e-6  # Removing star
            T = T[ou]
            r_mcfost = r_mcfost[ou]
            z_mcfost = grid[2, ou]

        # Selecting data points
        ou = abs(z_mcfost) / r_mcfost < h_r

        # If we have cylindrical grid, we search for closest grid point
        # Todo

        r_mcfost = r_mcfost[ou]
        T = T[ou]

        if log:
            plt.loglog(r_mcfost, T, "o", linewidth=0.2, **kwargs)
        else:
            plt.plot(r_mcfost, T, "o", linewidth=0.2, **kwargs)
        plt.xlabel("z [au]")
        plt.ylabel("T [K]")

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

from parameters import McfostParams, find_parameter_file
from disc_structure import McfostDisc

class McfostSED:

    _sed_th_file = ".sed_th.fits.gz";
    _sed_mc_file = "sed_mc.fits.gz";
    _sed_rt_file = "sed_rt.fits.gz";
    _temperature_file = "Temperature.fits.gz";

    def __init__(self, dir=None, **kwargs):
        # Correct path if needed
        dir = os.path.normpath(os.path.expanduser(dir))
        if (dir[-7:] != "data_th"):
            dir = os.path.join(dir,"data_th")
        self.dir = dir
        self.basedir = dir[:-8]

        # Search for parameter file
        para_file = find_parameter_file(dir)

        # Read parameter file
        self.P = McfostParams(para_file)

        # Read model results
        self._read(**kwargs)


    def _read(self):
        # Read SED files
        try:
            hdu = fits.open(self.dir+"/"+self._sed_th_file)
            self._sed_th = hdu[0].data
            self._wl_th = hdu[1].data
            hdu.close()
        except OSError:
            print('cannot open', self._sed_th_file)

        try:
            hdu = fits.open(self.dir+"/"+self._sed_mc_file)
            self._sed_mc = hdu[0].data
            hdu.close()
        except OSError:
            print('cannot open', self._sed_mc_file)

        try:
            hdu = fits.open(self.dir+"/"+self._sed_rt_file)
            self.sed = hdu[0].data
            self.wl = hdu[1].data
            hdu.close()
        except OSError:
            print('cannot open', self._sed_rt_file)


        # Read temperature file
        try:
            hdu = fits.open(self.dir+"/"+self._temperature_file)
            self.T = hdu[0].data
            hdu.close()
        except OSError:
            print('cannot open', _temperature_file)


    def plot(self, i, MC=False, **kwargs):
        # todo :
        # - add extinction
        # - separate contribution

        if (MC):
            _sed = self._sed_mc[0,0,i,]
        else:
            _sed = self.sed[0,0,i,]

        plt.loglog(self.wl, _sed, **kwargs)
        plt.xlabel("$\lambda$ [$\mu$m]")
        plt.ylabel("$\lambda$.F$_{\lambda}$ [W.m$^{-2}$]")

    def verif(self):
        # Compares the SED from step 1 and step 2 to check if energy is properly conserved
        current_fig = plt.gcf().number
        plt.figure(20)
        plt.loglog(self._wl_th, np.sum(self._sed_th[0,:,:],axis=0), color="black")
        plt.loglog(self.wl, np.sum(self._sed_mc[0,0,:,:],axis=0), color="red")
        plt.figure(current_fig)


    def plot_T(self,log=False, Tmin=None, Tmax=None):
        # For a cylindrical or spherical grid only at the moment
        # Todo:
        #  - add Voronoi grid
        #  - automatically compute/read the grid as required
        #  - add functions for radial and vertical cuts

        # We test if the grid structure already exist, if not we try to read it
        try:
            grid = self.disc.grid
        except:
            try:
                print("Trying to read grid structure")
                self.disc = McfostDisc(self.basedir)
                grid = self.disc.grid
            except AttributeError:
                print("Cannot read grid in "+self.basedir)

        plt.clf()

        T = self.T

        if (grid.ndim > 2):
            Voronoi = False
            r = grid[0,0,:,:]
            z = grid[1,0,:,:]
        else:
            Voronoi = True
            r = np.sqrt(grid[0,:]**2 + grid[1,:]**2)
            ou = (r > 1e-6) # Removing star
            T = T[ou]
            r = r[ou]
            z = grid[2,ou]

        if (Tmin is None):
            Tmin = T.min()
        if (Tmax is None):
            Tmax = T.max()

        if (log):
            if (Voronoi):
                plt.scatter(r,z/r,c=T,s=0.1, norm=colors.LogNorm(vmin=Tmin, vmax=Tmax))
            else:
                plt.pcolormesh(r,z/r,T, norm=colors.LogNorm(vmin=Tmin, vmax=Tmax))
            plt.xscale('log')
            plt.xlabel("r [au]")
            plt.ylabel("z/r")
        else:
            if (Voronoi):
                plt.scatter(r,z,c=T,s=0.1, norm=colors.LogNorm(vmin=Tmin, vmax=Tmax))
            else:
                plt.pcolormesh(r,z,T, norm=colors.LogNorm(vmin=Tmin, vmax=Tmax))
            plt.xlabel("r [au]")
            plt.ylabel("z [au]")
        cb = plt.colorbar()
        cb.set_label('T [K]')

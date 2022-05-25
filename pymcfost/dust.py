import os
import astropy.io.fits as fits
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

from .parameters import Params, find_parameter_file
from .utils import *

class Dust_model:

    def __init__(self,dir=None,**kwargs):
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
            self.polarisability = fits.getdata(self.dir+"/polarizability.fits.gz")

        except OSError:
            print('cannot open dust model in', self.dir)

    def plot_kappa(self,abs=True,scatt=True, linewidth=1, **kwargs):

        plt.loglog(self.wl,self.kappa,linewidth=linewidth, color="black")
        if abs:
            plt.loglog(self.wl,self.kappa_abs,linewidth=linewidth, color="red")
        if scatt:
            plt.loglog(self.wl,self.kappa_sca,linewidth=linewidth, color="blue")

        plt.xlabel("$\lambda$ [$\mu$m]")
        plt.ylabel("$\kappa$ [cm$^2$/g]")

    def print_kappa(self,file="opacities.txt"):
        np.savetxt(file, np.array([self.wl,self.kappa_abs,self.kappa_sca]).T ,header="# wl [micron]  kappa_abs [cm2/g] kappa_abs [cm2/g]")

    def plot_albedo(self,linewidth=1, **kwargs):

        plt.semilogx(self.wl,self.albedo,linewidth=linewidth)
        plt.xlabel("$\lambda$ [$\mu$m]")
        plt.ylabel("albedo")

    def plot_phase_function(self,k=0,linewidth=1, **kwargs):
        theta = np.arange(0,181)
        plt.semilogy(theta,self.phase_function[:k])

    def plot_polarisability(self,k=0,linewidth=1, **kwargs):
        theta = np.arange(0,181)
        plt.plot(theta,self.polarisability[:,k])

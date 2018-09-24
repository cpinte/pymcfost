import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import os

from parameters import McfostParams, find_parameter_file


class McfostDisc:

    def __init__(self, dir=None, **kwargs):
        # Correct path if needed
        dir = os.path.normpath(os.path.expanduser(dir))
        if (dir[-9:] != "data_disk"):
            dir = os.path.join(dir,"data_disk")
        self.dir = dir

        # Search for parameter file
        para_file = find_parameter_file(dir)

        # Read parameter file
        self.P = McfostParams(para_file)

        # Read model results
        self._read(**kwargs)

    def _read(self):
        # Read grid file
        try:
            hdu = fits.open(self.dir+"/grid.fits.gz")
        except OSError:
            print('cannot open grid.fits.gz')
        self.grid = hdu[0].data
        hdu.close()

        # Read gas density file
        try:
            hdu = fits.open(self.dir+"/gas_density.fits.gz")
        except OSError:
            print('cannot open gas_density.fits.gz')
        self.gas_density = hdu[0].data
        hdu.close()

        # Read volume file
        try:
            hdu = fits.open(self.dir+"/volume.fits.gz")
        except OSError:
            print('cannot open volume.fits.gz')
        self.volume = hdu[0].data
        hdu.close()

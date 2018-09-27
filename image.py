import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

from parameters import McfostParams, find_parameter_file
from disc_structure import McfostDisc

class McfostImage:

    _RT_file = "RT.fits.gz";
    _MC_file = "MC.fits.gz";

    def __init__(self, dir=None, **kwargs):
        # Correct path if needed
        dir = os.path.normpath(os.path.expanduser(dir))
        self.dir = dir

        # Search for parameter file
        para_file = find_parameter_file(dir)

        # Read parameter file
        self.P = McfostParams(para_file)

        # Read model results
        self._read(**kwargs)

    def _read(self):
        # Read ray-traced image
        try:
            hdu = fits.open(self.dir+"/"+self._RT_file)
            self.image = hdu[0].data
            # Read a few keywords in header
            self.pixelscale =  hdu[0].header['CDELT2'] * 3600. # arcsec
            self.unit     =  hdu[0].header['BUNIT']
            self.wl       =  hdu[0].header['WAVE'] # micron

            hdu.close()
        except OSError:
            print('cannot open', self._RT_file)

    def plot(self,i=0,iaz=0,log=True,vmin=None,vmax=None,dynamic_range=1e6,fpeak=None,axes_unit='arcsec',colorbar=True):
        # Todo:
        #  - plot Q, U, P, PI, Qphi, Uphi
        #  - option for color_bar
        #  - superpose polarization vector
        #  - plot a selected contribution

        if vmax is None: vmax = self.image.max()
        if fpeak is not None : vmax = self.image.max() * fpeak
        if vmin is None: vmin= vmax/dynamic_range

        # Compute pixel scakle and extent of image
        if axes_unit.lower() == 'arcsec':
            pix_scale = self.pixelscale
            xlabel = '$\Delta$ Ra ["]'
            ylabel = '$\Delta$ Dec ["]'
        elif axes_unit.lower() == 'au':
            pix_scale = self.pixelscale * self.P.map.distance
            xlabel = '$\Delta$ x [au]'
            ylabel = '$\Delta$ y [au]'
        elif axes_unit.lower() == 'pixels' or axes_unit.lower() == 'pixel':
            pix_scale = 1
            xlabel = '$\Delta$ x [pix]'
            ylabel = '$\Delta$ y [pix]'
        else:
            raise ValueError("Unknown unit for axes_units: "+axes_unit)
        halfsize = np.asarray(self.image.shape[-2:])/2 * pix_scale
        extent = [-halfsize[0], halfsize[0], -halfsize[1], halfsize[1]]

        plt.clf()
        plt.imshow(self.image[0,i,iaz,:,:], norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True), extent=extent, origin='lower')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if (colorbar):
            cb = plt.colorbar()
            formatted_unit = self.unit.replace("-1","$^{-1}$").replace("-2","$^{-2}$")
            cb.set_label("Flux density ["+formatted_unit+"]")

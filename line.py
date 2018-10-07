import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import os

from parameters import McfostParams, find_parameter_file
from disc_structure import McfostDisc

class McfostLine:

    _line_file = "lines.fits.gz"

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
            hdu = fits.open(self.dir+"/"+self._line_file)
            self.lines = hdu[0].data
            # Read a few keywords in header
            self.pixelscale =  hdu[0].header['CDELT2'] * 3600. # arcsec
            self.unit     =  hdu[0].header['BUNIT']
            self.cx       =  hdu[0].header['CRPIX1']
            self.cy       =  hdu[0].header['CRPIX2']
            self.nx       =  hdu[0].header['NAXIS1']
            self.ny       =  hdu[0].header['NAXIS2']

            self.cont = hdu[1].data

            self.ifreq = hdu[2].data
            self.freq = hdu[3].data
            self.velocity = hdu[4].data

            hdu.close()
        except OSError:
            print('cannot open', self._line_file)

        def plot_map(self, i=0, iTrans=0, insert=False, substract_cont=False, moment=None, v=None,
                 psf_FWHM=None,bmaj=None,bmin=None,bpa=None,plot_beam=False):

            ax = plt.gca()

            #--- Compute pixel scale and extent of image
            if axes_unit.lower() == 'arcsec':
                pix_scale = self.pixelscale
                xlabel = '$\Delta$ Ra ["]'
                ylabel = '$\Delta$ Dec ["]'
            elif axes_unit.lower() == 'au':
                pix_scale = self.pixelscale * self.P.map.distance
                xlabel = 'Distance from star [au]'
                ylabel = 'Distance from star [au]'
            elif axes_unit.lower() == 'pixels' or axes_unit.lower() == 'pixel':
                pix_scale = 1
                xlabel = '$\Delta$ x [pix]'
                ylabel = '$\Delta$ y [pix]'
            else:
                raise ValueError("Unknown unit for axes_units: "+axes_unit)
            halfsize = np.asarray(self.image.shape[-2:])/2 * pix_scale
            extent = [-halfsize[0], halfsize[0], -halfsize[1], halfsize[1]]

            #-- beam or psf : psf_FWHM and bmaj and bmin are in arcsec, bpa in deg
            i_convolve = False

            if bmaj is not None:
                sigma_x = bmin / pix_scale * (2.*np.sqrt(2.*np.log(2))) # in pixels
                sigma_y = bmaj / pix_scale * (2.*np.sqrt(2.*np.log(2))) # in pixels
                beam = Gaussian2DKernel(sigma_x,sigma_y,bpa * np.pi/180)
                i_convolve = True

            #-- Selecting convolution function
            if conv is None:
                conv = convolve



        def plot_line(self, i=0, iTrans=0, psf_FWHM=None,bmaj=None,bmin=None,bpa=None,plot_beam=False):
            pass

        def moment(self, i=0, iTrans=0, psf_FWHM=None,bmaj=None,bmin=None,bpa=None,plot_beam=False):
            pass

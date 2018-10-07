import astropy.io.fits as fits
import astropy.constants as SI
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
            self.nv       =  hdu[0].header['NAXIS3']

            if (self.unit == "JY/PIXEL"):
                self.is_casa = True
                self.restfreq = hdu[0].header['RESTFREQ']
                self.velocity_type = hdu[0].header['CTYPE3']
                if (self.velocity_type == "VELO-LSR"):
                    self.CRPIX3 = hdu[0].header['CRPIX3']
                    self.CRVAL3 = hdu[0].header['CRVAL3']
                    self.CDELT3 = hdu[0].header['CDELT3']
                    self.velocity = self.CRVAL3 + self.CDELT3 * (np.arange(1,self.nv+1) - self.CRPIX3) # km/s
                    #self.velocity = (self.nu - self.restfreq)/SI.c.value


            else:
                self.is_casa = False
                self.cont = hdu[1].data

                self.ifreq = hdu[2].data
                self.freq = hdu[3].data # frequencies of the transition
                self.velocity = hdu[4].data / 1000 #km/s

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

    def plot_line(self, i=0, iaz=0, iTrans=0, psf_FWHM=None,bmaj=None,bmin=None,bpa=None,plot_beam=False,plot_cont=True):

        if (self.is_casa):
            line = np.sum(np.sum(self.lines[:,:,:], axis=2), axis=1)
            ylabel = "Flux [Jy]"
        else:
            line = np.sum(np.sum(self.lines[iaz,i,iTrans,:,:,:], axis=2), axis=1)
            ylabel = "Flux [W.m$^{-2}$]"

        plt.plot(self.velocity, line)

        if (plot_cont):
            if (self.is_casa):
                Fcont = 0.5 * (line[0]+line[-1]) # approx the continuum
            else:
                Fcont = np.sum(self.cont[iaz,i,iTrans,:,:])
            plt.plot([self.velocity[0],self.velocity[-1]],[Fcont,Fcont])

        xlabel = "v [m.s$^{-1}$]"

        plt.xlabel(xlabel) ; plt.ylabel(ylabel)


    def moment(self, i=0, iTrans=0, psf_FWHM=None,bmaj=None,bmin=None,bpa=None,plot_beam=False):
        pass

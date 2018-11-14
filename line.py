import astropy.io.fits as fits
import astropy.constants as SI
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

from matplotlib.patches import Ellipse
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft

from .parameters import Params, find_parameter_file
from .disc_structure import Disc
from .utils import FWHM_to_sigma, default_cmap

import progressbar

class Line:

    _line_file = "lines.fits.gz"

    def __init__(self, dir=None, **kwargs):

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
                self.restfreq = hdu[0].header['RESTFRQ']
                self.velocity_type = hdu[0].header['CTYPE3']
                if (self.velocity_type == "VELO-LSR"):
                    self.CRPIX3 = hdu[0].header['CRPIX3']
                    self.CRVAL3 = hdu[0].header['CRVAL3']
                    self.CDELT3 = hdu[0].header['CDELT3']
                    self.velocity = self.CRVAL3 + self.CDELT3 * (np.arange(1,self.nv+1) - self.CRPIX3) # km/s
                    #self.velocity = (self.nu - self.restfreq)/SI.c.value
                else:
                    raise ValueError("Velocity type is not recognised")
            else:
                self.is_casa = False
                self.cont = hdu[1].data

                self.ifreq = hdu[2].data
                self.freq = hdu[3].data # frequencies of the transition
                self.velocity = hdu[4].data / 1000 #km/s

            hdu.close()
        except OSError:
            print('cannot open', self._line_file)

    def plot_map(self,i=0,iaz=0,iTrans=0,iv=None,insert=False,substract_cont=False,moment=None,v=None,
                 psf_FWHM=None,bmaj=None,bmin=None,bpa=None,plot_beam=None,axes_unit="arcsec",conv_method=None,
                 fmax=None,fmin=None,fpeak=None,dynamic_range=1e3,color_scale=None,colorbar=True,cmap=None):
        # Todo:
        # - allow user to change brightness unit : W.m-1, Jy, Tb
        # - plot moment maps
        # - print velocity when plotting a channel map
        # - print molecular info (eg CO J=3-2)
        # - add continnum subtraction

        # bmin and bamj in arcsec

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
        halfsize = np.asarray(self.lines.shape[-2:])/2 * pix_scale
        extent = [-halfsize[0], halfsize[0], -halfsize[1], halfsize[1]]

        #-- set color map
        if (cmap is None):
            if (moment==1):
                cmap = "RdBu"
            else:
                cmap = default_cmap

        #-- beam or psf : psf_FWHM and bmaj and bmin are in arcsec, bpa in deg
        i_convolve = False

        beam = None
        if bmaj is not None:
            sigma_x = bmin / self.pixelscale * FWHM_to_sigma # in pixels
            sigma_y = bmaj / self.pixelscale * FWHM_to_sigma # in pixels
            beam = Gaussian2DKernel(sigma_x,sigma_y,bpa * np.pi/180)
            i_convolve = True
            if plot_beam is None:
                plot_beam = True

        #-- Selecting convolution function
        if conv_method is None:
            conv_method = convolve_fft

        #-- Selection of image to plot
        if (moment is not None):
            im = self.get_moment_map(i=i, iaz=iaz, iTrans=iTrans, moment=moment, beam=beam, conv_method=conv_method)
        else:
            # individual channel
            if (self.is_casa):
                im = self.lines[iv,:,:]
            else:
                im = self.lines[iaz,i,iTrans,iv,:,:]

            #-- Convolve image
            if i_convolve:
                im = conv_method(im,beam)
                if plot_beam is None:
                    plot_beam = True

        #--- Plot range and color map
        _color_scale = 'lin'
        if fmax is None: fmax = im.max()
        if fpeak is not None : fmax = im.max() * fpeak
        if fmin is None:
            fmin= im.min()
        if color_scale is None :
            color_scale = _color_scale
        if color_scale == 'log':
            if (fmin == 0.):
                fmin = fmax/dynamic_range
            norm = colors.LogNorm(min=fmin, vmax=fmax, clip=True)
        elif color_scale == 'lin':
            norm = colors.Normalize(vmin=fmin, vmax=fmax, clip=True)
        else:
            raise ValueError("Unknown color scale: "+color_scale)

        #-- Make the plot
        plt.clf()
        plt.imshow(im, norm = norm, extent=extent, origin='lower',cmap=cmap)
        plt.xlabel(xlabel) ; plt.ylabel(ylabel)

        #-- Color bar
        unit = self.unit
        if colorbar:
            cb = plt.colorbar()
            formatted_unit = unit.replace("-1","$^{-1}$").replace("-2","$^{-2}$")

            if moment==0:
                cb.set_label("Flux ["+formatted_unit+"km.s$^{-1}$]")
            elif moment==1:
                cb.set_label("Velocity [km.s$^{-1}]$")
            elif moment==2:
                cb.set_label("Velocity dispersion [km.s$^{-1}$]")
            else:
                cb.set_label("Flux ["+formatted_unit+"]")

        #-- Adding velocity
        if (moment is None):
            ax = plt.gca()
            dx = 0.5
            dy = 0.9
            plt.text(0.5,0.9,f"v={self.velocity[iv]:<4.2f}$\,$km/s",horizontalalignment='center',color="white",transform=ax.transAxes)


        #--- Adding beam
        if plot_beam:
            ax = plt.gca()
            dx = 0.125
            dy = 0.125
            beam = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                           width=bmin, height=bmaj, angle=-bpa,
                           fill=True,  color="grey")
            ax.add_patch(beam)


    def plot_line(self, i=0, iaz=0, iTrans=0, psf_FWHM=None,bmaj=None,bmin=None,bpa=None,plot_beam=False,plot_cont=True):

        if (self.is_casa):
            line = np.sum(self.lines[:,:,:], axis=(1,2))
            ylabel = "Flux [Jy]"
        else:
            line = np.sum(self.lines[iaz,i,iTrans,:,:,:], axis=(1,2))
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


    def get_moment_map(self, i=0, iaz=0, iTrans=0, moment=0, beam=None, conv_method=None):
        """
        This returns the moment maps in physical units, ie:
         - M1 is the average velocity [km/s]
         - M2 is the velocity dispersion [km/s]
        """
        if (self.is_casa):
            cube = np.copy(self.lines[:,:,:])
        else:
            cube = np.copy(self.lines[iaz,i,iTrans,:,:,:])

        dv = self.velocity[1] - self.velocity[0]

        if beam is None:
            M0 = np.sum(cube,axis=0) * dv
        else:
            if moment==0:
                M0 = np.sum(cube,axis=0) * dv
                M0 = conv_method(M0,beam)
            else: # We need to convolve each channel indidually
                print("Convolving individual channel maps, this may take a bit of time ....")
                bar = progressbar.ProgressBar(maxval=self.nv, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
                for iv in range(self.nv):
                    bar.update(iv+1)
                    channel = np.copy(cube[iv,:,:])
                    cube[iv,:,:] = conv_method(channel,beam)
                    M0 = np.sum(cube,axis=0) * dv
                bar.finish()

        if (moment >=1):
            M1 = np.sum(cube[:,:,:] * self.velocity[:,np.newaxis,np.newaxis], axis=0) * dv / M0

        if (moment == 2):
            M2 = np.sqrt( np.sum(cube[:,:,:] * (self.velocity[:,np.newaxis,np.newaxis] - M1[np.newaxis,:,:])**2, axis=0) * dv / M0)

        if moment==0:
            return M0
        elif moment==1:
            return M1
        elif moment==2:
            return M2

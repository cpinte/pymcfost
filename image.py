import os

import astropy.io.fits as fits
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

from .parameters import Params, find_parameter_file
from .disc_structure import Disc
from .utils import bin_image, FWHM_to_sigma

class Image:

    _RT_file = "RT.fits.gz"
    _MC_file = "MC.fits.gz"

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
            hdu = fits.open(self.dir+"/"+self._RT_file)
            self.image = hdu[0].data
            # Read a few keywords in header
            self.pixelscale = hdu[0].header['CDELT2'] * 3600. # arcsec
            self.unit       = hdu[0].header['BUNIT']
            self.wl         = hdu[0].header['WAVE'] # micron
            self.cx         = hdu[0].header['CRPIX1']
            self.cy         = hdu[0].header['CRPIX2']
            self.nx         = hdu[0].header['NAXIS1']
            self.ny         = hdu[0].header['NAXIS2']
            self.is_casa    = (self.unit == "JY/PIXEL")
            hdu.close()
        except OSError:
            print('cannot open', self._RT_file)

    def plot(self,i=0,iaz=0,vmin=None,vmax=None,dynamic_range=1e6,fpeak=None,axes_unit='arcsec',
             colorbar=True,type='I',scale=None,pola_vector=False,vector_color="white",nbin=5,
             psf_FWHM=None,bmaj=None,bmin=None,bpa=None,plot_beam=False,conv_method=None,
             mask=None):
        # Todo:
        #  - plot a selected contribution
        #  - add a mask on the star ?

        # bmin and bamj in arcsec

        ax = plt.gca()

        pola_needed = type in ['Q','U','Qphi','Uphi','P','PI','PA'] or pola_vector
        contrib_needed = type in ['star','scatt','em_th','scatt_em_th']

        if pola_needed and contrib_needed:
            raise ValueError('Cannot separate both polarisation and contributions')

        #--- We first check if the requested image is present in the mcfost fits file
        ntype_flux = self.image.shape[0]
        if ntype_flux not in (4,8): # there is no pola
            if pola_needed:
                raise ValueError('The model does not have polarisation data')
        elif ntype_flux not in (5,8): # there is no contribution
            if contrib_needed:
                raise ValueError('The model does not have contribution data')

        #--- Compute pixel scale and extent of image
        if axes_unit.lower() == 'arcsec':
            pix_scale = self.pixelscale
            xlabel = r'$\Delta$ Ra ["]'
            ylabel = r'$\Delta$ Dec ["]'
        elif axes_unit.lower() == 'au':
            pix_scale = self.pixelscale * self.P.map.distance
            xlabel = 'Distance from star [au]'
            ylabel = 'Distance from star [au]'
        elif axes_unit.lower() == 'pixels' or axes_unit.lower() == 'pixel':
            pix_scale = 1
            xlabel = r'$\Delta$ x [pix]'
            ylabel = r'$\Delta$ y [pix]'
        else:
            raise ValueError("Unknown unit for axes_units: "+axes_unit)
        halfsize = np.asarray(self.image.shape[-2:])/2 * pix_scale
        extent = [-halfsize[0], halfsize[0], -halfsize[1], halfsize[1]]

        #--- Beam or psf: psf_FWHM and bmaj and bmin are in arcsec, bpa in deg
        i_convolve = False
        if psf_FWHM is not None:
            sigma = psf_FWHM / self.pixelscale * (2.*np.sqrt(2.*np.log(2))) # in pixels
            beam = Gaussian2DKernel(sigma)
            i_convolve = True
            bmin = psf_FWHM
            bmaj = psf_FWHM
            bpa=0

        if bmaj is not None:
            sigma_x = bmin / self.pixelscale * FWHM_to_sigma # in pixels
            sigma_y = bmaj / self.pixelscale * FWHM_to_sigma # in pixels
            beam = Gaussian2DKernel(sigma_x,sigma_y,bpa * np.pi/180)
            i_convolve = True

        #--- Selecting convolution function
        if conv_method is None:
            conv_method = convolve

        #--- Intermediate images
        if pola_needed:
            I = self.image[0,i,iaz,:,:]
            Q = self.image[1,i,iaz,:,:]
            U = self.image[2,i,iaz,:,:]
            if i_convolve:
                Q = conv_method(Q,beam)
                U = conv_method(U,beam)
        elif contrib_needed:
            if pola_needed:
                n_pola=4
            else:
                n_pola=1
            if type == "star":
                I = self.image[n_pola,i,iaz,:,:]
            elif type == "scatt":
                I = self.image[n_pola+1,i,iaz,:,:]
            elif type == "em_th":
                I = self.image[n_pola+2,i,iaz,:,:]
            elif type == "scatt_em_th":
                I = self.image[n_pola+3,i,iaz,:,:]
        else:
            if self.is_casa:
                I = self.image[i,iaz,:,:]
            else:
                I = self.image[0,i,iaz,:,:]

        if i_convolve:
            I = conv_method(I,beam)

        #--- Selecting image to plot & convolution
        unit = self.unit
        flux_name = type
        if type == 'I':
            flux_name = 'Flux density'
            im = I
            _scale = 'log'
        elif type == 'Q':
            im = Q
            _scale = 'symlog'
        elif type == 'U':
            im = U
            _scale = 'symlog'
        elif type == 'P':
            I = I + (I == 0.)*1e-30
            im = 100 * np.sqrt((Q/I)**2 + (U/I)**2)
            unit = "%"
            _scale = 'lin'
        elif type == 'PI':
            im = np.sqrt(np.float64(Q)**2 + np.float64(U)**2)
            _scale = 'log'
        elif type in ('Qphi','Uphi'):
            X = np.arange(1,self.nx+1) - self.cx
            Y = np.arange(1,self.ny+1) - self.cy
            X, Y = np.meshgrid(X,Y)
            two_phi = 2 * np.arctan2(Y,X)
            if type == 'Qphi':
                im =  Q * np.cos(two_phi) + U * np.sin(two_phi)
            else: # Uphi
                im = -Q * np.sin(two_phi) + U * np.cos(two_phi)
            _scale = 'symlog'

        #--- Plot range and color map
        if vmax is None:
            vmax = im.max()
        if fpeak is not None:
            vmax = im.max() * fpeak
        if vmin is None:
            if (type in ["Q","U"]):
                vmin = -vmax
            else:
                vmin = im.min()
        if scale is None:
            scale = _scale
        if scale == 'symlog':
            norm = colors.SymLogNorm(1e-6*vmax,vmin=vmin, vmax=vmax, clip=True)
        elif scale == 'log':
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        elif scale == 'lin':
            norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        else:
            raise ValueError("Unknown color scale: "+scale)

        #--- Making the actual plot
        plt.clf()
        plt.imshow(im, norm=norm, extent=extent, origin='lower')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if colorbar:
            cb = plt.colorbar()
            formatted_unit = unit.replace("-1","$^{-1}$").replace("-2","$^{-2}$")
            cb.set_label(flux_name+" ["+formatted_unit+"]")

        #--- Overplotting polarisation vectors
        if pola_vector:
            X = (np.arange(1,self.nx+1) - self.cx) * pix_scale
            Y = (np.arange(1,self.ny+1) - self.cy) * pix_scale
            X, Y = np.meshgrid(X,Y)

            Xb = bin_image(X,nbin,func=np.mean)
            Yb = bin_image(Y,nbin,func=np.mean)
            Ib = bin_image(I,nbin)
            Qb = bin_image(Q,nbin)
            Ub = bin_image(U,nbin)

            pola = 100 * np.sqrt((Qb/Ib)**2 + (Ub/Ib)**2)
            theta = 0.5 * np.arctan2(Ub,Qb)
            pola_x = -pola * np.sin(theta) # Ref is N (vertical axis) --> sin, and Est is toward left --> -
            pola_y =  pola * np.cos(theta)

            plt.quiver(Xb, Yb, pola_x, pola_y, headwidth=0, headlength=0,
                       headaxislength=0.0, pivot='middle', color=vector_color)

        #--- Adding beam
        if plot_beam:
            ax = plt.gca()
            dx = 0.125
            dy = 0.125
            beam = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                           width=bmin, height=bmaj, angle=-bpa,
                           fill=True, color="grey")
            ax.add_patch(beam)

        #--- Adding mask
        if mask is not None:
            ax = plt.gca()
            dx = 0.5
            dy = 0.5
            mask = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                           width=2*mask, height=2*mask,
                           fill=True, color='grey')
            ax.add_patch(mask)

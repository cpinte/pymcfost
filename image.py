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
            self.cx       =  hdu[0].header['CRPIX1']
            self.cy       =  hdu[0].header['CRPIX2']
            self.nx       =  hdu[0].header['NAXIS1']
            self.ny       =  hdu[0].header['NAXIS2']
            hdu.close()
        except OSError:
            print('cannot open', self._RT_file)

    def plot(self,i=0,iaz=0,log=True,vmin=None,vmax=None,dynamic_range=1e6,fpeak=None,axes_unit='arcsec',colorbar=True,type='I',color_scale=None):
        # Todo:
        #  - plot Q, U, P, PI, Qphi, Uphi
        #  - option for color_bar
        #  - superpose polarization vector
        #  - plot a selected contribution
        #  - add convolution
        #  - add a mask on the star

        pola_needed = type in ['Q','U','Qphi','Uphi','P','PI','PA']
        contrib_needed = type in ['star','scatt','em_th','scatt_em_th']

        if pola_needed and contrib_needed:
            raise ValueError('Cannot separate both polarisation and contributions')

        # We first check if the requested image is present in the mcfost fits file
        ntype_flux = self.image.shape[0]
        if ntype_flux != 4 and ntype_flux != 8: # there is no pola
            if pola_needed:
                raise ValueError('The model does not have polarisation data')
        elif ntype_flux != 5 and ntype_flux != 8: # there is no contribution
            if contrib_needed:
                raise ValueError('The model does not have contribution data')

        #-- Intermediate images
        if pola_needed:
            I = self.image[0,i,iaz,:,:]
            Q = self.image[1,i,iaz,:,:]
            U = self.image[2,i,iaz,:,:]
        elif contrib_needed:
            # todo
            I = self.image[0,i,iaz,:,:]
        else:
            I = self.image[0,i,iaz,:,:]

        #-- Convolution

        #--- Selecting image to plot
        unit = self.unit
        flux_name = type
        if type == 'I':
            flux_name = 'Flux density'
            im = I
            _color_scale = 'log'
        elif type == 'Q':
            im = Q
            _color_scale = 'log'
        elif type == 'U':
            im = U
            _color_scale = 'log'
        elif type == 'P':
            I = I + (I == 0.)*1e-30
            im = 100 * np.sqrt((Q/I)**2 + (U/I)**2)
            unit = "%"
            _color_scale = 'lin'
        elif type == 'PI':
            im = np.sqrt(Q**2 + U**2)
            _color_scale = 'log'
        elif type == 'Qphi' or type == 'Uphi':
            X = np.arange(1,self.nx+1) - self.cx
            Y = np.arange(1,self.ny+1) - self.cy
            X, Y = np.meshgrid(X,Y)
            two_phi = 2 * np.arctan2(Y,X)
            if type == 'Qphi':
                im =  Q * np.cos(two_phi) + U * np.sin(two_phi)
            else: # Uphi
                im = -Q * np.sin(two_phi) + U * np.cos(two_phi)

        #--- Plot range and color map
        if vmax is None: vmax = im.max()
        if fpeak is not None : vmax = im.max() * fpeak
        if vmin is None:
            if (type in ["Q","U"]):
                vmin = -vmax
            else:
                vmin= vmax/dynamic_range

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

        if color_scale is None : color_scale = _color_scale
        if color_scale == 'log':
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        elif color_scale == 'lin':
            norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        else:
            raise ValueError("Unknown color scale: "+color_scale)

        #--- Making the actual plot
        plt.clf()
        plt.imshow(im, norm = norm, extent=extent, origin='lower')

        plt.xlabel(xlabel) ; plt.ylabel(ylabel)

        if (colorbar):
            cb = plt.colorbar()
            formatted_unit = unit.replace("-1","$^{-1}$").replace("-2","$^{-2}$")
            cb.set_label(flux_name+" ["+formatted_unit+"]")

        #--- Overplotting polarisation vectors
        if pola_vector:
            X = np.arange(1,self.nx+1) - self.cx
            Y = np.arange(1,self.ny+1) - self.cy
            X, Y = np.meshgrid(X,Y) * pix_scale

            pola = 100 * np.sqrt((Q/I)**2 + (U/I)**2)
            theta = 0.5 * np.arctan2(U,Q)
            pola_X = - pola * sin(theta) # Ref is N (vertical axis) --> sin,  and Est is toward left --> -
            pola_Y = pola * cos(theta)

            # Todo: write a bit of code to undersample

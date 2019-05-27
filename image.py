import copy
import os

import astropy.io.fits as fits
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.constants as sc

from .parameters import Params, find_parameter_file
from .utils import bin_image, FWHM_to_sigma, default_cmap, Wm2_to_Jy, Wm2_to_Tb, Jy_to_Tb

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
            self.freq       = sc.c/(self.wl*1e-6)
            self.cx         = hdu[0].header['CRPIX1']
            self.cy         = hdu[0].header['CRPIX2']
            self.nx         = hdu[0].header['NAXIS1']
            self.ny         = hdu[0].header['NAXIS2']
            self.is_casa    = (self.unit == "JY/PIXEL")
            hdu.close()
        except OSError:
            print('cannot open', self._RT_file)

    def plot(self,i=0,iaz=0,vmin=None,vmax=None,dynamic_range=1e6,fpeak=None,
             axes_unit='arcsec',colorbar=True,type='I',scale=None,
             pola_vector=False,vector_color="white",nbin=5,psf_FWHM=None,
             bmaj=None,bmin=None,bpa=None,plot_beam=None,conv_method=None,
             mask=None,cmap=None,ax=None,no_xlabel=False,no_ylabel=False,
             no_xticks=False,no_yticks=False,title=None,limit=None,limits=None,
             coronagraph=None,clear=False,Tb=False):
        # Todo:
        #  - plot a selected contribution
        #  - add a mask on the star ?

        # bmin and bamj in arcsec

        if clear:
            plt.clf()

        if ax is None:
            ax = plt.gca()
            ax.cla()

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
        beam = None
        if psf_FWHM is not None:
            sigma = psf_FWHM / self.pixelscale * (2.*np.sqrt(2.*np.log(2))) # in pixels
            beam = Gaussian2DKernel(sigma)
            i_convolve = True
            bmin = psf_FWHM
            bmaj = psf_FWHM
            bpa=0
            if plot_beam is None:
                plot_beam = True

        if bmaj is not None:
            sigma_x = bmin / self.pixelscale * FWHM_to_sigma # in pixels
            sigma_y = bmaj / self.pixelscale * FWHM_to_sigma # in pixels
            beam = Gaussian2DKernel(sigma_x,sigma_y,bpa * np.pi/180)
            i_convolve = True
            if plot_beam is None:
                plot_beam = True

        #--- Selecting convolution function
        if conv_method is None:
            conv_method = convolve_fft

        #--- Intermediate images
        if pola_needed:
            I = self.image[0,i,iaz,:,:]
            Q = self.image[1,i,iaz,:,:]
            U = self.image[2,i,iaz,:,:]
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

        #--- Convolve with beam
        if i_convolve:
            I = conv_method(I,beam)
            if pola_needed:
                Q = conv_method(Q,beam)
                U = conv_method(U,beam)

        #-- Conversion to brightness temperature
        if Tb:
            if self.is_casa:
                I = Jy_to_Tb(I, self.freq, self.pixelscale)
            else:
                I = Wm2_to_Tb(I, self.freq, self.pixelscale)
                I = np.nan_to_num(I)
                print("Max Tb=",np.max(I), "K")

        #--- Coronagraph: in mas
        if coronagraph is not None:
            halfsize = np.asarray(self.image.shape[-2:])/2
            posx = np.linspace(-halfsize[0],halfsize[0],self.nx)
            posy = np.linspace(-halfsize[1],halfsize[1],self.ny)
            meshx, meshy = np.meshgrid(posx,posy)
            radius_pixel = np.sqrt(meshx**2 + meshy**2)
            radius_mas = radius_pixel * pix_scale * 1000
            I[radius_mas < coronagraph] = 0.
            if pola_needed:
                Q[radius_mas < coronagraph] = 0.
                U[radius_mas < coronagraph] = 0.

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

        #--- Plot range and color scale
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
            if (vmin <= 0.0):
                vmin = 1e-6 * vmax
            print(vmin)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        elif scale == 'lin':
            norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        else:
            raise ValueError("Unknown color scale: "+scale)

        #--- Set color map
        if cmap is None:
            cmap = default_cmap
        try:
            cmap = copy.copy(cm.get_cmap(cmap))
        except:
            raise ValueError("Unknown colormap: "+cmap)
        try:
            cmap.set_bad(cmap.colors[0])
        except:
            try:
                cmap.set_bad(cmap(0.0))
            except:
                raise Warning("Can't set bad values from given colormap")

        #--- Making the actual plot
        img = ax.imshow(im, norm=norm, extent=extent, origin='lower',cmap=cmap)

        if limit is not None:
            limits = [-limit,limit,-limit,limit]

        if limits is not None:
            ax.set_xlim(limits[0],limits[1])
            ax.set_ylim(limits[2],limits[3])

        if not no_xlabel:
            ax.set_xlabel(xlabel)
        if not no_ylabel:
            ax.set_ylabel(ylabel)

        if no_xticks:
            ax.get_xaxis().set_visible(False)
        if no_yticks:
            ax.get_yaxis().set_visible(False)

        if title is not None:
            ax.set_title(title)

        #--- Colorbar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(img,cax=cax)
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
            dx = 0.125
            dy = 0.125
            beam = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                           width=bmin, height=bmaj, angle=bpa,
                           fill=True, color="grey")
            ax.add_patch(beam)

        #--- Adding mask
        if mask is not None:
            dx = 0.5
            dy = 0.5
            mask = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                           width=2*mask, height=2*mask,
                           fill=True, color='grey')
            ax.add_patch(mask)

        #--- Return
        return img

    def calc_vis(self, i=0, iaz=0, hor=True, Jy=False, klambda=False, Mlambda=False, color='black'):

        # smoothing, christophe suggest I may need this later
        #   ou = 1;
        #   if (!is_void(champ)) {
        #     size=model.P.map.ny;
        #     x=indgen(size)(,-:1:size) - (size/2+1);
        #     y=indgen(size)(-:1:size,) - (size/2+1);
        #     distance = abs(x,y);

        #     ou = (distance * pix_size < 0.5 * champ) ;

        #     if (gauss==1) {
        #       FWHM = champ / pix_size; // OK : le FWHM est equivalent a la largeur de la porte !!! Cool !
        #       sigma = FWHM / (2*sqrt(2*log(2))); // en sec
        #       ou = gauss_kernel(size, sigma) ;
        #     }
        #     // write, "Applying a field of view of ", champ, "as" ;

        #     if (champ > 0.5 * im_size) {
        #       write, "WARNING : image seems small to aply the filed of view accurately" ;
        #       write, im_size, champ ;
        #     }
        #   }

        # error message if klambda and Mlambda sccales are selected
        if klambda and Mlambda:
            raise Exception("Cannot plot visabilities on two different scales (k and M), set one to False")

        # Selecting image
        im=self.image[0,i,iaz,:,:]

        # padding the image for a smoother curve
        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 0)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value
            return vector
        im = np.pad(im, 1000, pad_with)

        # fft
        fim = np.real(np.fft.fft2(np.fft.fftshift(im)))

        # Baselines
        size = len(fim)
        center = size/2

        # converting from arcsecond to radian
        pix_size = self.pixelscale/3600. * np.pi/180.

        # pixel size in the uv plane
        pix_fft = 1.0/pix_size

        # pixel in wavelength (normalising in wavelength)
        pix=self.wl*1e-6*pix_fft

        baselines=np.linspace(0,int(pix/2),int(size/2))

        # visabilities
        if hor:
            vis = fim[0,0:int(size/2)]
        else:
            vis = fim[0:int(size/2),0]

        # convert to Jy
        if Jy:
            Wm2_to_Jy(vis,sc.c/self.wl)
            ylabel="Correlated flux [Jy]"
        else:
            ylabel="Correlated flux [W.m$^{-2}$.Hz$^{-1}$]"

        if klambda:
            baselines = baselines / (self.wl * 1e-3)
            xlabel = "Baselines [k$\lambda$]"
        elif Mlambda:
            baselines = baselines / (self.wl * 1e-6)
            xlabel = "Baselines [M$\lambda$]"
        else:
            xlabel = "Baselines [m]"

        plt.plot(baselines, vis, color=color)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        return baselines, vis, fim

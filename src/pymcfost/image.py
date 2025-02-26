import copy
import os

import astropy.io.fits as fits
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft, AiryDisk2DKernel
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.constants as sc

from .parameters import Params, find_parameter_file
from .utils import *

class Image:
    """
    A class to handle MCFOST images and calculate various properties.

    This class reads and processes MCFOST radiative transfer output images, 
    providing methods to plot them and analyse their properties.

    Attributes:
        dir (str): Directory containing MCFOST output files
        image (numpy.ndarray): The image data array
        P (Params): MCFOST parameter object
        pixelscale (float): Image pixel scale in arcsec
        unit (str): Unit of the image data
        wl (float): Wavelength in microns
        freq (float): Frequency in Hz
        nx (int): Number of pixels in x direction
        ny (int): Number of pixels in y direction
        is_casa (bool): Whether the image is in CASA format
        star_positions (numpy.ndarray): Array of stellar positions
        star_properties (numpy.ndarray): Array of stellar properties
        extent (list): Image extent for plotting [xmin, xmax, ymin, ymax]

    Example:
        >>> image = Image(dir="path/to/mcfost/data")
        >>> image.plot(i=0, iaz=0)
    """

    _RT_file = "RT.fits.gz"
    _MC_file = "MC.fits.gz"

    def __init__(self, dir=None, **kwargs):
        """
        Initialize an Image object.

        Args:
            dir (str): Path to directory containing MCFOST output
            **kwargs: Additional arguments passed to _read()
        """
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
            hdu = fits.open(self.dir + "/" + self._RT_file)
            self.image = hdu[0].data
            # Read a few keywords in header
            self.header = hdu[0].header
            self.pixelscale = hdu[0].header['CDELT2'] * 3600.0  # arcsec
            self.unit = hdu[0].header['BUNIT']
            self.wl = hdu[0].header['WAVE']  # micron
            self.freq = sc.c / (self.wl * 1e-6)
            self.cx = hdu[0].header['CRPIX1']
            self.cy = hdu[0].header['CRPIX2']
            self.nx = hdu[0].header['NAXIS1']
            self.ny = hdu[0].header['NAXIS2']
            self.is_casa = self.unit == "JY/PIXEL"
            try:
                self.star_positions = hdu[1].data
            except:
                self.star_positions = []
            try:
                self.star_vr = hdu[2].data
            except:
                self.star_vr = []
            try:
                self.star_properties = hdu[3].data
            except:
                self.star_properties = []
            hdu.close()
        except OSError:
            print('cannot open', self._RT_file)

    def plot(
        self,
        i=0,
        iaz=0,
        vmin=None,
        vmax=None,
        dynamic_range=1e6,
        fpeak=None,
        axes_unit='arcsec',
        colorbar=True,
        colorbar_size=10,
        type='I',
        scale=None,
        pola_vector=False,
        vector_color="white",
        nbin=5,
        psf_FWHM=None,
        bmaj=None,
        bmin=None,
        bpa=None,
        plot_beam=None,
        beam_position=(0.125, 0.125), # fraction of plot width and height
        conv_method=None,
        mask=None,
        cmap=None,
        ax=None,
        no_xlabel=False,
        no_ylabel=False,
        no_xticks=False,
        no_yticks=False,
        title=None,
        limit=None,
        limits=None,
        rescale_r2 = False,
        clear=False,
        Tb=False,
        telescope_diameter=None,
        Jy=False,
        mJy=False,
        MJy=False,
        muJy=False,
        per_arcsec2=False,
        per_str=False,
        per_beam=False,
        shift_dx=0,
        shift_dy=0,
        plot_stars=False,
        sink_particle_size=6,
        sink_particle_color="cyan",
        sink_particle_marker=None,
        norm=False,
        interpolation=None,
        beam_color='grey',
        mask_color='grey'
    ):
        """
        Plot the MCFOST image.

        Args:
            i (int): Inclination index
            iaz (int): Azimuth angle index
            vmin (float, optional): Minimum value for color scale
            vmax (float, optional): Maximum value for color scale
            dynamic_range (float): Ratio between maximum and minimum plotted values
            fpeak (float, optional): Fraction of peak value to use as maximum
            axes_unit (str): Unit for axes ('arcsec', 'au', or 'pixels')
            colorbar (bool): Whether to show colorbar
            colorbar_size (int): Size of colorbar text
            type (str): Type of image to plot ('I', 'Q', 'U', etc.)
            scale (str, optional): Color scale ('log', 'lin', or 'sqrt')
            pola_vector (bool): Whether to plot polarization vectors
            vector_color (str): Color of polarization vectors
            nbin (int): Binning factor for polarization vectors
            psf_FWHM (float, optional): FWHM of Gaussian PSF in arcsec
            bmaj (float, optional): Beam major axis FWHM in arcsec
            bmin (float, optional): Beam minor axis FWHM in arcsec
            bpa (float, optional): Beam position angle in degrees
            plot_beam (bool, optional): Whether to plot beam
            beam_position (tuple): Position of beam (x,y) as fraction of plot
            conv_method (callable, optional): Convolution method to use
            mask (float, optional): Radius of central mask in arcsec
            cmap (str, optional): Matplotlib colormap name
            ax (matplotlib.axes, optional): Axes to plot on
            no_xlabel (bool): Whether to hide x-axis label
            no_ylabel (bool): Whether to hide y-axis label
            no_xticks (bool): Whether to hide x-axis ticks
            no_yticks (bool): Whether to hide y-axis ticks
            title (str, optional): Plot title
            limit (float, optional): Limit for rescaling image
            limits (list, optional): Limits for rescaling image
            rescale_r2 (bool): Whether to rescale image by r^2
            clear (bool): Whether to clear existing plot
            Tb (bool): Whether to convert image to brightness temperature
            telescope_diameter (float, optional): Telescope diameter in meters
            Jy (bool): Whether to convert image to Jy
            mJy (bool): Whether to convert image to mJy
            MJy (bool): Whether to convert image to MJy
            muJy (bool): Whether to convert image to microJy
            per_arcsec2 (bool): Whether to convert image to flux per arcsec^2
            per_str (bool): Whether to convert image to flux per str
            per_beam (bool): Whether to convert image to flux per beam
            shift_dx (float): Shift in x direction for plotting
            shift_dy (float): Shift in y direction for plotting
            plot_stars (bool): Whether to plot star positions
            sink_particle_size (int): Size of sink particle marker
            sink_particle_color (str): Color of sink particle marker
            sink_particle_marker (str, optional): Marker style for sink particles
            norm (bool): Whether to normalize image
            interpolation (str, optional): Interpolation method for image
            beam_color (str): Color of beam
            mask_color (str): Color of mask

        Returns:
            matplotlib.image.AxesImage: The plotted image
        """
        # Todo:
        #  - plot a selected contribution
        #  - add a mask on the star ?

        # bmin and bamj in arcsec

        if clear:
            plt.clf()

        if ax is None:
            ax = plt.gca()
            ax.cla()

        pola_needed = type in ['Q', 'U', 'Qphi', 'Uphi', 'P', 'PI', 'PA'] or pola_vector
        contrib_needed = type in ['star', 'scatt', 'em_th', 'scatt_em_th']

        if pola_needed and contrib_needed:
            raise ValueError('Cannot separate both polarisation and contributions')

        # --- We first check if the requested image is present in the mcfost fits file
        ntype_flux = self.image.shape[0]
        if ntype_flux not in (4, 8):  # there is no pola
            if pola_needed:
                raise ValueError('The model does not have polarisation data')
            n_pola = 1
        else:
            n_pola = 4
        if ntype_flux not in (5, 8):  # there is no contribution
            if contrib_needed:
                raise ValueError('The model does not have contribution data')


        # --- Compute pixel scale and extent of image
        if axes_unit.lower() == 'arcsec':
            pix_scale = self.pixelscale
            xlabel = r'$\Delta$ RA (")'
            ylabel = r'$\Delta$ Dec (")'
            xaxis_factor = -1
        elif axes_unit.lower() == 'au':
            pix_scale = self.pixelscale * self.P.map.distance
            xlabel = 'Distance from star (au)'
            ylabel = 'Distance from star (au)'
            xaxis_factor = 1
        elif axes_unit.lower() == 'pixels' or axes_unit.lower() == 'pixel':
            pix_scale = 1
            xlabel = r'$\Delta$ x (pix)'
            ylabel = r'$\Delta$ y (pix)'
            xaxis_factor = 1
        else:
            raise ValueError("Unknown unit for axes_units: " + axes_unit)
        halfsize = np.asarray(self.image.shape[-2:]) / 2 * pix_scale
        extent = [-halfsize[0]*xaxis_factor-shift_dx, halfsize[0]*xaxis_factor-shift_dx, -halfsize[1]-shift_dy, halfsize[1]-shift_dy]
        self.extent = extent

        # --- Beam or psf: psf_FWHM and bmaj and bmin are in arcsec, bpa in deg
        i_convolve = False
        beam = None

        if telescope_diameter is not None:
            # sigma of Gaussian is ~ 0.42 lambda/D
            psf_FWHM = 0.42 * self.wl*1e-6 /telescope_diameter * sigma_to_FWHM / arcsec
            print("psf FWHM is ",psf_FWHM,'"')

        if psf_FWHM is not None:
            #print("test")
            # sigma in pixels
            sigma = psf_FWHM / (self.pixelscale * 2*np.sqrt(2*np.log(2)))
            beam = Gaussian2DKernel(sigma,x_size=int(15*sigma),y_size=int(15*sigma))
            i_convolve = True
            bmin = psf_FWHM
            bmaj = psf_FWHM
            bpa = 0
            if plot_beam is None:
                plot_beam = True

        if bmaj is not None:
            sigma_x = bmin / self.pixelscale * FWHM_to_sigma  # in pixels
            sigma_y = bmaj / self.pixelscale * FWHM_to_sigma  # in pixels
            beam = Gaussian2DKernel(sigma_x, sigma_y, bpa * np.pi / 180)
            i_convolve = True
            if plot_beam is None:
                plot_beam = True

        # --- Selecting convolution function
        if conv_method is None:
            conv_method = convolve_fft

        # --- Intermediate images
        if pola_needed:
            I = self.image[0, iaz, i, :, :]
            Q = self.image[1, iaz, i, :, :]
            U = self.image[2, iaz, i, :, :]
        elif contrib_needed:
            if type == "star":
                I = self.image[n_pola, iaz, i, :, :]
            elif type == "scatt":
                I = self.image[n_pola + 1, iaz, i, :, :]
            elif type == "em_th":
                I = self.image[n_pola + 2, iaz, i, :, :]
            elif type == "scatt_em_th":
                I = self.image[n_pola + 3, iaz, i, :, :]
        else:
            if self.is_casa:
                I = self.image[iaz, i, :, :]
            else:
                I = self.image[0, iaz, i, :, :]

        # --- Convolution with psf
        if i_convolve:
            I = conv_method(I, beam)
            if pola_needed:
                Q = conv_method(Q, beam)
                U = conv_method(U, beam)

        # -- Conversion to brightness temperature
        if Tb:
            if self.is_casa:
                I = Jy_to_Tb(I, self.freq, self.pixelscale)
            else:
                I = Wm2_to_Tb(I, self.freq, self.pixelscale)
                I = np.nan_to_num(I)
                print("Max Tb=", np.max(I), "K")


        # -- Conversion to Jy
        if Jy:
            if not self.is_casa:
                I = Wm2_to_Jy(I, self.freq)
                if pola_needed:
                    Q = Wm2_to_Jy(Q, self.freq)
                    U = Wm2_to_Jy(U, self.freq)

        # -- Conversion to mJy
        if mJy:
            if not self.is_casa:
                I = Wm2_to_Jy(I, self.freq) * 1e3
                if pola_needed:
                    Q = Wm2_to_Jy(Q, self.freq) * 1e3
                    U = Wm2_to_Jy(U, self.freq) * 1e3
            else:
                I *= 1e3

        # -- Conversion to microJy
        if muJy:
            if not self.is_casa:
                I = Wm2_to_Jy(I, self.freq) * 1e6
                if pola_needed:
                    Q = Wm2_to_Jy(Q, self.freq) * 1e6
                    U = Wm2_to_Jy(U, self.freq) * 1e6
            else:
                I *= 1e6

        # -- Conversion to MJy
        if MJy:
            if not self.is_casa:
                I = Wm2_to_Jy(I, self.freq) * 1e-6
                if pola_needed:
                    Q = Wm2_to_Jy(Q, self.freq) * 1e-6
                    U = Wm2_to_Jy(U, self.freq) * 1e-6
            else:
                I *= 1e-6

        # --- Selecting image to plot
        unit = self.unit
        flux_name = type
        if type in ('I','star','scatt','em_th','scatt_em_th'):
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
            I = I + (I == 0.0) * 1e-30
            im = 100 * np.sqrt((Q / I) ** 2 + (U / I) ** 2)
            unit = "%"
            _scale = 'lin'
        elif type == 'PI':
            im = np.sqrt(np.float64(Q) ** 2 + np.float64(U) ** 2)
            _scale = 'log'
        elif type == 'PA':
            im = (0.5 * np.arctan2(U, Q) + 4*np.pi)%2*np.pi
            _scale = 'lin'
        elif type in ('Qphi', 'Uphi'):
            X = np.arange(1, self.nx + 1) - self.cx
            Y = np.arange(1, self.ny + 1) - self.cy
            X, Y = np.meshgrid(X, Y)
            two_phi = 2 * np.arctan2(Y, X)
            if type == 'Qphi':
                im = Q * np.cos(two_phi) + U * np.sin(two_phi)
            else:  # Uphi
                im = -Q * np.sin(two_phi) + U * np.cos(two_phi)
            _scale = 'symlog'

         # --- Rescale image by r^2
        if rescale_r2:
            halfsize = np.asarray(self.image.shape[-2:]) / 2
            posx = np.linspace(-halfsize[0]+shift_dx/pix_scale,
                                halfsize[0]+shift_dx/pix_scale, self.nx)
            posy = np.linspace(-halfsize[1]-shift_dy/pix_scale,
                                halfsize[1]-shift_dy/pix_scale, self.ny)
            meshx, meshy = np.meshgrid(posx, posy)
            radius_pixel2 = meshx**2 + meshy**2
            im = im * radius_pixel2
            # Normalizing to 1, as units become meaningless
            # Computing max of plotted region and normalize to it
            if limit is not None:
                limits = [limit, -limit, -limit, limit]
            if limits is not None:
                limit_pix_xmin = int(halfsize[0]+(-shift_dx-limits[0])/pix_scale)
                limit_pix_xmax = int(halfsize[0]+(-shift_dx-limits[1])/pix_scale)
                limit_pix_ymin = int(halfsize[1]+(shift_dy+limits[2])/pix_scale)
                limit_pix_ymax = int(halfsize[1]+(shift_dy+limits[3])/pix_scale)
                im = im/np.max(I[limit_pix_xmin:limit_pix_xmax,
                                limit_pix_ymin:limit_pix_ymax])
            else:
                im = im/np.max(im) # Normalizing to 1 over entire image

        if Tb:
            flux_name = "Tb"
            _scale = "lin"
            unit = "K"
            # turning off some flags that may interfere
            per_arcsec2 = False
            per_beam = False

        if Jy:
            unit = "Jy.pixel-1"

        if mJy:
            unit = "mJy.pixel-1"

        if MJy:
            unit = "MJy.pixel-1"

        if rescale_r2:
            unit = "arbitrary units" # max == 1

        # -- Conversion to flux per arcsec2 or per beam
        if per_arcsec2:
            im = im / self.pixelscale**2
            unit = unit.replace("pixel-1", "arcsec-2")
            unit = unit.replace("/pixel", "/arcsec2")

        if per_str:
            im = im / self.pixelscale**2 * (180/np.pi * 3600)**2
            unit = unit.replace("pixel-1", "str-1")
            unit = unit.replace("/pixel", "/str")

        if per_beam:
            beam_area = bmin * bmaj * np.pi / (4.0 * np.log(2.0))
            pix_area = self.pixelscale**2
            im *= beam_area/pix_area
            unit = unit.replace("pixel-1", "beam-1")
            unit = unit.replace("/pixel", "/beam")

        if norm:
            im = im / np.max(im)
            flux_name = "Normalised flux"
            unit = ""

        # --- Plot range and color scale
        if vmax is None:
            vmax = im.max()
        if fpeak is not None:
            vmax = im.max() * fpeak
        if vmin is None:
            if type in ["Q", "U"]:
                vmin = -vmax
            else:
                vmin = 1e-3 * vmax

        if scale is None:
            scale = _scale
        if scale == 'symlog':
            norm = mcolors.SymLogNorm(1e-6 * vmax, vmin=vmin, vmax=vmax, clip=True)
        elif scale == 'log':
            if vmin <= 0.0:
                vmin = 1e-5 * vmax
            if vmin < 0.9e-5 * vmax:
                print("WARNING : vmin ~< 1e-6 vmax may crash with recent versions of matplotlib")
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        elif scale == 'lin':
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        elif scale == 'sqrt':
            norm = mcolors.PowerNorm(0.5, vmin=vmin, vmax=vmax, clip=True)
        else:
            raise ValueError("Unknown color scale: " + scale)

        # --- Set color map
        if cmap is None:
            cmap = default_cmap
        try:
            cmap = copy.copy(cm.get_cmap(cmap))
        except:
            raise ValueError("Unknown colormap: " + cmap)
        try:
            cmap.set_bad(cmap.colors[0])
        except:
            try:
                cmap.set_bad(cmap(0.0))
            except:
                raise Warning("Can't set bad values from given colormap")

        # --- Making the actual plot
        image = ax.imshow(im, norm=norm, extent=extent, origin='lower', cmap=cmap, interpolation=interpolation)

        if limit is not None:
            limits = [limit, -limit, -limit, limit]

        if limits is not None:
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])

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

        # --- Colorbar
        if colorbar:
            cb = add_colorbar(image)
            formatted_unit = unit.replace("-1", "$^{-1}$").replace("-2", "$^{-2}$")
            if unit != "":
                cb.set_label(flux_name + " (" + formatted_unit + ")",size=colorbar_size)
            else:
                cb.set_label(flux_name,size=colorbar_size)


        # --- Overplotting polarisation vectors
        if pola_vector:
            X = (np.arange(1, self.nx + 1) - self.cx) * pix_scale * xaxis_factor
            Y = (np.arange(1, self.ny + 1) - self.cy) * pix_scale
            X, Y = np.meshgrid(X, Y)

            Xb = bin_image(X, nbin, func=np.mean)
            Yb = bin_image(Y, nbin, func=np.mean)
            Ib = bin_image(I, nbin)
            Qb = bin_image(Q, nbin)
            Ub = bin_image(U, nbin)

            pola = 100 * np.sqrt((Qb / np.maximum(Ib,1e-300)) ** 2 + (Ub / np.maximum(Ib,1e-300)) ** 2)
            theta = 0.5 * np.arctan2(Ub, Qb)
            if xaxis_factor < 0:
                theta += np.pi/2 #RH - commenting this out to test fix for plotting vecotrs 21/2

            # Ref is N (vertical axis) --> sin, and Est is toward left --> -
            #pola_x = pola * np.sin(theta)
            #pola_y = pola * np.cos(theta)
            # RH - switching sin and cos to test fix for plotting vecotrs 21/2
            pola_x = pola * np.cos(theta)
            pola_y = pola * np.sin(theta)

            ax.quiver(
                Xb,
                Yb,
                pola_x,
                pola_y,
                headwidth=0,
                headlength=0,
                headaxislength=0.0,
                pivot='middle',
                color=vector_color,
            )

        # --- Adding beam
        if plot_beam:
            dx, dy = beam_position
            beam = Ellipse(
                ax.transLimits.inverted().transform((dx, dy)),
                width=bmin,
                height=bmaj,
                angle=-bpa,
                fill=True,
                color=beam_color,
            )
            ax.add_patch(beam)

        # --- Adding mask
        if mask is not None:
            dx = 0.5
            dy = 0.5
            mask = Ellipse(
                ax.transLimits.inverted().transform((dx, dy)),
                width=2 * mask,
                height=2 * mask,
                fill=True,
                color=mask_color,
            )
            ax.add_patch(mask)

        #-- Add stars
        if plot_stars:
            # star_position in arcsec by default
            factor =  pix_scale / self.pixelscale
            if isinstance(plot_stars,bool):
                x_stars = self.star_positions[0,iaz,i,:] * factor * (-xaxis_factor)
                y_stars = self.star_positions[1,iaz,i,:] * factor
            else: # int or list of int
                x_stars = self.star_positions[0,iaz,i,plot_stars] * factor * (-xaxis_factor)
                y_stars = self.star_positions[1,iaz,i,plot_stars] * factor
            ax.scatter(x_stars-shift_dx, y_stars-shift_dy,
                       color=sink_particle_color,s=sink_particle_size,marker=sink_particle_marker)

        #-- Saving the last plotted quantity
        self.last_image = im

        plt.sca(ax)

        # --- Return
        return image

    def calc_vis(
        self,
        i=0,
        iaz=0,
        hor=True,
        Jy=False,
        klambda=False,
        Mlambda=False,
        color='black',
    ):
        """
        Calculate visibility profiles from the image.

        Args:
            i (int): Inclination index
            iaz (int): Azimuth angle index  
            hor (bool): Calculate horizontal (True) or vertical (False) profile
            Jy (bool): Convert flux to Jansky
            klambda (bool): Use kilolambda units for baselines
            Mlambda (bool): Use megalambda units for baselines
            color (str): Color for plotting

        Returns:
            tuple: (baselines, visibilities, 2D FFT of image)
        """

        # error message if klambda and Mlambda sccales are selected
        if klambda and Mlambda:
            raise Exception(
                "Cannot plot visabilities on two different scales (k and M), set one to False"
            )

        # Selecting image
        im = self.image[0, iaz, i, :, :]

        # padding the image for a smoother curve
        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 0)
            vector[: pad_width[0]] = pad_value
            vector[-pad_width[1] :] = pad_value
            return vector

        im = np.pad(im, 1000, pad_with)

        # fft
        fim = np.real(np.fft.fft2(np.fft.fftshift(im)))

        # Baselines
        size = len(fim)
        center = size / 2

        # converting from arcsecond to radian
        pix_size = self.pixelscale / 3600.0 * np.pi / 180.0

        # pixel size in the uv plane
        pix_fft = 1.0 / pix_size

        # pixel in wavelength (normalising in wavelength)
        pix = self.wl * 1e-6 * pix_fft

        baselines = np.linspace(0, int(pix / 2), int(size / 2))

        # visabilities
        if hor:
            vis = fim[0, 0 : int(size / 2)]
        else:
            vis = fim[0 : int(size / 2), 0]

        # convert to Jy
        if Jy:
            Wm2_to_Jy(vis, sc.c / self.wl)
            ylabel = "Correlated flux (Jy)"
        else:
            ylabel = "Correlated flux (W.m$^{-2}$.Hz$^{-1}$)"

        if klambda:
            baselines = baselines / (self.wl * 1e-3)
            xlabel = "Baselines (k$\lambda$)"
        elif Mlambda:
            baselines = baselines / (self.wl * 1e-6)
            xlabel = "Baselines (M$\lambda$)"
        else:
            xlabel = "Baselines (m)"

        plt.plot(baselines, vis, color=color)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        return baselines, vis, fim


    def writeto(self, filename, **kwargs):
        """
        Write the last plotted image to a FITS file.

        Args:
            filename (str): Output filename
            **kwargs: Additional arguments passed to fits.writeto()
        """
        fits.writeto(os.path.normpath(os.path.expanduser(filename)),self.last_image, self.header, **kwargs)

    def get_planet_rPA(self,iplanet):
        """
        Get projected radius and position angle of a planet.

        Args:
            iplanet (int): Index of the planet

        Returns:
            tuple: (radius in arcsec, position angle in degrees)
        """
        "Return the projected radius (arcsec) and PA of planet #iplanet in the image"
        dx, dy = self.star_positions[:,0,0,iplanet] - self.star_positions[:,0,0,0]
        PA = np.rad2deg(np.arctan2(dy,-dx)) - 90
        r = np.hypot(dx,dy)

        return r, PA

def spectral_index(model1, model2, i=0, iaz=0):

    log_nuFnu1 = np.log(np.maximum(model1.image[0, iaz, i, :, :], 1e-300))
    log_nuFnu2 = np.log(np.maximum(model2.image[0, iaz, i, :, :], 1e-300))

    dlog_nu = np.log(model1.wl) - np.log(model2.wl)

    return (log_nuFnu2 - log_nuFnu1) / dlog_nu - 3

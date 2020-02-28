import os

import astropy.io.fits as fits
from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from . import  plots

try:
    import progressbar
except ImportError:
    print('WARNING: progressbar is not present')
from scipy import interpolate

from .parameters import Params, find_parameter_file
from .utils import FWHM_to_sigma, default_cmap, Wm2_to_Tb, Jy_to_Tb


class Map:
    
    # see Line plot_map for details on each parameter
    def __init__(self,
        i, iaz, iTrans, v, iv, insert, substract_cont, moment,
        psf_FWHM, bpa, plot_beam, axes_unit, conv_method,
        colorbar, cmap, ax, no_xlabel, no_ylabel, no_xticks,
        no_yticks, title, limit, limits, Delta_v, shift_dx, shift_dy, plot_stars, bmin, bmaj, line=None):
        
        self.i = i
        self.iaz = iaz
        self.iTrans = iTrans
        self.v = v
        self.iv = iv
        self.insert = insert
        self.substract_cont = substract_cont
        self.moment = moment
        self.psf_FWHM = psf_FWHM
        self.bpa = bpa
        self.plot_beam = plot_beam
        self.axes_unit = axes_unit
        self.conv_method = conv_method
        self.colorbar = colorbar
        self.cmap = cmap
        self.ax = ax
        self.no_xlabel = no_xlabel
        self.no_ylabel = no_ylabel
        self.no_xticks = no_xticks
        self.no_yticks = no_yticks
        self.title = title
        self.limit = limit
        self.limits = limits
        self.Delta_v = Delta_v
        self.shift_dx = shift_dx
        self.shift_dy = shift_dy
        self.plot_stars = plot_stars 
               
        self.im = None
        self.image = None
        self.moments = None
        
        # instance of Line class
        self.line = line
        self.norm = None
        
        self.bmin = bmin
        self.bmaj = bmaj


    def get_moment(self, moment):
        # have to recalculate each time if it is not M0
        if (moment == 0 and self.moment != 0) or self.moments[moment] is None:
            return self.line.get_moment_map(self, moment)
        else:
            return self.moments[moment]
    
    def create_colorbar(self, label):
        if self.colorbar:
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(self.image, cax=cax)
            cb.set_label(label)

    def create_cb_label(self):
        formatted_unit = self.line.unit.replace("-1", "$^{-1}$").replace("-2", "$^{-2}$")
        if self.Tb:
            return "T$_\mathrm{b}$ [K]"
        elif self.moment == 0:
            return "Flux [" + formatted_unit + "km.s$^{-1}$]"
        elif self.moment == 1:
            return "Velocity [km.s$^{-1}]$"
        elif self.moment == 2:
            return "Velocity dispersion [km.s$^{-1}$]"
        else:
            return "Flux [" + formatted_unit + "]"
    
            
    def add_beam_stars_vlabels(self, ax, bmin, bmaj, line):
        # -- Adding velocity
        have_label = False
        if self.moment is None and have_label:
            ax.text(
                0.5,
                0.1,
                f"$\Delta$v={line.velocity[self.iv]:<4.2f}$\,$km/s",
                horizontalalignment='center',
                color="white",
                transform=self.ax.transAxes,
            )

        # --- Adding beam
        if self.plot_beam:
            dx = 0.125
            dy = 0.125
            beam = Ellipse(
                ax.transLimits.inverted().transform((dx, dy)),
                width=bmin,
                height=bmaj,
                angle=self.bpa,
                fill=True,
                color="grey",
            )
            ax.add_patch(beam)

        #-- Add stars
        if self.plot_stars: # todo : different units
            self.show_stars(ax)

    def format_plot(self, ax, no_xlabel=None, no_ylabel=None, title=None):
        
        if no_xlabel is None:
            no_xlabel = self.no_xlabel
        if no_ylabel is None:
            no_ylabel = self.no_ylabel
        if title is None:
            title = self.title
    
        if self.limit is not None:
            self.limits = [-self.limit, self.limit, -self.limit, self.limit]

        if self.limits is not None:
            ax.set_xlim(self.limits[0], self.limits[1])
            ax.set_ylim(self.limits[2], self.limits[3])

        if not no_xlabel:
            ax.set_xlabel(self.xlabel)
        if not no_ylabel:
            ax.set_ylabel(self.ylabel)

        if self.no_xticks:
            ax.get_xaxis().set_visible(False)
        if self.no_yticks:
            ax.get_yaxis().set_visible(False)

        if title is not None:
            ax.set_title(title)

    def set_ax(self, ax):
        self.ax = ax
            
    def iv_labels_extent(self, line, axes_unit):

        # -- Selecting channel corresponding to a given velocity
        if self.v is not None:
            self.iv = np.abs(line.velocity - self.v).argmin()
            print("Selecting channel #", self.iv)
        else:
            self.iv = None
        # --- Compute pixel scale and extent of image
        if axes_unit.lower() == 'arcsec':
            pix_scale = line.pixelscale
            self.xlabel = r'$\Delta$ RA ["]'
            self.ylabel = r'$\Delta$ Dec ["]'
        elif axes_unit.lower() == 'au':
            pix_scale = line.pixelscale * line.P.map.distance
            self.xlabel = 'Distance from star [au]'
            self.ylabel = 'Distance from star [au]'
        elif axes_unit.lower() == 'pixels' or axes_unit.lower() == 'pixel':
            pix_scale = 1
            self.xlabel = r'$\Delta$ x [pix]'
            self.ylabel = r'$\Delta$ y [pix]'
        else:
            raise ValueError("Unknown unit for axes_units: " + axes_unit)
        halfsize = np.asarray(line.lines.shape[-2:]) / 2 * pix_scale
        self.extent = [halfsize[0] - self.shift_dx, -halfsize[0] - self.shift_dx, -halfsize[1] - self.shift_dy, halfsize[1] - self.shift_dy]
        
    
    def i_convolving(self, bmin, bmaj, line):
        # -- beam or psf : psf_FWHM and bmaj and bmin are in arcsec, bpa in deg
        i_convolve = False
        self.beam = None
        if self.psf_FWHM is not None:
            # in pixels
            sigma = self.psf_FWHM / line.pixelscale * FWHM_to_sigma
            self.beam = Gaussian2DKernel(sigma)
            i_convolve = True
            bmin = psf_FWHM
            bmaj = psf_FWHM
            self.bpa = 0
            if self.plot_beam is None:
                self.plot_beam = True

        if bmaj is not None:
            sigma_x = bmin / line.pixelscale * FWHM_to_sigma  # in pixels
            sigma_y = bmaj / line.pixelscale * FWHM_to_sigma  # in pixels
            self.beam = Gaussian2DKernel(sigma_x, sigma_y, self.bpa * np.pi / 180)
            i_convolve = True
            if self.plot_beam is None:
                self.plot_beam = True

        # -- Selecting convolution function
        if self.conv_method is None:
            self.conv_method = convolve_fft

        return i_convolve
            
    def create_channel_im(self, i_convolve, Tb, line):
        # individual channel
        if line.is_casa:
            cube = line.lines[:, :, :]
            # im = self.lines[iv+1,:,:])
        else:
            cube = line.lines[self.iaz, self.i, self.iTrans, :, :, :]
            # im = self.lines[i,iaz,iTrans,iv,:,:]

            # -- continuum substraction
            if self.substract_cont:
                cube = np.maximum(cube - line.cont[self.iaz, self.i, self.iTrans, np.newaxis, :, :], 0.0)

        # Convolve spectrally
        if self.Delta_v is not None:
            print("Spectral convolution at ", self.Delta_v, "km/s")
            # Creating a Hanning function with 101 points
            n_window = 101
            w = np.hanning(n_window)

            # For each pixel, resampling the spectrum between -FWHM to FWHM
            # then integrating over convolution window
            v_new = line.velocity[self.iv] + np.linspace(-1, 1, n_window) * self.Delta_v
            iv_min = int(self.iv - self.Delta_v / line.dv - 1)
            iv_max = int(self.iv + self.Delta_v / line.dv + 2)

            im = np.zeros([line.nx, line.ny])
            for j in range(line.ny):
                for i in range(line.nx):
                    f = interpolate.interp1d(
                        line.velocity[iv_min:iv_max], cube[iv_min:iv_max, i, j]
                    )
                    im[i, j] = np.average(f(v_new))
        else:
            im = cube[self.iv, :, :]
            
        # -- Convolve image
        if i_convolve:
            im = self.conv_method(im, self.beam)
            if self.plot_beam is None:
                self.plot_beam = True

        # -- Conversion to brightness temperature
        # if Tb:
#             if line.is_casa:
#                 im = Jy_to_Tb(im, line.freq[self.iTrans], line.pixelscale)
#             else:
#                 im = Wm2_to_Tb(im, line.freq[self.iTrans], line.pixelscale)
#                 im = np.nan_to_num(im)
#             print("Max Tb=", np.max(im), "K")

        self.im = im
            
    def create_color_scale(self, fmin, fmax, fpeak, color_scale, dynamic_range):

        if fmax is None:
            fmax = self.im.max()
            print("max: ", fmax)
        if fpeak is not None:
            fmax = self.im.max() * fpeak
        if fmin is None:
            fmin = self.im.min()
            print("min: ", fmin)

        if color_scale is None:
            color_scale = 'lin'
        if color_scale == 'log':
            if fmin <= 0.0:
                fmin = fmax / dynamic_range
            self.norm = colors.LogNorm(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'lin':
            self.norm = colors.Normalize(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'sqrt':
            self.norm = colors.PowerNorm(gamma=1./2., vmin=fmin, vmax=fmax, clip=True)
        else:
            raise ValueError("Unknown color scale: " + color_scale)
            
            
    def show_stars(self, ax, which="all"):
        # add being able to choose colour and style
        
        if which not in ["primary", "companion", "binary", "all"]:
            raise ValueError("invalid stars requested. Must be one of primary, secondary, or all")
    
        if self.moment is None or self.moment != 1:
                color = "cyan"
        else:
            color = "black"
            
        x_star = self.line.star_positions[0, self.iaz, self.i, :]
        y_stars = self.line.star_positions[1, self.iaz, self.i, :]      

        if len(x_star) == 2:
            which = "binary"
        elif which == "all":
            ax.scatter(x_star, y_stars, color=color, s=10, edgecolors='black', linewidth=0.2, marker='o')  
            return 

        if which in ["primary", "binary"]:
            ax.scatter(x_star[0], y_stars[0], color=color, s=20, edgecolors='black', linewidth=0.2, marker='*')
            
        if which in ["companion", "binary"]:
            ax.scatter(x_star[1], y_stars[1], color=color, s=10, edgecolors='black', linewidth=0.2, marker='o')


    def create_plot(self, ax, cmap):
        # -- Make the plot
        ax.cla()
        self.image = ax.imshow(self.im, norm=self.norm, extent=self.extent, origin='lower', cmap=cmap)
        ax.tick_params(direction='in', length=6, width=1, color='w', top=True, right=True, labelsize=14)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-0.5, 0, 0.5])

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
            hdu = fits.open(self.dir + "/" + self._line_file)
            self.lines = hdu[0].data
            # Read a few keywords in header
            self.pixelscale = hdu[0].header['CDELT2'] * 3600.0  # arcsec
            self.unit = hdu[0].header['BUNIT']
            self.cx = hdu[0].header['CRPIX1']
            self.cy = hdu[0].header['CRPIX2']
            self.nx = hdu[0].header['NAXIS1']
            self.ny = hdu[0].header['NAXIS2']
            self.nv = hdu[0].header['NAXIS3']

            if self.unit == "JY/PIXEL":
                self.is_casa = True
                self.restfreq = hdu[0].header['RESTFREQ']
                self.freq = [self.restfreq]
                self.velocity_type = hdu[0].header['CTYPE3']
                if self.velocity_type == "VELO-LSR":
                    self.CRPIX3 = hdu[0].header['CRPIX3']
                    self.CRVAL3 = hdu[0].header['CRVAL3']
                    self.CDELT3 = hdu[0].header['CDELT3']
                    # velocity in km/s
                    self.velocity = self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)
                else:
                    raise ValueError("Velocity type is not recognised")

                try:
                    self.star_positions = hdu[1].data
                except:
                    self.star_positions = []
            else:
                self.is_casa = False
                self.cont = hdu[1].data

                self.ifreq = hdu[2].data
                self.freq = hdu[3].data  # frequencies of the transition
                self.velocity = hdu[4].data / 1000  # km/s
                try:
                    self.star_positions = hdu[5].data
                except:
                    self.star_positions = []

            self.dv = self.velocity[1] - self.velocity[0]

            hdu.close()
        except OSError:
            print('cannot open', self._line_file)

    def plot_map(
        self,
        i=0,                        # (int) inclination index
        iaz=0,                      # (int) azimuth index 
        iTrans=0,                   # ?
        v=None,                     # (float) Velocity channel plotted (km/s)
        iv=None,                    # ?
        insert=False,               # ?
        substract_cont=False,       # (bool) subtract continuum from data
        moment=None,                # (int) moment to be plotted
        psf_FWHM=None,              # ?
        bmaj=None,                  # (float) size of beam's major axis (arcseconds)
        bmin=None,                  # (float) size of beam's minor axis  (arcseconds)
        bpa=None,                   # (float) beam's position angle (degrees)
        plot_beam=None,             # (bool) a representation of the beam size will be plotted
        axes_unit="arcsec",         # (str) unit for axes
        conv_method=None,           # (?) method of convolving beam
        fmax=None,                  # (float) maximum value to be shown on plot
        fmin=None,                  # (float) minimum value to be shown on plot
        fpeak=None,                 # (float) ?
        dynamic_range=1e3,          # (float) ?
        color_scale=None,           # (str) Scale used when plotting data e.g. one of 'lin', 'log', 'sqrt'
        colorbar=True,              # (bool) is a colorbar displayed next to the plot?
        cmap=None,                  # (str) color scheme for colormap
        ax=None,                    # (plt.axis) axis the map will be plotted on
        no_xlabel=False,            # (bool) no label for the x-axis of the plot
        no_ylabel=False,            # (bool) no label for the y-axis of the plot
        no_xticks=False,            # (bool) no ticks along the x-axis of the plot
        no_yticks=False,            # (bool) no ticks along the y-axis of the plot
        title=None,                 # (str) set title of the plot
        limit=None,                 # (int) ?
        limits=None,                # (List<int>) ?
        Tb=False,                   # (bool) Use a temperature scale
        Delta_v=None,               # (float) Uncertainty spread of velocity (km/s)
        shift_dx=0,                 # ?
        shift_dy=0,                 # ?
        plot_stars=False,           # (bool) plot all stars, or no stars
        subtractor=None             # (Map) pre-calculated map, which is used to take away from this instance
                                    #       e.g. if we wanted to create a M1 residual plot 
    ):
        
        # Puts properties into a wrapper class for later use
        map = Map(i, iaz, iTrans, v, iv, insert, substract_cont, moment,
        psf_FWHM, bpa, plot_beam, axes_unit, conv_method,
        colorbar, cmap, ax, no_xlabel, no_ylabel, no_xticks,
        no_yticks, title, limit, limits, Delta_v, shift_dx, shift_dy, plot_stars, bmin, bmaj, line=self)
            
        self.subtractor = subtractor
        map.Tb = Tb

        if ax is None:
            ax = plt.gca()
        
        map.iv_labels_extent(self, axes_unit)
        # -- set color map
        if cmap is None:
            if moment in [1,9]:
                cmap = "RdBu_r"
            else:
                cmap = default_cmap
                
        map.cmap = cmap

        i_convolve = map.i_convolving(bmin, bmaj, self)
        

        # -- Selection of image to plot
        if moment is not None:            
            map.im = self.get_moment_map(map, moment)
        else:
            map.create_channel_im(i_convolve, Tb, self)
        
        if subtractor is not None:
            if subtractor.moment != moment:
                raise ValueError("Subtractor map does not have the correct moment")
            map.im -= subtractor.im
                    
        im = map.im
        if Tb:
            if self.is_casa:
                im = Jy_to_Tb(im, self.freq[map.iTrans], self.pixelscale)
            else:
                im = Wm2_to_Tb(im, self.freq[map.iTrans], self.pixelscale)
                im = np.nan_to_num(im)
            print("Max Tb=", np.max(im), "K")
            print("Min Tb=", np.min(im), "K")
        map.im = im

        map.create_color_scale(fmin, fmax, fpeak, color_scale, dynamic_range)
        map.create_plot(map.ax, cmap)
        map.format_plot(map.ax)
        map.add_beam_stars_vlabels(map.ax, bmin, bmaj, self)

        if colorbar:
            map.create_colorbar(map.create_cb_label())
        
        return map

    def plot_line(
        self,
        i=0,
        iaz=0,
        iTrans=0,
        psf_FWHM=None,
        bmaj=None,
        bmin=None,
        bpa=None,
        plot_beam=False,
        plot_cont=True,
    ):

        if self.is_casa:
            line = np.sum(self.lines[:, :, :], axis=(1, 2))
            ylabel = "Flux [Jy]"
        else:
            line = np.sum(self.lines[iaz, i, iTrans, :, :, :], axis=(1, 2))
            ylabel = "Flux [W.m$^{-2}$]"

        plt.plot(self.velocity, line)

        if plot_cont:
            if self.is_casa:
                Fcont = 0.5 * (line[0] + line[-1])  # approx the continuum
            else:
                Fcont = np.sum(self.cont[iaz, i, iTrans, :, :])
            plt.plot([self.velocity[0], self.velocity[-1]], [Fcont, Fcont])

        xlabel = "v [m.s$^{-1}$]"

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


    def convolve_channels(self, cube, beam, conv_method):
        print("Convolving individual channel maps, this may take a bit of time ....")
        try:
            bar = progressbar.ProgressBar(
                maxval = self.nv,
                widgets = [
                    progressbar.Bar('=', '[', ']'),' ', progressbar.Percentage(),
                ], )
            bar.start()
        except:
            pass
        for iv in range(self.nv):
            try:
                bar.update(iv + 1)
            except:
                pass
            cube[iv, :, :] = conv_method(np.copy(cube[iv, :, :]), beam)
            M0 = np.sum(cube, axis=0) * self.dv
        try:
            bar.finish()
        except:
            pass
        return M0
    
    
    def calc_M0(self, moments, beam, cube, conv_method, original_moment):
        """ Calcualtes moment 0 """
        if beam is None:
            return np.sum(cube, axis=0) * self.dv
        else:
            if original_moment == 0:
                return conv_method(np.sum(cube, axis=0) * self.dv, beam)
            else:                   
                return self.convolve_channels(cube, beam, conv_method)
    
    def calc_moment(self, moment, beam, cube, conv_method, moments, original_moment):
        """ Calculates and return an indiviudal moment. Assumes moments needed for each one have already been calculated"""
    
        if moment == 0:
            return self.calc_M0(moments, beam, cube, conv_method, original_moment)
            
        elif moment == 1:
            return np.sum(cube[:, :, :] * self.velocity[:, np.newaxis, np.newaxis], axis=0) * self.dv / moments[0]
            
        elif moment == 2:
            return np.sqrt(np.sum(cube[:, :, :]
                    * (self.velocity[:, np.newaxis, np.newaxis] - moments[1][np.newaxis, :, :]) ** 2,
                    axis=0,) * self.dv / moments[0])
        
        
    
    def calc_moments(self, moment, beam, cube, conv_method, moments):
        """ Calculates a set of mathematical moments and assigns them to the appropriate index of the moments list"""
        if moment == 9:
                return self.velocity[(np.argmax(cube, axis=0))]
                
        for i in range(moment + 1):
            moments[i] = self.calc_moment(i, beam, cube, conv_method, moments, moment)

    def get_moment_map(self, map, moment):
        """
        Calculates and returns a moment for a given map, with a predetermined set of properties.
        Moments calculated are saved within the map for later use.
    
        Parameters:
            map (Map): Container for the properties needed to calculate the moment map
            moment (int): moment which will be calculated
    
        Returns : The moment maps in physical units, ie:
                - M1 is the average velocity [km/s]
                - M2 is the velocity dispersion [km/s]
         
        """
        i = map.i
        iaz = map.iaz
        iTrans = map.iTrans
        beam = map.beam
        conv_method = map.conv_method
        substract_cont = map.substract_cont
        
        # currently does not account for moments greater than M2
        if map.moments is None:
            map.moments = [None] * 3

        self.dv = self.velocity[1] - self.velocity[0]
        
        if self.is_casa:
            cube = np.copy(self.lines[:, :, :])
        else:
            cube = np.copy(self.lines[iaz, i, iTrans, :, :, :])
        if substract_cont:
            cube = np.maximum(cube - self.cont[iaz, i, iTrans, np.newaxis, :, :], 0.0)
        
        self.calc_moments(moment, beam, cube, conv_method, map.moments)

        return map.moments[moment]
                   
                   
def plot_contours(map, moment, levels=4, ax=None, specific_values=[], colors='black', linewidths=0.25):    
    """
    Overplot contours of a given map and moment onto an axis.
    
    Parameters:
        map (Map): map of data to be plotted
        moment (int): The moment that will be the source of the contour lines
        levels (int): How many contour levels will be plotted. This is overridden if specific values are specified
        ax (plt.axis): The pyplot axis on which the contours will be plotted on. This allows for contours to be placed over an existing plot (e.g. moment 1 contours over an already plotted moment 0)
                        Defaults to map.ax if no other is specified
        specific_values (List<int>): Specific contour levels that can be specified to be plotted. Must be in ascending order
        colors: (str or List<str>): Colors for the contour lines/levels
        linewidths= (int or List<int>): Line thickness of contour lines/levels
        
    Example:
        If we wanted to plot v=0, v=-1.5 and v=+1.5 of a map onto an axis...
        plot_contours(map, 1, specific_values=[-1.5, 0, 1.5])
    """
    im = map.get_moment(moment)
    if ax is None:
        ax = map.ax
    
    if len(specific_values) != 0:
        levels = specific_values
    
    ax.contour(im, extent=map.extent, origin='lower', levels=levels, colors=colors, linewidths=linewidths)
    

def replot(ax, map, no_xlabel=None, no_ylabel=None, title=None, cmap=None):
    """ 
    Allows for an already processed map to be plotted on a different set of axis.
    This may be done to look at how a plot may look with different cmaps, or to overplot
    contours or stars.
    
    Parameters:
        ax (plt.axis): axis the map is being plotted onto
        map (pym.Map): previously processed Map instance (e.g. a map that has already been
                        used to plot a moment, velocity etc..)
        cmap (str or Colormap instance): colormap used for the plot
    """
    if cmap is None:
        cmap = map.cmap
    
    map.create_plot(ax, cmap)
    map.format_plot(ax, no_xlabel, no_ylabel, title)
    map.add_beam_stars_vlabels(ax, map.bmin, map.bmaj, map.line)
    
    
def create_colorbar(figure, map, ax, fontsize='18', rotation=270, tick_label_size=16):
    """
    Places a colorbar with the fmin and fmax, and the appropriate label sourced from the given map.
    Ensure fmin/fmax are the same or have been scaled appropriately if using one colorbar for many plots
    
    The below will produce a colorbar along the side of a group of subplots, with the fmin and fmax values
    specified in the m1_map, as well as the appropriate label from it.
    
        fig, allAxes = plt.subplots(....)
        ...
        m1_map = data.plot_map(...ax=allAxes[j][i]...)
        ...
        create_colorbar(fig, m1_map, allAxes)
        
    Returns a colorbar instance
    """
    cb = figure.colorbar(map.image, ax=ax, orientation='vertical')
    label = map.create_cb_label()
    cb.set_label(label, fontsize=fontsize, rotation=rotation, labelpad=30)
    cb.ax.tick_params(labelsize=tick_label_size) 
    
    return cb
    
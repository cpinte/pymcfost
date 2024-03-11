import os, warnings
import astropy.io.fits as fits
from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import interpolate
from scipy.ndimage import convolve1d
import scipy.constants as sc

try:
    import progressbar
except ImportError:
    warnings.warn("progressbar is not present", UserWarning)

from .parameters import Params, find_parameter_file
from .utils import FWHM_to_sigma, default_cmap, Wm2_to_Tb, Jy_to_Tb,  Wm2_to_Jy, add_colorbar
from .wake import  plot_wake

DEFAULT_LINE_FILE = "lines.fits.gz"

class Line:

    def __init__(self, dir=None, line_file=None, **kwargs):

        # Correct path if needed
        dir = os.path.normpath(os.path.expanduser(dir))
        self.dir = dir

        # Search for parameter file
        para_file = find_parameter_file(dir)

        # Read parameter file
        self.P = Params(para_file)

        # If user specified line_file
        if line_file is not None:
            self._line_file = line_file
        # otherwise
        else:
            self._line_file = DEFAULT_LINE_FILE

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
                self.freq = np.array([self.restfreq])
                self.velocity_type = hdu[0].header['CTYPE3']
                if self.velocity_type == "VELO-LSR":
                    self.CRPIX3 = hdu[0].header['CRPIX3']
                    self.CRVAL3 = hdu[0].header['CRVAL3']
                    self.CDELT3 = hdu[0].header['CDELT3']
                    # velocity in km/s
                    self.velocity = self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)
                else:
                    raise ValueError("Velocity type is not recognised")

                self.nu = self.restfreq * (1 - self.velocity * 1000 / sc.c)

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
                try:
                    self.star_vr = hdu[6].data
                except:
                    self.star_vr = []

            self.dv = self.velocity[1] - self.velocity[0]

            hdu.close()
        except OSError:
            print('cannot open', self._line_file)

    def plot_map(
        self,
        i=0,
        iaz=0,
        iTrans=0,
        v=None,
        iv=None,
        insert=False,
        subtract_cont=False,
        moment=None,
        psf_FWHM=None,
        bmaj=None,
        bmin=None,
        bpa=None,
        plot_beam=None,
        beam_position=(0.125, 0.125), # fraction of plot width and height
        axes_unit="arcsec",
        conv_method=None,
        fmax=None,
        fmin=None,
        fpeak=None,
        dynamic_range=1e3,
        color_scale=None,
        colorbar=True,
        colorbar_size=10,
        cmap=None,
        ax=None,
        no_xlabel=False,
        no_ylabel=False,
        no_vlabel=False,
        no_xticks=False,
        no_yticks=False,
        vlabel_position=(0.5, 0.1), # fraction of plot width and height
        vlabel_size=8, # size in points
        title=None,
        title_size=14,
        limit=None,
        limits=None,
        Tb=False,
        Jy=False,
        mJy=False,
        per_arcsec2=False,
        per_beam=False,
        Delta_v=None,
        shift_dx=0,
        shift_dy=0,
        plot_stars=False,
        sink_particle_size=6,
        sink_particle_color="cyan",
        M0_threshold=None,
        iv_support=None,
        v_minmax = None,
        rms=0,
        interpolation=None,
        # from casa - contours
        plot_type="imshow",
        alpha=1.0,
        linewidths=None,
        zorder=None,
        origin='lower',
        levels=None,
        colors=None
    ):
        # Todo:
        # - print molecular info (eg CO J=3-2)

        if ax is None:
            ax = plt.gca()

        # -- Selecting channel corresponding to a given velocity
        if v is not None:
            iv = np.abs(self.velocity - v).argmin()
            print("Selecting channel #", iv)

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
        halfsize = np.asarray(self.lines.shape[-2:]) / 2 * pix_scale
        extent = [-halfsize[0]*xaxis_factor-shift_dx, halfsize[0]*xaxis_factor-shift_dx, -halfsize[1]-shift_dy, halfsize[1]-shift_dy]
        self.extent = extent

        # -- set color map
        if cmap is None:
            if moment in [1,9]:
                cmap = "RdBu_r"
            else:
                cmap = default_cmap
        unit = self.unit

        # -- beam or psf : psf_FWHM and bmaj and bmin are in arcsec, bpa in deg
        i_convolve = False
        beam = None
        if psf_FWHM is not None:
            # in pixels
            sigma = psf_FWHM / self.pixelscale * FWHM_to_sigma
            beam = Gaussian2DKernel(sigma)
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

        if plot_beam is None:
            plot_beam = False

        # -- Selecting convolution function
        if conv_method is None:
            conv_method = convolve_fft

        # -- Selection of image to plot
        if moment is not None:
            im = self.get_moment_map(
                i=i,
                iaz=iaz,
                iTrans=iTrans,
                moment=moment,
                i_convolve = i_convolve,
                beam=beam,
                conv_method=conv_method,
                subtract_cont=subtract_cont,
                M0_threshold=M0_threshold,
                iv_support=iv_support,
                v_minmax=v_minmax)

            if moment in [1,2,9]:
                Tb = False

        else:
            # individual channel
            if self.is_casa:
                cube = self.lines[:, :, :]
                # im = self.lines[iv+1,:,:])
            else:
                cube = self.lines[iaz, i, iTrans, :, :, :]
                # im = self.lines[i,iaz,iTrans,iv,:,:]

                # -- continuum subtraction
                if subtract_cont:
                    cube = np.maximum(cube - self.cont[iaz, i, iTrans, np.newaxis, :, :], 0.0)

            # Convolve spectrally
            if Delta_v is not None:
                cube = self._spectral_convolve(cube, Delta_v)

            if iv is None:
                raise ValueError("plot_map needs either v or iv")
            im = cube[iv, :, :]

            # -- Convolve image
            if i_convolve:
                im = conv_method(im, beam)

        if plot_beam is None:
            plot_beam = True

        # -- Conversion to brightness temperature
        if Tb:
            if self.is_casa:
                im = Jy_to_Tb(im, self.freq[iTrans], self.pixelscale)
            else:
                im = Wm2_to_Tb(im, self.freq[iTrans], self.pixelscale)
                im = np.nan_to_num(im)
            print("Max Tb=", np.max(im), "K")

            # turning off some flags that may interfere
            per_arcsec2 = False
            per_beam = False

        # -- Conversion to Jy
        if Jy:
            if not self.is_casa:
                im = Wm2_to_Jy(im, self.freq[iTrans])
                unit = unit.replace("W.m-2", "Jy")

        # -- Conversion to mJy
        if mJy:
            if not self.is_casa:
                im = Wm2_to_Jy(im, self.freq[iTrans]) * 1e3
                unit = unit.replace("W.m-2", "mJy")

        # -- Conversion to flux per arcsec2 or per beam
        if per_arcsec2:
            im = im / self.pixelscale**2
            unit = unit.replace("pixel-1", "arcsec-2")
        if per_beam:
            beam_area = bmin * bmaj * np.pi / (4.0 * np.log(2.0))
            pix_area = self.pixelscale**2
            im *= beam_area/pix_area
            unit = unit.replace("pixel-1", "beam-1")

        # -- Adding noise
        if rms > 0.0:
            noise = np.random.randn(im.size).reshape(im.shape)
            if i_convolve:
                noise = convolve_fft(noise, beam)
            noise *= rms / np.std(noise)
            im += noise

        # --- Plot range and color map`
        _color_scale = 'lin'
        if fmax is None:
            fmax = im.max()
        if fpeak is not None:
            fmax = im.max() * fpeak
        if fmin is None:
            fmin = im.min()

        if color_scale is None:
            color_scale = _color_scale
        if color_scale == 'log':
            if fmin <= 0.0:
                fmin = fmax / dynamic_range
            norm = mcolors.LogNorm(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'lin':
            norm = mcolors.Normalize(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'sqrt':
            norm = mcolors.PowerNorm(0.5, vmin=fmin, vmax=fmax, clip=True)
        else:
            raise ValueError("Unknown color scale: " + color_scale)

        # -- Make the plot
        ax.cla()
        image = ax.imshow(im, norm=norm, extent=extent, origin='lower', cmap=cmap,interpolation=interpolation)

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
            ax.set_title(title,size=title_size)

        # start from casa - contours
        if plot_type=="imshow":
            image = ax.imshow(
                im,
                norm=norm,
                extent=extent,
                origin='lower',
                cmap=cmap,
                alpha=alpha,
                interpolation=interpolation,
                zorder=zorder
            )
        elif plot_type=="contourf":
            image = ax.contourf(
                im,
                extent=extent,
                origin='lower',
                levels=levels,
                cmap=cmap,
                linewidths=linewidths,
                alpha=alpha,
                zorder=zorder
            )
        elif plot_type=="contour":
            if colors is not None:
                cmap=None
            image = ax.contour(
                im,
                extent=extent,
                origin='lower',
                levels=levels,
                cmap=cmap,
                linewidths=linewidths,
                alpha=alpha,
                zorder=zorder,
                colors=colors
            )
        # end from casa - contours

        # -- Color bar
        if colorbar:
            cb = add_colorbar(image)
            formatted_unit = unit.replace("-1", "$^{-1}$").replace("-2", "$^{-2}$")
            if moment == 0:
                if Tb:
                    cb.set_label("\int T$_\mathrm{b}\,\mathrm{d}v$ (K.km.s$^{-1}$)",size=colorbar_size)
                else:
                    cb.set_label("Flux (" + formatted_unit + ".km.s$^{-1}$)",size=colorbar_size)
            elif moment == 1 or moment == 9:
                cb.set_label("Velocity (km.s$^{-1})$",size=colorbar_size)
            elif moment == 2:
                cb.set_label("Velocity dispersion (km.s$^{-1}$)",size=colorbar_size)
            else:
                if Tb:
                    cb.set_label("T$_\mathrm{b}$ (K)",size=colorbar_size)
                else:
                    cb.set_label("Flux (" + formatted_unit + ")",size=colorbar_size)

        # -- Adding velocity
        if moment is None:
            if not no_vlabel:
                vlabx, vlaby = vlabel_position
                ax.text(
                    vlabx, vlaby,
                    f"$\Delta$v={self.velocity[iv]:<4.2f}$\,$km/s",
                    horizontalalignment='center',
                    size=vlabel_size,
                    color="white",
                    transform=ax.transAxes,
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
                color="grey",
            )
            ax.add_patch(beam)

        #-- Add stars
        if plot_stars:
            factor = pix_scale / self.pixelscale
            if isinstance(plot_stars,bool):
                x_stars = self.star_positions[0,iaz,i,:] * factor * (-xaxis_factor)
                y_stars = self.star_positions[1,iaz,i,:] * factor
            else: # int or list of int
                x_stars = self.star_positions[0,iaz,i,plot_stars] * factor * (-xaxis_factor)
                y_stars = self.star_positions[1,iaz,i,plot_stars] * factor
            ax.scatter(x_stars-shift_dx, y_stars-shift_dy,
                        color=sink_particle_color,s=sink_particle_size)

        #-- Saving the last plotted quantity
        self.last_image = im

        plt.sca(ax)

        return image

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
        subtract_cont=False
    ):

        if self.is_casa:
            line = np.sum(self.lines[:, :, :], axis=(1, 2))
            ylabel = "Flux (Jy)"
        else:
            if subtract_cont:
                lines = np.maximum(self.lines[iaz, i, iTrans, :, :, :] - self.cont[iaz, i, iTrans, np.newaxis, :, :], 0.0)
            else:
                lines = self.lines[iaz, i, iTrans, :, :, :]

            line = np.sum(lines, axis=(1, 2))
            ylabel = "Flux (W.m$^{-2}$)"

        print(line)
        plt.semilogy(self.velocity, line)

        if plot_cont:
            if self.is_casa:
                Fcont = 0.5 * (line[0] + line[-1])  # approx the continuum
            else:
                Fcont = np.sum(self.cont[iaz, i, iTrans, :, :])
            plt.plot([self.velocity[0], self.velocity[-1]], [Fcont, Fcont])

        xlabel = "v (m.s$^{-1}$)"

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def get_moment_map(self, i=0, iaz=0, iTrans=0, moment=0, i_convolve=False,
                       beam=None, conv_method=None,subtract_cont=False,M0_threshold=None,
                       iv_support=None, v_minmax = None):
        """
        This returns the moment maps in physical units, ie:
         - M1 is the average velocity (km/s)
         - M2 is the velocity dispersion (km/s)
        """
        if self.is_casa:
            cube = np.copy(self.lines[:, :, :])
        else:
            cube = np.copy(self.lines[iaz, i, iTrans, :, :, :])

        if subtract_cont:
            cube = np.maximum(cube - self.cont[iaz, i, iTrans, np.newaxis, :, :], 0.0)

        dv = self.velocity[1] - self.velocity[0]
        v = self.velocity

        if v_minmax is not None:
            vmin = np.min(v_minmax)
            vmax = np.max(v_minmax)
            iv_support = np.array(np.where(np.logical_and((self.velocity > vmin),(self.velocity < vmax)))).ravel()
            print("Selecting channels:", iv_support)

        if iv_support is None:
            nv = self.nv
        else:
            nv = len(iv_support)
            v = v[iv_support]
            cube = cube[iv_support,:,:]

        # Moment 0, 1 and 2
        if moment == 0: # We do not need to convolve the whole cube
            M0 = np.sum(cube, axis=0) * dv
            if i_convolve:
                M0 = conv_method(M0, beam)
        else:  # We need to convolve each channel indidually
            if i_convolve:
                print("Convolving individual channel maps, this may take a bit of time ....")
                try:
                    bar = progressbar.ProgressBar(
                        maxval=self.nv,
                        widgets=[
                            progressbar.Bar('=', '[', ']'),
                            ' ',
                            progressbar.Percentage(),
                        ]
                    )
                    bar.start()
                except:
                    pass
                for iv in range(nv):
                    try:
                        bar.update(iv + 1)
                    except:
                        pass
                    channel = np.copy(cube[iv, :, :])
                    cube[iv, :, :] = conv_method(channel, beam)
                try:
                    bar.finish()
                except:
                    pass

            if moment <=2:
                M0 = np.sum(cube, axis=0) * dv

            if moment >= 1 and moment <=2:
                M1 = np.sum(cube[:, :, :] * v[:, np.newaxis, np.newaxis], axis=0) * dv / M0

            if moment == 2:
                M2 = np.sqrt(np.sum(cube[:, :, :]
                                    * (v[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :])**2,
                                    axis=0,) * dv / M0)

            # Peak flux
            if moment == 8:
                M8 = np.max(cube, axis=0)

            # Velocity of the peak
            if moment == 9:
                vmax_index = np.argmax(cube, axis=0)
                M9 = v[(vmax_index)]

                #print(vmax_index.shape)
                #
                ## Extract the maximum and neighboring pixels
                #print("test1")
                #f_max = cube[(vmax_index)]
                #print(f_max.shape)
                #print("test2")
                #f_minus = cube[(vmax_index-1)]
                #print("test3")
                #f_plus = cube[(vmax_index+1)]
                #
                ## Work out the polynomial coefficients
                #print("test4")
                #a0 = 13. * f_max / 12. - (f_plus + f_minus) / 24.
                #print("test5")
                #a1 = 0.5 * (f_plus - f_minus)
                #print("test6")
                #a2 = 0.5 * (f_plus + f_minus - 2*f_max)
                #
                ## Compute the maximum of the quadratic
                #x_max = idx - 0.5 * a1 / a2
                #y_max = a0 - 0.25 * a1**2 / a2
                #
                #M9 = xmax

        if moment == 0:
            M = M0
        elif moment == 1:
            M = M1
        elif moment == 2:
            M = M2
        elif moment == 8:
            M = M8
        elif moment == 9:
            M = M9

        if M0_threshold is not None:
            M = np.ma.masked_where(M0 < M0_threshold, M)

        return M


    def _spectral_convolve(self, cube, Delta_v):

        print("Spectral convolution at ", Delta_v, "km/s ->",Delta_v / self.dv, "channels")

        # Creating a Hanning function with k points per mcfost delta_v
        # could be optmised to have k points accrod the hanning function, but this is fast
        k = 100

        width = 2 * Delta_v / self.dv # in channels
        nchan = int(width)+1
        if nchan == 1:
            return cube
        if nchan%2 == 0:
            nchan=nchan+1

        # number of point for the high res pectral response
        n = int(width*k)+1
        if n%2 != 0:
            n=n+1

        w = np.hanning(n)
        w = w/np.sum(w)

        # rebinning the spectral response to the mcfost resolution (and padding with zeros as required)
        n_pad = (nchan * k - n) // 2

        if n_pad > 0:
            pad = np.zeros(n_pad) * 1.0
            w = np.concatenate((pad,w,pad))

        w = w.reshape(nchan,k).sum(-1)
        cube = convolve1d(cube, w, mode='constant',axis=0, cval=0.)

        return cube

    def plot_wake(self,i=0,i_planet=1,HonR=0.1,q=0.25, **kwargs):

        inclinations = self.P.calc_inclinations()

        # plot_wake uses the dynamite convention
        if (self.P.map.RT_imax<90):
            inc = -inclinations[i]
        else:
            inc= 180 -inclinations[i]

        xy = self.star_positions[:,0,0,i_planet]

        plot_wake(xy,inc,-self.P.map.PA,HonR,q, **kwargs)

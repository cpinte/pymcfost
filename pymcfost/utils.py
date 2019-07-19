import numpy as np
from scipy.interpolate import interp1d
import scipy.constants as sc
import astropy.constants as const
import astropy.units as u

default_cmap = "inferno"

sigma_to_FWHM = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_to_sigma = 1.0 / sigma_to_FWHM
arcsec = np.pi / 648000


def bin_image(im, n, func=np.sum):
    # bin an image in blocks of n x n pixels
    # return a image of size im.shape/n

    nx = im.shape[0]
    nx_new = nx // n
    x0 = (nx - nx_new * n) // 2

    ny = im.shape[1]
    ny_new = ny // n
    y0 = (ny - ny_new * n) // 2

    return np.reshape(
        np.array(
            [
                func(im[x0 + k1 * n : (k1 + 1) * n, y0 + k2 * n : (k2 + 1) * n])
                for k1 in range(nx_new)
                for k2 in range(ny_new)
            ]
        ),
        (nx_new, ny_new),
    )


def Wm2_to_Jy(nuFnu, nu):
    '''
    Convert from W.m-2 to Jy
    nu [Hz]
    '''
    return 1e26 * nuFnu / nu


def Jy_to_Wm2(Fnu, nu):
    '''
    Convert from Jy to W.m-2
    nu [Hz]
    '''
    return 1e-26 * Fnu * nu


def Jybeam_to_Tb(Fnu, nu, bmaj, bmin):
    '''
     Convert Flux density in Jy/beam to brightness temperature [K]
     Flux [Jy]
     nu [Hz]
     bmaj, bmin in [arcsec]

     T [K]
    '''
    beam_area = bmin * bmaj * arcsec ** 2 * np.pi / (4.0 * np.log(2.0))
    exp_m1 = 1e26 * beam_area * 2.0 * sc.h / sc.c ** 2 * nu ** 3 / Fnu
    hnu_kT = np.log1p(np.maximum(exp_m1, 1e-10))

    Tb = sc.h * nu / (hnu_kT * sc.k)

    return Tb


def Jy_to_Tb(Fnu, nu, pixelscale):
    '''
     Convert Flux density in Jy/pixel to brightness temperature [K]
     Flux [Jy]
     nu [Hz]
     bmaj, bmin in [arcsec]

     T [K]
    '''
    pixel_area = (pixelscale * arcsec) ** 2
    exp_m1 = 1e16 * pixel_area * 2.0 * sc.h / sc.c ** 2 * nu ** 3 / Fnu
    hnu_kT = np.log1p(exp_m1 + 1e-10)

    Tb = sc.h * nu / (hnu_kT * sc.k)

    return Tb


def Wm2_to_Tb(nuFnu, nu, pixelscale):
    """Convert flux converted from Wm2/pixel to K using full Planck law.
        Convert Flux density in Jy/beam to brightness temperature [K]
        Flux [W.m-2/pixel]
        nu [Hz]
        bmaj, bmin, pixelscale in [arcsec]
        """
    pixel_area = (pixelscale * arcsec) ** 2
    exp_m1 = pixel_area * 2.0 * sc.h * nu ** 4 / (sc.c ** 2 * nuFnu)
    hnu_kT = np.log1p(exp_m1)

    Tb = sc.h * nu / (sc.k * hnu_kT)

    return Tb


class DustExtinction:

    import os

    __dirname__ = os.path.dirname(__file__)

    wl = []
    kext = []

    _extinction_dir = __dirname__ + "/extinction_laws"
    _filename_start = "kext_albedo_WD_MW_"
    _filename_end = "_D03.all"
    V = 5.47e-1  # V band wavelength in micron

    def __init__(self, Rv=3.1, **kwargs):
        self.filename = (
            self._extinction_dir
            + "/"
            + self._filename_start
            + str(Rv)
            + self._filename_end
        )
        self._read(**kwargs)

    def _read(self):

        with open(self.filename, 'r') as file:
            f = []
            for line in file:
                if (not line.startswith("#")) and (
                    len(line) > 1
                ):  # Skipping comments and empty lines
                    line = line.split()
                    self.wl.append(float(line[0]))
                    kpa = float(line[4])
                    albedo = float(line[1])
                    self.kext.append(kpa / (1.0 - albedo))

            # Normalize extinction in V band
            kext_interp = interp1d(self.wl, self.kext)
            kextV = kext_interp(self.V)
            self.kext /= kextV

    def redenning(self, wl, Av):
        """
        Computes extinction factor to apply for a given Av
        Flux_red = Flux * redenning

        wl in micron

        """
        kext_interp = interp1d(self.wl, self.kext)
        kext = kext_interp(wl)
        tau_V = 0.4 * np.log(10.0) * Av

        return np.exp(-tau_V * kext)

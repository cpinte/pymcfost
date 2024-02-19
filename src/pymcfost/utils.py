import numpy as np
from scipy.interpolate import interp1d
from scipy import ndimage
import scipy.constants as sc
import astropy.constants as const
import astropy.units as u
import cmasher as cmr

default_cmap = cmr.arctic

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
    exp_m1 = 1e26 * pixel_area * 2.0 * sc.h / sc.c ** 2 * nu ** 3 / Fnu
    hnu_kT = np.log1p(np.maximum(exp_m1, 1e-10))

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


# -- Functions to deal the synthesized beam.
def _beam_area(self):
    """Beam area in arcsec^2"""
    return np.pi * self.bmaj * self.bmin / (4.0 * np.log(2.0))

def _beam_area_str(self):
    """Beam area in steradian^2"""
    return self._beam_area() * arcsec ** 2

def _pixel_area(self):
    return self.pixelscale ** 2

def _beam_area_pix(self):
    """Beam area in pix^2."""
    return self._beam_area() / self._pixel_area()


def telescope_beam(wl,D):
    """ wl and D in m, returns FWHM in arcsec"""
    return 0.989 * wl/D / 4.84814e-6


def make_cut(im, x0,y0,x1,y1,num=None,plot=False):
    """
    Make a cut in image 'im' along a line between (x0,y0) and (x1,y1)
    x0, y0,x1,y1 are pixel coordinates
    """

    if plot:
        vmax = np.max(im)
        vmin = vmax * 1e-6
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        plt.imshow(im,origin="lower", norm=norm)
        plt.plot([x0,x1],[y0,y1])


    if num is not None:
        # Extract the values along the line, using cubic interpolation
        x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
        zi = ndimage.map_coordinates(im, np.vstack((y,x)))
    else:
        # Extract the values along the line at the pixel spacing
        length = int(np.hypot(x1-x0, y1-y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        zi = im[y.astype(int), x.astype(int)]

    return zi


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


def Hill_radius():
    pass
    #d * (Mplanet/3*Mstar)**(1./3)


def splash2mcfost(anglex, angley, anglez):
    #Convert the splash angles to mcfost angles

    # Base unit vector
    x0 = [1,0,0]
    y0 = [0,1,0]
    z0 = [0,0,1]

    # Splash rotated vectors
    x = _rotate_splash_axes(x0,-anglex,-angley,-anglez)
    y = _rotate_splash_axes(y0,-anglex,-angley,-anglez)
    z = _rotate_splash_axes(z0,-anglex,-angley,-anglez)

    # MCFOST angles
    mcfost_i = np.arccos(np.dot(z,z0)) * 180./np.pi

    if abs(mcfost_i) > 1e-30:
        print("test1")
        # angle du vecteur z dans le plan (-y0,x0)
        mcfost_az = (np.arctan2(np.dot(z,x0), -np.dot(z,y0)) ) * 180./np.pi
        # angle du vecteur z0 dans le plan x_image, y_image (orientation astro + 90deg)
        mcfost_PA = -( np.arctan2(np.dot(x,z0), np.dot(y,z0)) )  * 180./np.pi
    else:
        print("test2")
        mcfost_az = 0.
        # angle du vecteur y dans le plan x0, y0
        mcfost_PA = (np.arctan2(np.dot(y,x0),np.dot(y,y0)) ) * 180./np.pi


    print("anglex =",anglex, "angley=", angley, "anglez=", anglez,"\n")
    print("Direction to oberver=",z)
    print("x-image=",x)
    print("y_image = ", y,"\n")
    print("MCFOST parameters :")
    print("inclination =", mcfost_i)
    print("azimuth =", mcfost_az)
    print("PA =", mcfost_PA)

    return [mcfost_i, mcfost_az, mcfost_PA]

def _rotate_splash(xyz, anglex, angley, anglez):
    # Defines rotations as in splash
    # This function is to rotate the data

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    # rotate about z
    if np.abs(anglez) > 1e-30:
        r = np.sqrt(x**2+y**2)
        phi = np.arctan2(y,x)
        phi -= anglez/180*np.pi
        x = r*np.cos(phi)
        y = r*np.sin(phi)

    # rotate about y
    if np.abs(angley) > 1e-30:
        r = np.sqrt(z**2+x**2)
        phi = np.arctan2(z,x)
        phi -= angley/180*np.pi
        x = r*np.cos(phi)
        z = r*np.sin(phi)

    # rotate about x
    if np.abs(anglex) > 1e-30:
        r = np.sqrt(y**2+z**2)
        phi = np.arctan2(z,y)
        phi -= anglex/180*np.pi
        y = r*np.cos(phi)
        z = r*np.sin(phi)

    return np.array([x,y,z])


def _rotate_splash_axes(xyz, anglex, angley, anglez):
    # Defines rotations as in splash, but in reserve order
    # as we rotate the axes instead of the data

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    # rotate about x
    if np.abs(anglex) > 1e-30:
        r = np.sqrt(y**2+z**2)
        phi = np.arctan2(z,y)
        phi -= anglex/180*np.pi
        y = r*np.cos(phi)
        z = r*np.sin(phi)

    # rotate about y
    if np.abs(angley) > 1e-30:
        r = np.sqrt(z**2+x**2)
        phi = np.arctan2(z,x)
        phi -= angley/180*np.pi
        x = r*np.cos(phi)
        z = r*np.sin(phi)

    # rotate about z
    if np.abs(anglez) > 1e-30:
        r = np.sqrt(x**2+y**2)
        phi = np.arctan2(y,x)
        phi -= anglez/180*np.pi
        x = r*np.cos(phi)
        y = r*np.sin(phi)

    return np.array([x,y,z])

def rotate_vec(u,v,angle):
    '''
      rotate a vector (u) around an axis defined by another vector (v)
      by an angle (theta) using the Rodrigues rotation formula
    '''
    k = v/np.sqrt(np.inner(v,v))
    w = np.cross(k,u)
    k_dot_u = np.inner(k,u)
    for i,uval in enumerate(u):
        u[i] = u[i]*np.cos(angle) + w[i]*np.sin(angle) + k[i]*k_dot_u*(1.-np.cos(angle))
    return u

def rotate_coords(x,y,z,inc,PA):
    '''
       rotate x,y,z coordinates into the observational plane
    '''
    k = [-np.sin(PA), np.cos(PA), 0.]
    xvec = [x,y,z]
    xrot = rotate_vec(xvec,k,inc)
    return xrot[0],xrot[1],xrot[2]

def rotate_to_obs_plane(x,y,inc,PA):
    '''
       same as rotate_coords but takes 2D x,y as arrays
    '''
    for i,xx in enumerate(x):    # this can probably be done more efficiently
        x[i],y[i],dum = rotate_coords(x[i],y[i],0.,inc,PA)
    return x,y

def planet_position(model, i_planet, i_star, ):
    '''
     Returns planet position [arcsec] and PA [deg] in the map
    '''
    xy_planet = model.star_positions[:,0,0,i_planet]
    xy_star = model.star_positions[:,0,0,i_star]
    dxy = xy_planet - xy_star

    dist = np.hypot(dxy[0],dxy[1])
    PA = np.rad2deg(np.arctan2(dxy[1],-dxy[0])) + 360 - 90

    return [dist, PA]

def add_colorbar(mappable, shift=None, width=0.05, ax=None, trim_left=0, trim_right=0, side="right",**kwargs):
    # creates a color bar that does not shrink the main plot or panel
    # only works for horizontal bars so far

    if ax is None:
        ax = mappable.axes

    # Get current figure dimensions
    try:
        fig = ax.figure
        p = np.zeros([1,4])
        p[0,:] = ax.get_position().get_points().flatten()
    except:
        fig = ax[0].figure
        p = np.zeros([ax.size,4])
        for k, a in enumerate(ax):
            p[k,:] = a.get_position().get_points().flatten()
    xmin = np.amin(p[:,0]) ; xmax = np.amax(p[:,2]) ; dx = xmax - xmin
    ymin = np.amin(p[:,1]) ; ymax = np.amax(p[:,3]) ; dy = ymax - ymin

    if side=="top":
        if shift is None:
            shift = 0.2
        cax = fig.add_axes([xmin + trim_left, ymax + shift * dy, dx - trim_left - trim_right, width * dy])
        cax.xaxis.set_ticks_position('top')
        return fig.colorbar(mappable, cax=cax, orientation="horizontal",**kwargs)
    elif side=="right":
        if shift is None:
            shift = 0.05
        cax = fig.add_axes([xmax + shift*dx, ymin, width * dx, dy])
        cax.xaxis.set_ticks_position('top')
        return fig.colorbar(mappable, cax=cax, orientation="vertical",**kwargs)

def get_planet_r_az(disk_PA, disk_inc, planet_r, planet_PA):
    # For a given disk PA and inc, and planet PA, and projected separation
    # gives the az to be passed to the planet_az option and the deprojected separation
    # input :
    # disk_PA, inc and  planet_PA in degrees
    # planet_r radius in arcsec or au
    # output :
    #  r in same unit as planet_r
    # planet_az in degrees (is value to be passed to mcfost)

    # test.get_planet_rPA does the opposite from a mcfost image
    # Note this basically the same function as dynamite.to_mcfost
    # We just keep it here for conveniencefor now

    dPA = planet_PA - disk_PA
    az = np.arctan(np.tan(np.deg2rad(dPA)) / np.cos(np.deg2rad(disk_inc)))
    az = - np.rad2deg(az) # conversion to deg and correct convention for mcfost


    y_p = planet_r * np.sin(np.deg2rad(dPA))
    x_p = planet_r * np.cos(np.deg2rad(dPA))

    x = x_p
    y = y_p / np.cos(np.deg2rad(disk_inc))

    r = np.hypot(x,y)

    # test :
    #mcfost.get_planet_r_az(62.5,50.2, 0.60521173 * 157.2, 11.619613647460938)
    # should give : (130.00000395126042, 62.500002877567475)

    # mcfost inclination is oppositte to dynamite
    if (disk_inc<0):
        mcfost_inc=-disk_inc
    else: # to avoid the bug in red/blue PA with negative inclination in mcfost
        mcfost_inc=180-disk_inc
        az = -az

    print("MCFOST parameter should be:")
    print("i=",mcfost_inc,"deg")
    print("planet r=",r,"au")
    print("planet az=",az,"deg (for cmd line option)")

    return mcfost_inc, r, az

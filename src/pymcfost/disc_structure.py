import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os

from .parameters import Params, find_parameter_file
from .run import run

class Disc:
    """
    A class to handle MCFOST disc structure data.

    This class reads and processes MCFOST disc structure outputs, providing methods 
    to access the spatial grid, gas density, and add geometric features.

    Attributes:
        dir (str): Directory containing MCFOST output files
        P (Params): MCFOST parameter object
        grid (numpy.ndarray): Spatial grid structure
        gas_density (numpy.ndarray): Gas density distribution
        volume (numpy.ndarray): Cell volumes

    Example:
        >>> disc = Disc(dir="path/to/mcfost/data")
        >>> r = disc.r()  # Get radial coordinates
    """

    def __init__(self, dir=None, **kwargs):
        """
        Initialize a Disc object.

        Args:
            dir (str): Path to directory containing MCFOST output
            **kwargs: Additional arguments passed to _read()
        """
        # Correct path if needed
        dir = os.path.normpath(os.path.expanduser(dir))
        if dir[-9:] != "data_disk":
            dir = os.path.join(dir, "data_disk")
        self.dir = dir

        # Search for parameter file
        para_file = find_parameter_file(dir)

        # Read parameter file
        self.P = Params(para_file)

        # Read model results
        self._read(**kwargs)

    def _read(self):
        # Read grid file
        try:
            hdu = fits.open(self.dir + "/grid.fits.gz")
            self.grid = hdu[0].data
            hdu.close()
        except OSError:
            print('cannot open grid.fits.gz')

        # Read gas density file
        try:
            hdu = fits.open(self.dir + "/gas_density.fits.gz")
            self.gas_density = hdu[0].data
            hdu.close()
        except OSError:
            print('cannot open gas_density.fits.gz')

        # Read volume file
        try:
            hdu = fits.open(self.dir + "/volume.fits.gz")
            self.volume = hdu[0].data
            hdu.close()
        except OSError:
            print('cannot open volume.fits.gz')

    def r(self):
        """
        Get radial coordinates of the grid.

        Returns:
            numpy.ndarray: Radial coordinates in AU
        """
        if self.grid.ndim > 2:
            return self.grid[0, :, :, :]
        else:
            return np.sqrt(self.grid[0, :] ** 2 + self.grid[1, :] ** 2)

    def z(self):
        """
        Get vertical coordinates of the grid.

        Returns:
            numpy.ndarray: Vertical coordinates in AU
        """
        if self.grid.ndim > 2:
            return self.grid[1, :, :, :]
        else:
            return self.grid[2, :]

    def add_spiral(
        self, a=30, sigma=10, f=1, theta0=0, rmin=None, rmax=None, n_az=None
    ):
        """
        Add a geometrical spiral on a 2D (or 3D) mcfost density grid
        and return a 3D array which can be directly written as a fits
        file for mcfost to read

        geometrical spiral r = a (theta - theta0)
        surface density is mutiply by f at the crest of the spiral
        the spiral has a Gaussian profil in (x,y) with sigma given in au

        Args:
            a (float): Spiral parameter, r = a(θ - θ₀)
            sigma (float): Width of the spiral in AU
            f (float): Density enhancement factor at spiral crest
            theta0 (float): Initial angle in radians
            rmin (float, optional): Inner radius in AU
            rmax (float, optional): Outer radius in AU
            n_az (int, optional): Number of azimuthal points

        Returns:
            numpy.ndarray: Modified density structure with spiral
        """

        if self.grid.ndim <= 2:
            ValueError("Can only add a spiral on a cylindrical or spherical grid")

        if n_az is None:
            n_az = self.grid.shape[1]
        phi = np.linspace(0, 2 * np.pi, n_az, endpoint=False)

        r = self.grid[0, 0, 0, :]

        if rmin is None:
            rmin = r.min()
        if rmax is None:
            rmax = r.max()

        x = r[np.newaxis, :] * np.cos(phi[:, np.newaxis])
        y = r[np.newaxis, :] * np.sin(phi[:, np.newaxis])

        # Just to test
        # x = np.linspace(-100,100,500)
        # x, y = np.meshgrid(x,x)
        # r = np.sqrt(x**2 + y**2) # we recalcule in preparation for other types of grid

        # rc=50, hc=0.15, alpha=1.5, beta=0.25
        # theta_c = 0.
        # theta = theta_c + np.sign(r - rc)/hc * \
        #        ((r/rc)**(1+beta) * (1/(1+beta) - 1/(1-alpha + beta) * (r/rc)**(-alpha)) \
        #         - 1/(1+beta) - 1/(1-alpha + beta))

        r_spiral = np.geomspace(rmin, rmax, num=5000)
        theta = r_spiral / a + theta0

        x_spiral = r_spiral * np.cos(theta)
        y_spiral = r_spiral * np.sin(theta)

        correct = np.ones(x.shape)

        # This is really badly implemented, but fast enough that we don't care
        sigma2 = sigma ** 2
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                d2 = np.min((x_spiral - x[i, j]) ** 2 + (y_spiral - y[i, j]) ** 2)
                correct[i, j] += f * np.exp(-0.5 * d2 / sigma2)

        triang = tri.Triangulation(x.flatten(), y.flatten())
        plt.tripcolor(triang, correct.flatten(), shading='flat')

        return self.gas_density[np.newaxis, :, :] * correct[:, np.newaxis, :]


    def add_banana(
           self, r0=None, phi0=None, f=1, delta_r=None, delta_phi=None, n_az=None, sigma=None
    ):
        """
        Add a geometrical banana on a 2D (or 3D) mcfost density grid
        and return a 3D array which can be directly written as a fits
        file for mcfost to read

        surface density is mutiply by (1+f) at the crest of the banana
        the banana has a Gaussian profile in (r,phi) with sigma given in au, degrees

        Args:
            r0 (float): Radial position of enhancement in AU
            phi0 (float): Azimuthal position in degrees
            f (float): Density enhancement factor
            delta_r (float): Radial width in AU
            delta_phi (float): Azimuthal width in degrees
            n_az (int, optional): Number of azimuthal points
            sigma (str): Path to sigma data file

        Returns:
            numpy.ndarray: Modified density structure with banana feature
        """
        phi0 = np.radians(phi0)
        delta_phi = np.radians(delta_phi)
        
        sigma_data = fits.open(sigma)[0].data
        if self.grid.ndim <= 2:
            ValueError("Can only add a banana on a cylindrical or spherical grid")

        if n_az is None:
            n_az = self.grid.shape[1]
        phi = np.linspace(0, 2 * np.pi, n_az, endpoint=False)

        r = self.grid[0, 0, 0, :]

        # r is radius of each cell hypot(x.y)
        # phi is angle of each cell arctan2(y,x)
        # r = np.hypot(x,y)
        # phi = np.arctan2(y,x)

        correct = 1 + f * (np.exp(-0.5 * ((r-r0)/delta_r)**2)[np.newaxis, :] * (np.exp(-0.5 * ((phi-phi0)/delta_phi)**2)[:, np.newaxis]))
        return sigma_data*correct

        # We want to conserve mass
        #correct = correct/np.mean(correct)

        #x = r[np.newaxis, :] * np.cos(phi[:, np.newaxis])
        #y = r[np.newaxis, :] * np.sin(phi[:, np.newaxis])
        #triang = tri.Triangulation(x.flatten(), y.flatten())
        #plt.tripcolor(triang, correct.flatten(), shading='flat')

        #return self.gas_density[np.newaxis, :, :] * correct[:, np.newaxis, :]
        #return self.surface_density[:,np.newaxis] * correct[:,:]


def check_grid(model):
    """
    We check if the disc structure already exists
    if not, we check if it exists
    if not, we try to compute it
    """

    try:
        grid = model.disc.grid
    except:
        try:
            print("Trying to read grid structure ...")
            model.disc = Disc(model.basedir)
            grid = model.disc.grid
        except:
            print("No grid structure, trying to create it ...")
            run(model.P.filename, options=model.P.options+" -disk_struct")
            try:
                print("Trying to read grid structure again ...")
                model.disc = Disc(model.basedir)
                grid = model.disc.grid
            except AttributeError:
                print("Cannot read grid in " + model.basedir)

    return grid


def _plot_cutz(model, y, r=None, dr=None, log=None, **kwargs):

    grid = check_grid(model)

    if grid.ndim > 2:
        r_mcfost = grid[0, 0, 0, :]
        i = np.argmin(np.abs(r_mcfost - r))
        print("selected_radius =", r_mcfost[i])
        z_mcfost = grid[1, 0, :, i]

        y = y[:,i]

        if log:
            plt.loglog(z_mcfost, y, **kwargs)
        else:
            plt.plot(z_mcfost, y, **kwargs)

    else:
        r_mcfost = np.sqrt(grid[0, :] ** 2 + grid[1, :] ** 2)
        ou = r_mcfost > 1e-6  # Removing star
        y = y[ou]
        r_mcfost = r_mcfost[ou]
        z_mcfost = grid[2, ou]

        # Selecting data points
        ou = (r_mcfost > r - dr) & (r_mcfost < r + dr)

        z_mcfost = z_mcfost[ou]
        y = y[ou]

        #plt.plot(z_mcfost, T, "o", **kwargs)
        fig = plt.gcf()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        density = ax.scatter_density(z_mcfost,y, **kwargs)

    plt.xlabel("z [au]")

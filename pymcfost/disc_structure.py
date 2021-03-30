import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os

from parameters import Params, find_parameter_file


class Disc:
    def __init__(self, dir=None, **kwargs):
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
        if self.grid.ndim > 2:
            return self.grid[0, :, :, :]
        else:
            return np.sqrt(self.grid[0, :] ** 2 + self.grid[1, :] ** 2)

    def z(self):
        if self.grid.ndim > 2:
            return self.grid[1, :, :, :]
        else:
            return self.grid[2, :]

    def add_spiral(
        self, a=30, sigma=10, f=1, theta0=0, rmin=None, rmax=None, n_az=None
    ):
        """ Add a geometrucal spiral on a 2D (or 3D) mcfost density grid
        and return a 3D array which can be directly written as a fits
        file for mcfost to read

        geometrical spiral r = a (theta - theta0)
        surface density is mutiply by f at the crest of the spiral
        the spiral has a Gaussin profil in (x,y) with sigma given in au
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

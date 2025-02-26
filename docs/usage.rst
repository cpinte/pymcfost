Usage
=====

The pymcfost package provides classes to work with MCFOST image and line data.

Running MCFOST
--------------

The package provides a function to run MCFOST directly:

.. code-block:: python

   from pymcfost import run
   
   # Run MCFOST with a parameter file
   run("model.para")
   
   # Run with additional options
   run("model.para", options="-img 1.3")
   
   # Run silently and save output to a log file
   run("model.para", silent=True, logfile="mcfost.log")

Image Class
-----------

The ``Image`` class handles continuum images from MCFOST radiative transfer calculations.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pymcfost import Image
   import matplotlib.plt as plt
   
   # Load an image from a directory containing MCFOST output
   image = Image(dir="path/to/mcfost/data")
   
   # Plot the image
   image.plot()

   # save image to file
   plt.savefig('mcfost-image.pdf',bbox_inches='tight')

   # also show on screen
   plt.show()

Advanced usage
~~~~~~~~~~~~~~

.. code-block:: python

   from pymcfost import Image
   import matplotlib.plt as plt

   # Load an image from a directory containing MCFOST output
   image = Image(dir="path/to/mcfost/data")

   # set up the plotting page
   fig, axes = plt.subplots(1,1,figsize=(8,8))

   # Plot the image
   image.plot(
       ax=axes,
       i=0,              # inclination index
       iaz=0,           # azimuth angle index
       vmin=0.,       # minimum value for colorscale
       vmax=50.,       # maximum value for colorscale
       dynamic_range=1e6, # ratio between max and min values
       colorbar=True,   # show colorbar
       psf_FWHM=None,   # convolve with Gaussian beam of this FWHM (in arcsec)
       plot_stars=True  # show stellar positions
       Tb=True          # use brightness temperature
   )

   plt.show()

Plotting model visibilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pymcfost import Image

   # Load an image from a directory containing MCFOST output
   image = Image(dir="path/to/mcfost/data")

   # Calculate and plot model visibilities
   baselines, vis, fim = image.calc_vis(
       i=0,       # inclination index  
       iaz=0,     # azimuth index
       klambda=True  # plot in units of kilolambda
   )

Line Class
----------

The ``Line`` class handles spectral line data from MCFOST calculations.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pymcfost import Line
   
   # Load line data
   line = Line(dir="path/to/mcfost/data")
   
   # Plot channel map
   line.plot_map(
       i=0,           # inclination index
       iaz=0,         # azimuth angle index
       iTrans=0,      # transition index
       v=None,        # velocity in km/s (alternative to iv)
       iv=None,       # velocity channel index
       moment=None,   # moment map to plot (0=integrated intensity, 1=velocity, 2=dispersion)
       psf_FWHM=None, # beam FWHM in arcsec
       colorbar=True  # show colorbar
   )

SED Class
---------

The ``SED`` class handles spectral energy distributions and temperature structures.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pymcfost import SED
   
   # Load SED data
   sed = SED(dir="path/to/mcfost/data")
   
   # Plot SED
   sed.plot(
       i=0,           # inclination index
       iaz=0,         # azimuth angle index
       MC=False,      # plot Monte Carlo results
       contrib=True   # show individual contributions
   )
   
   # Plot temperature structure
   sed.plot_T(
       iaz=0,         # azimuth angle index
       log=True       # use logarithmic scale
   )
   
   # Plot vertical temperature profile
   sed.plot_Tz(
       r=100.0,       # radius in AU
       dr=5.0         # radial range to average over
   )

Common Parameters
-----------------

Many classes share some common parameters:

- ``i``: Index for inclination angle
- ``iaz``: Index for azimuth angle  
- ``psf_FWHM``: FWHM of Gaussian beam for convolution (in arcsec)
- ``bmaj``, ``bmin``, ``bpa``: Beam major/minor axis and position angle
- ``axes_unit``: Units for axes ('arcsec', 'au', or 'pixels')
- ``plot_stars``: Whether to show stellar positions
- ``colorbar``: Whether to show colorbar

The plotting methods return matplotlib objects that can be further customized. 

Disc Structure
--------------

The ``Disc`` class handles the spatial structure and density distribution of the disc.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pymcfost import Disc
   
   # Load disc structure
   disc = Disc(dir="path/to/mcfost/data")
   
   # Get spatial coordinates
   r = disc.r()  # radial coordinates
   z = disc.z()  # vertical coordinates
   
   # Add a spiral feature
   new_density = disc.add_spiral(
       a=30,          # spiral parameter
       sigma=10,      # width in AU
       f=1.5,         # density enhancement
       rmin=20,       # inner radius
       rmax=100       # outer radius
   )

Dust Model
----------

The ``Dust_model`` class handles dust opacity properties.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pymcfost import Dust_model
   
   # Load dust model
   dust = Dust_model(dir="path/to/mcfost/data")
   
   # Plot opacities
   dust.plot_kappa(abs=True, scatt=True)
   
   # Plot albedo
   dust.plot_albedo()
   
   # Save opacity data
   dust.print_kappa(file="opacities.txt")

CASA Simulations
----------------

The package provides functions to create synthetic ALMA observations using CASA.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pymcfost import Image, CASA_simdata
   
   # Load an image
   image = Image(dir="path/to/mcfost/data")
   
   # Quick simulation with beam convolution
   pseudo_CASA_simdata(
       image,
       bmaj=0.5,        # beam major axis in arcsec
       bmin=0.3,        # beam minor axis in arcsec
       bpa=30,          # beam position angle in degrees
       rms=1e-4         # noise level
   )
   
   # Full CASA/ALMA simulation
   CASA_simdata(
       image,
       obstime=3600,    # 1 hour observation
       config="alma.cycle6.3",  # ALMA configuration
       pwv=0.5,         # precipitable water vapor
       decl="-22d59m59.8"  # source declination
   )

The CASA simulation functions require CASA to be installed on your system.

Directory Structure
-------------------

pymcfost expects MCFOST output files to be organized in specific subdirectories:

- ``data_th/``: Contains SED and temperature data
- ``data_disk/``: Contains disc structure data
- ``data_[wavelength]/``: Contains image data at specific wavelengths
- ``UV/``: Contains dust opacity data

When specifying directories to pymcfost classes, you can either point to these specific subdirectories or to the parent directory containing them. 
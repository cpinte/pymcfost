# pymcfost

pymcfost is a python package that provides an interface to the 3D radiative transfer code mcfost.

pymcfost offers (or will offer) the following functionalities:

 - set up continuum and line models
 - read a single model or library of models
 - plot basic quantities, e.g. density structures, temperature maps
 - plot observables : SEDs, image (with convolution), polarisation maps, visibilities, channels maps (with spatial and spectral convolution), moment maps.
 - convert units, e.g. W.m-2 to Jy or brightness temperature
 - provides an interface to the ALMA CASA simulator
 - consistent interface with the casa python package to compare observations with models (ask C. Pinte)
 - (TBD) read and plot dust models
 - (TBD) direct interface to the Mie, DHS and aggregates dust properties calculations
 - (TBD) direct interface to the ML chemical predictions
 - (TBD) direct interface to the Voronoi mesh interface (available via fits files only so far)
 - (TBD) consistent interface with PLONK (Phantom/splash python interface)

pymcfost was born as an attempt to port in python the functions that were available in the yorick-mcfost code. The goal is to provide a simple and light interface to explore a single (or a few) model(s).

The fitting routines of the yorick interface are yet to be ported into pymcfost.
An alternative python distribution is available at https://github.com/cpinte/mcfost-python . It is more tailored towards handling large grid of models and model fitting.


## Installation:

```
git clone https://github.com/cpinte/pymcfost.git
cd pymcfost
python3 setup.py install
```

If you don't have the `sudo` rights, use `python3 setup.py install --user`.

To install in developer mode: (i.e. using symlinks to point directly
at this directory, so that code changes here are immediately available
without needing to repeat the above step):

	```
 python3 setup.py develop
```

## Main structural differences with mcfost-python so far:

- python >= 3.6 vs python 2.x
- only parameter file >= 3.0
- handles parameter files with mutiple zones, dust population, molecules, stars, etc. Parameter files are stored in objects rather than dicts, allowing more flexibility.
- does not and will not handle observational data, only models

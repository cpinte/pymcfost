# pymcfost

pymcfost is a python package that provides an interface to the 3D radiative transfer code mcfost.

pymcfost offers the following functionalities:

 - set up continuum and line models
 - read a single model or library of models
 - plot basic quantities, e.g. density structures, temperature maps
 - plot observables : SEDs, image (with convolution), polarisation maps, visibilities, channels maps (with spatial and spectral convolution), moment maps.
 - convert units, e.g. W.m-2 to Jy or brightness temperature
 - provides an interface to the ALMA CASA simulator

pymcfost was born as an attempt to port in python the functions that were available in the yorick-mcfost code.
The goal is to provide a simple and light interface to explore a single (or a few) model(s).

An alternative python distribution is available at https://github.com/cpinte/mcfost-python . It is more tailored towards handling large grid of models and model fitting.


Main structural differences with mcfost-python so far:
------------------------------------------------------

- python >= 3.6 vs python 2.x
- only parameter file >= 3.0
- handles parameter files with mutiple zones, dust population, molecules, stars, etc. Parameter files are stored in objects rather than dicts, allowing more flexibility.
- does not and will not handle observational data, only models

# pymcfost

This is an attempt to port in python the functions that were available in the yorick-mcfost code.
The goal is to provide a simple and light interface to explore a single (or a few) model(s).

A more complete python distribution (in particular dealing with sets of models and model fitting) is available at https://github.com/cpinte/mcfost-python


Differences with mcfost-python so far:
--------------------------------------

- python >= 3.6 vs python 2.x
- only parameter file >= 3.0
- handles parameter files with mutiple zones, dust population, molecules, stars, etc. Parameter files are stored in objects rather than dicts, allowing more flexibility.
- does not and will not handle observational data, only models
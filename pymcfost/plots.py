import os

import astropy.io.fits as fits
from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

try:
    import progressbar
except ImportError:
    print('WARNING: progressbar is not present')
from scipy import interpolate

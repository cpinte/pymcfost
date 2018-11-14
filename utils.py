import numpy as np
from scipy.interpolate import interp1d

default_cmap = "inferno"

FWHM_to_sigma = 1./(2.*np.sqrt(2.*np.log(2)))

def bin_image(im, n, func=np.sum):
    # bin an image in blocks of n x n pixels
    # return a image of size im.shape/n

    nx = im.shape[0]
    nx_new = nx//n
    x0 = (nx - nx_new * n)//2

    ny = im.shape[1]
    ny_new = ny//n
    y0 = (ny - ny_new * n)//2

    return np.reshape(np.array([func(im[x0+k1*n:(k1+1)*n,y0+k2*n:(k2+1)*n]) for k1 in range(nx_new) for k2 in range(ny_new)]),(nx_new,ny_new))


class DustExtinction:

    wl = []
    kext = []

    _extinction_dir = "extinction_laws"
    _filename_start = "kext_albedo_WD_MW_"
    _filename_end = "_D03.all"
    V = 5.47e-1 # V band wavelength in micron

    def __init__(self,Rv=3.1, **kwargs):
        self.filename = self._extinction_dir+"/"+self._filename_start+str(Rv)+self._filename_end
        self._read(**kwargs)

    def _read(self):

        with open(self.filename, 'r') as file:
            f = []
            for line in file:
                if ((not line.startswith("#")) and (len(line)>1)): # Skipping comments and empty lines
                    line = line.split()
                    self.wl.append(float(line[0]))
                    kpa = float(line[4])
                    albedo = float(line[1])
                    self.kext.append(kpa/(1.-albedo))

            # Normalize extinction in V band
            kext_interp = interp1d(self.wl,self.kext)
            kextV = kext_interp(self.V)
            self.kext /= kextV

    def redenning(self,wl, Av):
        """
        Computes extinction factor to apply for a given Av
        Flux_red = Flux * redenning

        wl in micron

        """
        kext_interp = interp1d(self.wl,self.kext)
        kext = kext_interp(wl)
        tau_V = 0.4*np.log(10.)*Av

        return np.exp(-tau_V * kext)

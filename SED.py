import astropy.io.fits as fits
import matplotlib.pyplot as plt

from parameters import McfostParams, find_parameter_file

class McfostSED:

    _sed_th_file = ".sed_th.fits.gz";
    _sed_mc_file = "sed_mc.fits.gz";
    _sed_rt_file = "sed_rt.fits.gz";
    _temperature_file = "Temperature.fits.gz";

    def __init__(self, dir=None, **kwargs):
        self.dir = dir

        # Search for parameter file
        para_file = find_parameter_file(dir)
        print(para_file)

        # Read parameter file
        self.P = McfostParams(para_file)

        # Read model results
        self._read(**kwargs)


    def _read(self):
        # Read sed files
        try:
            hdu = fits.open(self.dir+"/"+self._sed_th_file)
        except OSError:
            print('cannot open', _sed_th_file)
        self._sed_th = hdu[0].data
        self._wl_th = hdu[1].data
        hdu.close()

        try:
            hdu = fits.open(self.dir+"/"+self._sed_mc_file)
        except OSError:
            print('cannot open', self._sed_mc_file)
        self._sed_mc = hdu[0].data
        hdu.close()

        try:
            hdu = fits.open(self.dir+"/"+self._sed_rt_file)
        except OSError:
            print('cannot open', _sed_rt_file)
        self.sed = hdu[0].data
        self.wl = hdu[1].data
        hdu.close()

        # Read temperature file
        try:
            hdu = fits.open(self.dir+"/"+self._temperature_file)
        except OSError:
            print('cannot open', _temperature_file)
        self.T = hdu[0].data
        hdu.close()


    def plot(self, i, MC=False, **kwargs):
        # todo :
        # - add extinction
        # - separate contribution
        # - change units

        if (MC):
            _sed = self._sed_mc[0,0,i,]
        else:
            _sed = self.sed[0,0,i,]

        plt.loglog(self.wl, _sed, **kwargs)
        plt.xlabel("$\lambda$ [$\mu$m]")
        plt.ylabel("$\lambda.F_{\lambda}$ [W.m$^{-2}$]")

    def verif(self):
        pass

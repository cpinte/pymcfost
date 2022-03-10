import numpy as np
import os
import subprocess
import scipy.constants as sc
from scipy import interpolate
from .image import Image
from .utils import Wm2_to_Jy, FWHM_to_sigma
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve
from scipy.ndimage import convolve1d

def pseudo_CASA_simdata(model,i=0,iaz=0,iTrans=None,simu_name = "pseudo_casa",beam=None,bmaj=None,bmin=None,bpa=None,subtract_cont=False,Delta_v=None, rms=0):
    """
    Generate a fits file as if it was a CASA simdata output
     - convolve spatially with beam (bmin, bmaj in arsec, bpa in degrees)
     - convolve spectrally with Delta_v in km/s
     - todo : resample in velocity as required

    Basically generate a CASA fits file with perfect uv coverage and no noise
    """

    workdir = "CASA/"
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    _CASA_clean(workdir)

    is_image = isinstance(model, Image)

    # --- Checking arguments
    if not is_image:
        if iTrans is None:
            raise Exception("Missing transition number iTrans : iTrans is a python index from 0 to nTrans-1")

        nTrans = model.freq.size
        if iTrans > nTrans - 1:
            raise Exception(f"ERROR: iTrans is not in the computed range : nTrans={nTrans}")

    if beam is not None:
        bmaj = beam
        bmin = beam
        bpa = 0

    if bmaj is None:
        raise Exception("Missing beam")

    # -- Flux setup : we want Jy/pixel first
    if is_image:
        if model.is_casa:
            image = model.image[:, :]
        else:
            # Convert to Jy
            image = Wm2_to_Jy(model.image[0, iaz, i, :, :], sc.c / model.wl)
    else:  # cube
        if model.is_casa:
            # -- continuum subtraction
            if subtract_cont:
                image = model.lines[:,:,:] - model.lines[0,:,:]
            else:
                image = model.lines[:, :, :]
        else:
            # -- continuum subtraction
            if subtract_cont:
                image = model.lines[iaz, i, iTrans,:,:,:] - model.cont[iaz, i, iTrans, np.newaxis, :, :]
            else:
                image = model.lines[iaz, i, iTrans,:,:,:]

            # Convert to Jy
            image = Wm2_to_Jy(image, model.freq[iTrans])

    #-- writing fits file
    hdr = fits.Header()
    hdr["EXTEND"] = True
    hdr["OBJECT"] = "mcfost"
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CRVAL1"] = 0.0
    hdr["CRPIX1"] = int(model.nx / 2 + 1)
    hdr["CDELT1"] = -model.pixelscale / 3600.0
    hdr["CUNIT1"] = "deg"

    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL2"] = 0.0
    hdr["CRPIX2"] = int(model.ny / 2 + 1)
    hdr["CDELT2"] = model.pixelscale / 3600.0
    hdr["CUNIT2"] = "deg"

    if is_image:
        # 3rd axis
        hdr["CTYPE3"] = "STOKES"
        hdr["CRVAL3"] = 1.0
        hdr["CDELT3"] = 1.0
        hdr["CRPIX3"] = 1

        # 4th axis
        hdr["CTYPE4"] = "FREQ"
        hdr["CRVAL4"] = model.freq # Hz
        hdr["CDELT4"] = 2e9  # 2GHz by default
        hdr["CRPIX4"] = 0
    else:
        hdr["CTYPE3"] = "VELO-LSR"
        hdr["CRVAL3"] = 0.0  # line center
        hdr["CRPIX3"] = model.nv//2 + 1
        hdr["CDELT3"] = model.dv * 1000
        hdr["CUNIT3"] = "m/s"

    hdr["RESTFREQ"] = model.freq[iTrans]  # Hz
    hdr["BUNIT"] = "JY/BEAM"
    hdr["BTYPE"] = "Intensity"
    hdr["BMAJ"] = bmaj/3600.
    hdr["BMIN"] = bmin/3600.
    hdr["BPA"] = bpa

    # Convolve spectrally
    if Delta_v is not None:
        image = model._spectral_convolve(image, Delta_v)

    print(f"Spatial convolution at {bmaj} x {bmin}")
    #-- Convolution by beam
    sigma_x = bmin / model.pixelscale * FWHM_to_sigma  # in pixels
    sigma_y = bmaj / model.pixelscale * FWHM_to_sigma  # in pixels
    beam = Gaussian2DKernel(sigma_x, sigma_y, bpa * np.pi / 180)

    if image.ndim == 2:
        image = convolve_fft(image, beam)
    else:
        for iv in range(image.shape[0]):
            image[iv,:,:] = convolve_fft(image[iv,:,:], beam)

    #-- Jy/pixel to Jy/beam
    beam_area = bmin * bmaj * np.pi / (4.0 * np.log(2.0))
    pix_area = model.pixelscale**3
    image *= beam_area/pix_area

    print(f"Peak flux is {np.max(image)} Jy/beam")

	#-- For testing purpose : this needs to be updated and to come before
	#-- compute the scale factor in 1 channel once,
	#-- then add noise before spatial and spectral convolution
    if rms > 0.0:
        noise = np.random.randn(image.size).reshape(image.shape)
        for iv in range(image.shape[0]):
            noise[iv,:,:] = convolve_fft(noise[iv,:,:], beam)
        if Delta_v is not None:
            noise =  model._spectral_convolve(noise, Delta_v)
        print(np.std(noise), beam_area/pix_area)
        noise *= rms / np.std(noise)
        image += noise

    hdu = fits.PrimaryHDU(image, header=hdr)
    hdul = fits.HDUList(hdu)

    hdul.writeto(workdir + simu_name + ".fits", overwrite=True)



#	if rms > 0.0:
#		noise = np.random.randn(cube.size).reshape(cube.shape)
#		noise = np.array([convolve(c, beam) for c in noise])
#		noise = np.convolve(noise, spectral_response, axis=0)
#		noise *= rms / np.std(noise)
#		image += noise

def CASA_simdata(
    model,
    i=0,
    iaz=0,
    obstime=None,
    config=None,
    resol=None,
    sampling_time=None,
    pwv=0.0,
    decl="-22d59m59.8",
    phase_noise=False,
    name="simu",
    iTrans=None,
    rt=True,
    only_prepare=False,
    interferometer='alma',
    mosaic=False,
    mapsize=None,
    channels=None,
    width=None,
    correct_flux=1.0,
    simu_name=None,
    ms=None,
    n_iter=10000,
    hourangle="transit"):
    """
    Prepare a MCFOST model for the CASA alma simulator

    Generates 2 files:
      - a python CASA script
      - a fits file with the required dimensions and  keywords

    Then run the simulator and export a fits file with the results

    Tested to work with CASA 5.4.0-68 on MacOS : command line to call CASA is assumed to be "casa"
    """

    workdir = "CASA/"
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    _CASA_clean(workdir)

    is_image = isinstance(model, Image)

    # --- Checking arguments
    if not is_image:
        if iTrans is None:
            raise Exception("Missing transition number iTrans")

        nTrans = model.freq.size
        if iTrans > nTrans - 1:
            raise Exception(f"ERROR: iTrans is not in the computed range : nTrans={nTrans}")

    if ms is None:
        # --- Setting a configuration and observing time for simalma
        simobs_custom = False

        if obstime is None:
            raise Exception("Missing obstime")

        if sampling_time is None:
            sampling_time = obstime / 1000

        if config is None:
            if resol is None:
                raise Exception("Missing config or resol")
            else:
                resol_name = f"{resol:2.2f}"
                resol_name_script = f"alma_={resol:6.6f}arcsec"
        else:
            if isinstance(config, int):
                config = f"alma.cycle6.{config}"
                resol_name = "_config=" + config
                resol_name_script = config
            else: # we do not change config
                resol_name_script = config

    else:
        # -- Setting up for simobs_custom
        simobs_custom = True

    # -- Thermal noise
    if pwv is None:
        print("pwv not specified --> No thermal noise")
        th_noise = "''"
        lth_noise = 0
    else:
        th_noise = "'tsys-atm'"
        spwv = f"{pwv:4.2f}"
        is_th_noise = True

    # -- Frequency setup
    if is_image:
        freq = sc.c / (model.wl * 1e-6) * 1e-9  # [Ghz]
        inwidth = 8  # 8 Ghz par defaut en continu
        print(f"Setting channel width to 8Ghz")
        inchan = 1  # 1 channel pour continu
    else:  # cube
        dv = model.dv * 1000.0  # [m/s]
        freq = model.freq[iTrans] * 1e-9  # [Ghz]

        if width is None:
            inwidth = dv / sc.c * model.freq[iTrans] * 1e-9  # [Ghz]
            print(f"Setting channel width to {dv:f} m/s")
        else:
            inwidth = width

        if channels is None:
            inchan = 2 * model.P.mol.molecule[imol].nv + 1
            channels = np.arange(inchan)
        else:
            if isinstance(channels, int):
                inchan = 1
            else:
                inchan = len(channels)

    # -- Flux setup
    if is_image:
        if model.is_casa:
            image = model.image[:, :]
        else:
            image = Wm2_to_Jy(model.image[0, iaz, i, :, :], sc.c / model.wl)  # Convert to Jy
            image = image[
                np.newaxis, np.newaxis, :, :
            ]  # Adding spectral & pola dimensions
    else:  # cube
        if model.is_casa:
            image = model.lines[channels, :, :]
        else:
            image = Wm2_to_Jy(model.lines[iaz, i, iTrans, channels, :, :], model.freq[iTrans])  # Convert to Jy
    if (image.ndim == 2):  # Adding extra spectral dimension if there is only 1 channel selected
        image = image[np.newaxis, :, :]

    # -- pixels
    incell = model.pixelscale

    # -- Filenames
    if simu_name is None:
        simu_name = "casa_simu"
    if is_image:
        simu_name = simu_name + f"_lambda={model.wl}_obstime={obstime}_decl=" + decl

    # ---------------------------------------------
    # -- fits file
    # ---------------------------------------------

    hdr = fits.Header()
    hdr["EXTEND"] = True
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CRVAL1"] = 0.0
    hdr["CRPIX1"] = int(model.nx / 2 + 1)
    hdr["CDELT1"] = -model.pixelscale / 3600.0

    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL2"] = 0.0
    hdr["CRPIX2"] = int(model.ny / 2 + 1)
    hdr["CDELT2"] = model.pixelscale / 3600.0

    if is_image:
        # 3rd axis
        hdr["CTYPE3"] = "STOKES"
        hdr["CRVAL3"] = 1.0
        hdr["CDELT3"] = 1.0
        hdr["CRPIX3"] = 1

        # 4th axis
        hdr["CTYPE4"] = "FREQ"
        hdr["CRVAL4"] = freq * 1e9  # Hz
        hdr["CDELT4"] = 2e9  # 2GHz by default
        hdr["CRPIX4"] = 0
    else:
        # WARNING this is incorrect : the simulator will still work but the velocity axis in the output file will be off
        hdr["CTYPE3"] = "VELO-LSR"
        hdr["CRVAL3"] = 0.0  # line center
        hdr["CRPIX3"] = inchan
        hdr["CDELT3"] = dv

    hdr["RESTFREQ"] = freq * 1e9  # Hz
    hdr["BUNIT"] = "JY/PIXEL"
    hdr["BTYPE"] = "Intensity"

    hdu = fits.PrimaryHDU(image, header=hdr)
    hdul = fits.HDUList(hdu)
    hdul.writeto(workdir + simu_name + ".raw.fits", overwrite=True)

    # ---------------------------------------------
    # -- CASA script
    # ---------------------------------------------
    # spatial setup
    txt = f"""project = 'DISK'
skymodel = '{simu_name}.raw.fits'
dryrun = False
modifymodel = True
inbright = 'unchanged'
indirection = 'J2000 18h00m00.02 {decl}' # mosaic center, or list of pointings
hourangle = '{hourangle}'
incell = '{incell}arcsec'
mapsize = ''
pointingspacing = '1.0arcmin'
setpointings = True
predict = True
complist = ''
refdate = '2012/06/21/03:25:00'
"""
    # Spectral setup
    txt += f"""inchan = {inchan}
incenter = '{freq:17.15e}Ghz'
inwidth = '{inwidth:17.15e}Ghz'
"""
    if simobs_custom:
        txt += "vis = '" + ms + "'\n"
    else:
        # Observing time
        txt += f"""totaltime = '{obstime}s'
integration = '{sampling_time}s'
"""

        # Configuration
        txt += f"repodir=os.getenv(\"CASAPATH\").split(\' \')[0]\n"
        if resol is None:
            if isinstance(config, str):
                txt += f"antennalist = repodir+'/data/alma/simmos/" + config + ".cfg'\n"
            elif isinstance(config, list):
                txt += f"antennalist = ["
                for i, c in enumerate(config):
                    if i > 0:
                        txt += ","
                    txt += "repodir+'/data/alma/simmos/" + c + ".cfg'"
                txt += "]\n"
        else:
            txt += f"antennalist = \"alma;"+ resol_name +"arcsec\"\n"

    # Noise
    txt += f"thermalnoise = " + th_noise + "\n"
    if is_th_noise:
        txt += f"user_pwv = {pwv}\n"
        if not simobs_custom:
            txt += "vis = project+'.noisy.ms' # clean the data with *thermal noise added*\n"

    # Imaging
    txt += f"""image = True
cleanmode = 'clark'
imsize = [{model.nx},{model.ny}]
cell = ''
niter = {n_iter}
threshold = '0.0mJy'
weighting = 'natural'
outertaper = []
stokes = 'I'
"""

    # default simalma values
    txt += f"""analyze = True
graphics = 'file'
overwrite = True
verbose = False
async = False
"""

    # Actual script
    if simobs_custom:
        if is_image:
            txt += f"mode = 'cont'\n"
        else:
            txt += f"mode = 'line'\n"
        txt += f"simobs_custom()\n"
        txt += (
            "exportfits(imagename=project+'/'+project,fitsimage='"
            + simu_name
            + f".fits',overwrite=True)\n"
        )
    else:
        txt += f"simalma()\n"
        txt += (
            "exportfits(imagename=project+'/'+project+'."
            + resol_name_script
            + f".noisy.image',fitsimage='"
            + simu_name
            + f".fits',overwrite=True)\n"
        )

    # txt += "pl.savefig('"+simu_name+f".png')\n"
    txt += "exit\n"

    # writing the script to disk
    outfile = open(workdir + simu_name + ".py", 'w')
    outfile.write(txt)
    outfile.close()

    if not only_prepare:
        return _run_CASA(simu_name)


def _run_CASA(simu_name, node_dir=""):
    print("Starting casa ...")

    workdir = "CASA/" + node_dir + "/"
    _CASA_clean(workdir)

    # -- Do we run the simulator with phase noise ?
    # fh = cfitsio_open(workdir+simu_name+".raw.fits","r");
    # phase_noise = cfitsio_get(fh, "phase_noise") ;
    # cfitsio_close,fh;

    # Running the simulator
    homedir = os.getcwd()
    os.chdir(workdir)

    # cmd="/Applications/CASA.app/./Contents/Resources/python/regressions/admin/runcasa_from_shell.sh 0 "+simu_name+".py"
    cmd = "casa --nogui -c " + simu_name + ".py"
    subprocess.call(cmd.split())
    os.chdir(homedir)

    # system, "mv "+workdir+"ALMA_disk.png  "+workdir+simu_name+".png" ;
    # system, "mv "+workdir+"ALMA_disk.fits "+workdir+simu_name+".fits" ;
    #  system, "mv "+workdir+"casapy.log "+workdir+simu_name+".log" ;

    # if (phase_noise) system, "mv "+workdir+"ALMA_disk_phase-noise.fits "+workdir+simu_name+"_phase-noise.fits" ;

    _CASA_clean(workdir)

    print("CASA simulation DONE")
    # write, "Simulation done in ",tac(), " sec" ;

    return simu_name


def _CASA_clean(workdir):
    cmd = (
        "rm -rf "
        + workdir
        + "DISK* "
        + workdir
        + "disk.fits "
        + workdir
        + "*.last "
        + workdir
        + "disk.py "
        + workdir
        + "ALMA_disk.png "
        + workdir
        + "ALMA_disk.fits "
        + workdir
        + "*.log*"
    )
    subprocess.call(cmd.split())

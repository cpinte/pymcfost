import numpy as np
import subprocess
import scipy.constants as sc

def CASA_simdata(model, i=0, iaz=0, obstime=None,config=None,resol=None,sampling_time=None,pwv=0.,decl="-22d59m59.8",phase_noise=False,name="simu",iTrans=None,rt=True,only_prepare=False,interferometer='alma',mosaic=False,mapsize=None,channels=None,width=None,correct_flux=1.0,simu_name=None):
    """
    prepare un model pour le simulateur ALMA de CASA :
    - taille des pixels
    - Jy/pixels
    - frequency
    - width in Ghz

    Si resol est donne (en arcsec), simdata calcule la configuration correspondante

    Genere un fichier fits et un script pour simdata
    """

    workdir="CASA/"
    _CASA_clean, workdir

    is_image = isinstance(model, image.Image)

    #--- Checking arguments
    if not is_image:
        if iTrans is None:
            raise Exception("Missing transition number iTrans")

        nTrans = model.freq.size
        if iTrans > nTrans-1:
            raise Exception("ERROR: iTrans is not in the computed range")


    if obstime is None:
        raise Exception("Missing obstime")


    if config is None:
        resol_name = f"_resol={resol:2.2f}"
        resol_name_script = f"alma_={resol:6.6f}arcsec"
    else:
        if isinstance(config,int):
            sconfig = f"{config:2i}"
            resol_name = "_config="+interferometer+"."+sconfig2 ;
            resol_name_script = sconfig2 ;
        else:
            raise Exception("config must be an integer")


    if pwv is None:
        print("pwv not specified --> No thermal noise" )
        th_noise = "''"
        lth_noise = 0
    else:
        th_noise = "'tsys-atm'"
        spwv = f"{pwv:4.2f}"
        is_th_noise = True


    #-- Frequency setup
    if is_image:
        freq = model.freq * 1e-9 # [Ghz]
        inwidth = 8 # 8 Ghz par defaut en continu
        inchan = 1 # 1 channel pour continu
    else: # cube
        dv = self.dv * 1000. # [m/s]
        freq = model.freq[iTrans] * 1e-9 # [Ghz]

        if width is None:
            inwidth = dv/sc.c * model.freq[iTrans]

        if channels is None:
            inchan = 2*model.P.mol.nv+1
            channels = np.arrange(inchan)
        else:
            inchan = channels.size()

    #-- Flux setup
    if is_image:
        if is_casa:
            image = model.image[:,:]
        else:
            image = Wm2_to_Jy(vis,model.image[0,iaz,i,:,:],sc.c/ model.wl) # Convert to Jy
    else: # cube
        if is_casa:
            image = model.line[:,:,channels]
        else:
            image = Wm2_to_Jy(vis,model.line[:,:,channels,iTrans,iaz,i],sc.c/ model.wl) # Convert to Jy

    #-- pixels
    incell = model.pixelscale

    #-- Filenames
    if simu_name is None:
        simu_name = "casa_simu_"
    if is_image:
        simu_name = simu_name+f"_lambda={model.wl}_obstime={obstime}_decl="+decl

    #-- CASA script
    # spatial setup
    txt=f"""project = 'DISK'
dryrun = False
modifymodel = True
inbright = 'unchanged'
indirection = 'J2000 18h00m00.02 {decl}' # mosaic center, or list of pointings
incell = '{incell}arcsec'
mapsize = ''
pointingspacing = '1.0arcmin'
setpointings = True
predict = True
complist = ''"""


    # Spectral setup
    txt += f"""inchan = {inchan}
incenter = '{incenter:17.15e}Ghz'
inwidth = '{inwidth:17.15e}Ghz'"""

    # Observing time
    txt += f"""totaltime = '{obstime}s'
integration = '{integartion}s'
refdate = '2012/06/21/03:25:00'"""

    # Configuration
    txt += f"repodir=os.getenv(\"CASAPATH\").split(\' \')[0]"
    if resol is None:
        txt += f"antennalist = repodir+'/data/alma/simmos/"+sconfig+".cfg"
    else:
        txt += f"antennalist = \"alma;%farcsec\" \% "+sresol

    # Noise
    txt += f"thermalnoise = "+th_noise
    if is_th_noise:
        txt += f"""user_pwv = {pwv}
        vis = project+'.noisy.ms' # clean the data with *thermal noise added*"""

    # Imaging
    txt += f"""image = True
 cleanmode = 'clark'
 imsize = [{model.nx},{model.ny}]
 cell = ''
 niter = {n_iter}
 threshold = '0.0mJy'
 weighting = 'natural'
 outertaper = []
 stokes = 'I'"""

    # default simalma values
    txt +=f"""analyze = True
graphics = 'file'
overwrite = True
verbose = False
async = False"""

    # Actual script
    txt += f"simalma()"
    txt += "pl.savefig('"+simu_name+f".png')"
    txt += "exportfits(imagename=project+'/'+project+'."+resol_name_script+f".noisy.image',fitsimage='"+simu_name+f".fits')"
    txt += "exit"

    if not only_prepare:
        return _run_CASA(simu_name)


def _run_CASA(simu_name):
    pass

def _CASA_clean(workdir):
    cmd = "rm -rf "+workdir+"DISK* "+workdir+"disk.fits "+workdir+"*.last "+workdir+"disk.py "+workdir+"ALMA_disk.png "+workdir+"ALMA_disk.fits "+workdir+"*.log*"
    subprocess.call(cmd.split())

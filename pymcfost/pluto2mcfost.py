import sys
import matplotlib.pyplot as plt
from .plutoTools import readVTKSpherical
import numpy as np
import copy
from scipy import interpolate
from astropy import units as u
from astropy import constants as c
import astropy.io.fits as fits
import subprocess
from .parameters import Params
from .disc_structure import Disc

def pluto2mcfost(
        filename,
        nr=100,
        nz=100,
        umass=1,
        udist = 10,
        mcfost_ref_file="/Users/cpinte/mcfost/src/ref3.0_3D.para",
        mcfost_filename="mcfost_PLUTO.para",
        fitsname=None,
):

    # umass in solar masses
    # udist in au

    # Inner and outer radii over which we plot the solution
    rin=1.0
    rout=3.0


    ##########################
    ## Units options
    #########################

    # Mass of central object (modifiable)
    Mstar=umass * c.M_sun
    # unit of length (=inner radius of the run, here taken to be 10 AU) (modifiable)
    R0=(udist*u.au).to(u.m)
    #Assumed surface density at R0 (eq 5 in Riols 2020)
    Sigma0=200*(R0.to(u.au)/u.au)**(-0.5)*u.g/(u.m)**2
    # Disc aspect-ratio
    epsilon=0.05

    ###########################
    # Computed quantities
    ###########################

    #Time unit
    T0=np.sqrt(R0**3/(Mstar*c.G)).to(u.s)
    #Velocity unit
    V0=R0/T0

    #density unit rho=sigma/(sqr(Zpi)H)
    rho0=Sigma0/(np.sqrt(2*np.pi)*epsilon*R0)
    # pressure unit
    P0=rho0*V0*V0

    #density unit rho=sigma/(sqr(Zpi)H)
    rho0=Sigma0/(np.sqrt(2*np.pi)*epsilon*R0)

    # Read data file
    V=readVTKSpherical(filename)

    # Apply units
    V.data['rho']= np.squeeze((rho0*V.data['rho']).value)
    V.data['prs']= np.squeeze((P0*V.data['prs']).value)
    V.data['vx1']= np.squeeze((V0*V.data['vx1']).value)
    V.data['vx2']= np.squeeze((V0*V.data['vx2']).value)
    V.data['vx3']= np.squeeze((V0*V.data['vx3']).value)

    V.r = V.r * udist # V.r in au

    # Compute the sound speed
    V.data['cs']=np.sqrt(V.data['prs']/V.data['rho'])

    n_r = V.r.size
    n_theta = V.theta.size

    # define cylindrical coordinates of the pluto model
    [r,theta] = np.meshgrid(V.r,V.theta,indexing='ij')

    rcyl = r*np.sin(theta)
    z    = r*np.cos(theta)

    # Density and velocity data
    rho = V.data['rho']

    # Create the cartesian components (in units of sound speed)
    vr = V.data['vx1']*np.sin(theta)+V.data['vx2']*np.cos(theta)
    vz = V.data['vx1']*np.cos(theta)-V.data['vx2']*np.sin(theta)
    vphi = V.data['vx3']

    # --- mcfost : updating values from PLUTO
    # --- Setting up a parameter file and creating the corresponding grid
    # Setting up the mcfost model
    P = Params(mcfost_ref_file)
    P.grid.n_rad = nr
    P.grid.n_rad_in = 1
    P.grid.nz = nz
    P.grid.n_az = 1

    P.zones[0].Rin = np.min(V.r)
    P.zones[0].edge = 0.0
    P.zones[0].Rout = np.max(V.r)
    P.zones[0].Rc = 0.0

    # Don't compute SED
    P.simu.compute_SED = False

    # -- Turn off symmetries
    P.simu.image_symmetry = False
    P.simu.central_symmetry = False
    P.simu.axial_symmetry = False

    # -- Write new parameter file
    P.writeto(mcfost_filename)

    # --- Running mcfost to create the grid
    subprocess.call(["rm", "-rf", "data_disk", "data_disk_old"])
    result = subprocess.call(["mcfost", mcfost_filename, "-disk_struct"])

    # --- Reading mcfost grid
    mcfost_disc = Disc("./")

    # --- print the mcfost radial grid to check that it is the same as FARGO's
    print("MCFOST radii=")
    print(mcfost_disc.r()[0, 0, :])

    mcfost_r = mcfost_disc.r().transpose() # dims are now, r, z, theta
    mcfost_z = mcfost_disc.z().transpose()

    # -- computing the 3D density structure for mcfost
    rho_mcfost    = interpolate.griddata((rcyl.flatten(),z.flatten()),rho.flatten() ,(mcfost_r,mcfost_z),fill_value=0)
    vr_mcfost     = interpolate.griddata((rcyl.flatten(),z.flatten()),vr.flatten()  ,(mcfost_r,mcfost_z),fill_value=0)
    vz_mcfost     = interpolate.griddata((rcyl.flatten(),z.flatten()),vz.flatten()  ,(mcfost_r,mcfost_z),fill_value=0)
    vphi_mcfost   = interpolate.griddata((rcyl.flatten(),z.flatten()),vphi.flatten(),(mcfost_r,mcfost_z),fill_value=0)

    velocities = np.array([vr_mcfost.transpose(),vphi_mcfost.transpose(),vz_mcfost.transpose()])
    #velocities = np.moveaxis(velocities, 0, -1)

    primary_hdu = fits.PrimaryHDU(np.abs(rho_mcfost.transpose()))
    second_hdu = fits.ImageHDU(np.abs(rho_mcfost.transpose()))
    tertiary_hdu = fits.ImageHDU(velocities)
    primary_hdu.header['hierarch read_gas_velocity'] = 2
    primary_hdu.header['hierarch gas_dust_ratio'] = 100
    primary_hdu.header['hierarch read_gas_density'] = 1
    primary_hdu.header['read_n_a'] = 0
    hdul = fits.HDUList([primary_hdu, second_hdu, tertiary_hdu])

    # --- Write a fits file for mcfost
    if fitsname is not None:
        print("Writing ", fitsname, " to disk")
        hdul.writeto(fitsname, overwrite=True)
    else:
        print("Fits file for mcfost was not created")

    return mcfost_r, mcfost_z, rho_mcfost, vr_mcfost, vphi_mcfost, vz_mcfost

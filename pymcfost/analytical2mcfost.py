import numpy as np
import astropy.io.fits as fits
import astropy.units as u
import sys
import subprocess
from .parameters import Params
from .disc_structure import Disc


def analytic_params_import(file):

    parobj = open(file)
    params = {} # this is a dictionary
    for line in parobj:
        line = line.strip() # removes additional spaces after a line and also the newline character
        if line and line[0] != '%': # the first 'if line' allows to remove blank lines.
            line = line.split()
            if len(line) == 1:
                params[line[0]] = None

            elif len(line) == 2:
                params[line[0]] = line[1]

            elif len(line) > 2:
                params[line[0]] = [ line[1] ]
                for i in range(2,len(line)):
                    params[line[0]] += [ line[i] ]

    Rmax = float(params['Rdisc'])
    if 'Rmin' in params.keys():
        Rmin = float(params['Rmin'])
    else:
        Rmin = Rdisc/50

    Nr = int(params['Nr'])
    Nphi = int(params['Nphi'])

    Rplanet = float(params['Rplanet'])
    PA = float(params['PA'])
    i = float(params['inclination'])
    PAp = float(params['PAp'])
    M = float(params['Mstar'])
    Mp = float(params['Mplanet'])

    cw = int(params['cw'])

    return Nr, Nphi, Rmin, Rmax, Rplanet, PAp, i, PA, M, Mp, cw



def analytic2mcfost(
    data_dir="/home/tom/Documents/Protoplanetary_Discs_Project/HD_163296_analytics_for_channels_grid/Analytics_output/hd163_4Mj",
    nz=50,
    analytic_params_file="/home/tom/Documents/Protoplanetary_Discs_Project/HD_163296_analytics_for_channels_grid/Analytics_output/hd163_4Mj/hd163.param",
    mcfost_ref_file="/home/tom/Documents/pymcfost/pymcfost/ref3.0_3D.para",
    mcfost_filename="mcfost_hd163_4Mj.para",
    fitsname='hd163_analytic_4Mj',
):

    # read in properties from analytics
    Nr, Nphi, Rmin, Rmax, Rplanet, PAp, i, PA, M, Mp, cw = analytic_params_import(analytic_params_file)

    # set no. of sink particles (code modifications needed for anything other than 2)
    Nsink = 2

    from astropy import constants as const
    from os import makedirs

    # -- Setup constants
    G = const.G.cgs.value
    au = const.au.cgs.value
    M_sol = const.M_sun.cgs.value

    # --- Reading data
    rho = np.load(data_dir + "/density.npy")
    vrad = 1e3*np.load(data_dir + "/vr.npy")
    vtheta = 1e3*np.load(data_dir + "/vphi.npy")

    # --- reshape
    Rho = rho.transpose()
    Vrad = vrad.transpose()
    Vtheta = vtheta.transpose()

    # --- mcfost : updating values from analytic kinks parameter file
    # --- Setting up a parameter file and creating the corresponding grid
    P = Params(mcfost_ref_file)
    P.grid.n_rad = Nr
    P.grid.n_rad_in = 1
    P.grid.nz = nz
    P.grid.n_az = Nphi

    # -- Disc geometry
    P.zones[0].Rin = Rmin
    P.zones[0].edge = 0.0
    P.zones[0].Rout = Rmax
    P.zones[0].Rc = 0.0

    # -- Don't compute SED
    P.simu.compute_SED = False

    # -- pixel grid
    P.map.nx = 1001
    P.map.ny = 1001

    # -- ROTATION DIR. NEEDS TO BE CORRECT FOR THE FOLLOWING TO WORK
    P.map.RT_az_min = -1*PAp # 45, planet angle
    P.map.RT_az_max = -1*PAp
    P.map.RT_n_az = 1

    # -- distance
    P.map.distance = 101.5

    # -- PA
    P.map.PA = PA - 90

    # -- inclination
    P.map.RT_imin = -1*i
    P.map.RT_imax = -1*i
    P.map.RT_ntheta = 1

    P.phot.nphot_T = 1.28e+07

    # -- velocity channel settings
    P.mol.molecule[0].v_max = 3.2     # max velocity
    P.mol.molecule[0].nv = 20         # number of velocity channels
    P.mol.molecule[0].n_trans = 1     # no. of lines in ray tracing

    # -- Turn off symmetries
    P.simu.image_symmetry = False
    P.simu.central_symmetry = False
    P.simu.axial_symmetry = False

    # -- Write new parameter file
    P.writeto(mcfost_filename)

    # --- Running mcfost to create the grid
    subprocess.call(["rm", "-rf", "data_disk", "data_disk_old"])
    subprocess.call(["mcfost", mcfost_filename, "-disk_struct"])

    # --- Reading mcfost grid
    mcfost_disc = Disc("./")

    # -- same dimension order as FARGO
    mcfost_z = mcfost_disc.z()

    # -- taking only half the grid (+z)
    mcfost_z = mcfost_z[:, nz:, :]
    mcfost_z = mcfost_z.transpose()  # dims are now, r, z, theta

    # -- defining H [au]
    h = 0.05 * mcfost_disc.r()[:, 0, :].transpose()

    # -- computing the 3D density structure for mcfost
    rho_mcfost = Rho[:, np.newaxis, :] * np.exp(-0.5 * (mcfost_z / h[:, np.newaxis, :]))
    vtheta_mcfost = np.repeat(Vtheta[:, np.newaxis, :], nz, axis=1)
    vrad_mcfost = np.repeat(Vrad[:, np.newaxis, :], nz, axis=1)
    vz_mcfost = np.zeros((220,50,200))

    # -- create master velocities array
    velocities = np.array([vrad_mcfost.transpose(),vtheta_mcfost.transpose(),vz_mcfost.transpose()])

    # -- setup HDUs for fits file
    primary_hdu = fits.PrimaryHDU(np.abs(rho_mcfost.transpose()))
    second_hdu = fits.ImageHDU(np.abs(rho_mcfost.transpose()))
    tertiary_hdu = fits.ImageHDU(velocities)

    # -- set header properties for mcfost
    primary_hdu.header['hierarch read_gas_velocity'] = 2
    primary_hdu.header['hierarch gas_dust_ratio'] = 100
    primary_hdu.header['hierarch read_gas_density'] = 1
    primary_hdu.header['read_n_a'] = 0

    # -- sink particles properties in header (code modifications needed for anything other than 2)
    primary_hdu.header['hierarch N_sink'] = Nsink

    # -- star properties in header
    primary_hdu.header['hierarch M_star'] = M          # star mass in M_sol
    primary_hdu.header['hierarch x_star'] = 0.         # star at origin
    primary_hdu.header['hierarch y_star'] = 0.
    primary_hdu.header['hierarch z_star'] = 0.
    primary_hdu.header['hierarch vx_star'] = 0.        # star not moving
    primary_hdu.header['hierarch vy_star'] = 0.
    primary_hdu.header['hierarch vz_star'] = 0.

    # -- planet properties in header
    primary_hdu.header['hierarch M_planet'] = Mp       # planet mass in M_jup
    primary_hdu.header['hierarch x_planet'] = Rplanet  # planet x position simply orbital radius [AU]
    primary_hdu.header['hierarch y_planet'] = 0.
    primary_hdu.header['hierarch z_planet'] = 0.
    primary_hdu.header['hierarch vx_planet'] = 0.      # planet velocity just keplerian rotation [m/s] (positive for anticlockwise, which is cw=-1)
    primary_hdu.header['hierarch vy_planet'] = np.sqrt(G*M*M_sol/(Rplanet*au)) / 100 * -cw
    primary_hdu.header['hierarch vz_planet'] = 0.

    # -- setup fits file
    hdul = fits.HDUList([primary_hdu, second_hdu, tertiary_hdu])

    # --- Write a fits file for mcfost
    if fitsname is not None:
        print("Writing ", fitsname, " to disk")
        hdul.writeto(fitsname + '.fits', overwrite=True)
    else:
        print("Fits file for mcfost was not created")

    return 'Done'

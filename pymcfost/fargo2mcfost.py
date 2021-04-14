import numpy as np
import astropy.io.fits as fits
import astropy.units as u
import sys
import subprocess
from parameters import Params
import parameters as p
from disc_structure import Disc


def fargo2mcfost(
    data_dir,
    dump_number,
    nz=5,
    mcfost_ref_file="/Users/cpinte/mcfost/src/ref3.0_3D.para",
    mcfost_filename="mcfost_FARGO.para",
    fitsname=None,
):

    # --- Need to read fargo .par file here, hard-coded for now
    # --- TODO : read Frame == C ou F  to detect is planet is rotating or not
    # --- if C : add keplrian velocity of planet
    # TODO : we need to read FACTORUNITMASS & FACTIORUNITLENGHT
    # -- MM
    Runit = 10.0
    Munit = 1.0  # star always has a mass == 1
    Frame = "C"
    Rmin = 4
    Rmax = 25

    # -- TODO : easiest if probably to read Rmin, Rmax, Nrad, Nsec here too
    # -- then we can skip dims.dat

    # --- Load dims.dat
    # This is an output file which contains for example the number of grid points
    try:
        dims = np.loadtxt(data_dir + '/dims.dat')
    except:
        sys.exit('Could not load dims.dat')
    N_t = int(dims[5])  # Number of outputs
    N_R = int(dims[6])  # Number of radial gridpoint
    N_theta = int(dims[7])  # Number of angular grid points

    # --- Load planet0.dat
    try:
        planet0 = np.loadtxt(data_dir + '/planet0.dat')
    except:
        sys.exit('Could not load planet0.dat!')
    dummy_i = -1

    # -- Todo : I have have no idea what those lines are doing
    # -- the test planet0[i, 0] == dummy_i is always false
    for i in range(planet0.shape[0]):
        if planet0[i, 0] == dummy_i:
            planet0[i - 1 : planet0.shape[0] - 1, :] = planet0[i:, :]
        dummy_i = planet0[i, 0]

    # Cartesian coordinates of the planet
    planet_x = planet0[dump_number, 1] * u.AU
    planet_y = planet0[dump_number, 2] * u.AU
    # Transform to polar coordinates
    planet_R = np.sqrt(planet_x ** 2 + planet_y ** 2)
    planet_theta = np.arctan2(planet_y, planet_x)

    # --- physical ctes
    grav = 39.4862194  # (au^3) / (solar mass * (yr^2))
    Timeunit = np.sqrt(Runit ** 3.0 / (grav * Munit))  # yr
    Lumunit = (
        Munit * Runit * Runit / (Timeunit * Timeunit * Timeunit)
    )  #  Msun*au^2*yr^-3
    Lsun = 2.7e-4  # Msun au^2/yr^3
    Lestrella = 1.0 * (Lsun / Lumunit)  #
    Rgas = (
        3.67e-4
    )  # //au^2/(yr^2 K)   R =0.000183508935; // R = 4124 [J/kg/K] = 0.000183508935 (au^2) / ((year^2) * kelvin)
    Tempunit = 2.35 * (grav * Munit / (Rgas * Runit))  # Kelvin
    densINcgs = 8888035.76  # 1 Msun/au^2 = 8888035.76 grams / (cm^2)
    DensUnit = densINcgs * Munit / Runit ** 2.0  # grams / (cm^2)
    QplusUnit = 1.6e13  # erg/s/cm^2
    vel_unit = Runit / Timeunit * 474057.172  # 1 au/yr = 474 057.172 cm/s

    # --- Does the planet rotate
    if Frame == "C":
        vtheta_planet = np.sqrt(grav * Munit / planet_R) * vel_unit
    else:
        vtheta_planet = 0.0

    # --- Reading data
    rho = np.fromfile(
        data_dir + "/gasdens{0:d}.dat".format(dump_number), dtype='float64'
    )
    temp = np.fromfile(
        data_dir + "/gasTemperature{0:d}.dat".format(dump_number), dtype='float64'
    )
    vrad = np.fromfile(
        data_dir + "/gasvrad{0:d}.dat".format(dump_number), dtype='float64'
    )
    vtheta = np.fromfile(
        data_dir + "/gasvtheta{0:d}.dat".format(dump_number), dtype='float64'
    )

    # --- Correct units + add velocity offset if needed
    rho_cgs = DensUnit * rho  # gr/cm^2
    temp_cgs = Tempunit * temp  # Kelvin
    vrad_cgs = vel_unit * vrad
    vtheta_cgs = vel_unit * vtheta + vtheta_planet.value

    # --- reshape
    Rho = DensUnit * rho.reshape(N_R, N_theta)
    Temp = Tempunit * temp.reshape(N_R, N_theta)

    # --- mcfost : updating values from FARGO
    # --- Setting up a parameter file and creating the corresponding grid
    P = Params(mcfost_ref_file)
    P.grid.n_rad = N_R
    P.grid.n_rad_in = 1
    P.grid.nz = nz
    P.grid.n_az = N_theta

    P.zones[0].Rin = Rmin
    P.zones[0].edge = 0.0
    P.zones[0].Rout = Rmax
    P.zones[0].Rc = 0.0

    # --- TODO : compute spectral type for given stellar mass
    # -- CP

    # -- Turn off symmetries
    P.simu.image_symmetry = False
    P.simu.central_symmetry = False
    P.simu.axial_symmetry = False

    # -- TODO : compute gas mass properly and put it in mcfost_FARGO.para
    # -- MM

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

    # -- same dimension order as FARGO
    mcfost_z = mcfost_disc.z()

    # --taking only half the grid (+z)
    mcfost_z = mcfost_z[:, nz + 1 :, :]
    mcfost_z = mcfost_z.transpose()  # dims are now, r, z, theta

    # -- TODO: defining H [au]
    # --- MM
    h = 0.05 * mcfost_disc.r()[:, 0, :].transpose()

    # -- computing the 3D density structure for mcfost
    rho_mcfost = Rho[:, np.newaxis, :] * np.exp(-0.5 * (mcfost_z / h[:, np.newaxis, :]))

    # --- Write a fits file for mcfost
    if fitsname is not None:
        print("Writing ", fitsname, " to disk")
        fits.writeto(fitsname, rho_mcfost.transpose(), overwrite=True)
        # --- TODO : add velocity : CP
    else:
        print("Fits file for mcfost was not created")

    return rho_mcfost


def analytic2mcfost(
    data_dir="/home/tom/Documents/Protoplanetary_Discs_Project/Analytical_Kinks/hd163_angletest",
    nz=50,
    analytic_params_file="/home/tom/Documents/Protoplanetary_Discs_Project/Analytical_Kinks/hd163_angletest/hd163.param",
    mcfost_ref_file="/home/tom/Documents/pymcfost/pymcfost/ref3.0_3D.para",
    mcfost_filename="mcfost_hd163_angletest.para",
    fitsname='hd163_analytic_angletest',
):

    p.analytic_params_import(analytic_params_file)

    # --- Reading data
    rho = np.load(
        data_dir + "/density.npy")

    print('after loading:', np.min(rho),np.max(rho))

    vrad = 1e3*np.load(
        data_dir + "/vr.npy")

    vtheta = 1e3*np.load(
        data_dir + "/vphi.npy")

    # --- reshape
    #Rho = DensUnit*rho.transpose()
    Rho = rho.transpose()
    Vrad = vrad.transpose()
    Vtheta = vtheta.transpose()

    # --- mcfost : updating values from analytic kinks parameter file
    # --- Setting up a parameter file and creating the corresponding grid
    P = Params(mcfost_ref_file)
    P.grid.n_rad = p.Nr
    P.grid.n_rad_in = 1
    P.grid.nz = nz
    P.grid.n_az = p.Nphi

    P.zones[0].Rin = 1.0
    P.zones[0].edge = 0.0
    P.zones[0].Rout = p.Rmax
    P.zones[0].Rc = 0.0

    # Don't compute SED
    P.simu.compute_SED = False

    P.map.nx = 1001 # pixel grid
    P.map.ny = 1001

    # ROTATION DIR. NEEDS TO BE CORRECT FOR THE FOLLOWING TO WORK

    P.map.RT_az_min = -1*p.PAp # 45, planet angle
    P.map.RT_az_max = -1*p.PAp
    P.map.RT_n_az = 1

    P.map.distance = 101.5 # distance

    P.map.PA = p.PA - 90 # PA

    P.map.RT_imin = -1*p.i # inclination
    P.map.RT_imax = -1*p.i
    P.map.RT_ntheta = 1

    P.phot.nphot_T = 1.28e+07
    P.mol.molecule[0].nv = 100 # number of velocity channels
    P.mol.molecule[0].n_trans = 1 # no. of lines in ray tracing

    # --- TODO : compute spectral type for given stellar mass
    # -- CP

    # -- Turn off symmetries
    P.simu.image_symmetry = False
    P.simu.central_symmetry = False
    P.simu.axial_symmetry = False

    # -- TODO : compute gas mass properly and put it in mcfost_FARGO.para
    # -- MM

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

    # -- same dimension order as FARGO
    mcfost_z = mcfost_disc.z()

    # --taking only half the grid (+z)
    #mcfost_z = mcfost_z[:, nz :, :]
    mcfost_z = mcfost_z[:, nz:, :]
    mcfost_z = mcfost_z.transpose()  # dims are now, r, z, theta

    # -- TODO: defining H [au]
    # --- MM
    h = 0.05 * mcfost_disc.r()[:, 0, :].transpose()

    # -- computing the 3D density structure for mcfost
    rho_mcfost = Rho[:, np.newaxis, :] * np.exp(-0.5 * (mcfost_z / h[:, np.newaxis, :]))
    vtheta_mcfost = np.repeat(Vtheta[:, np.newaxis, :], nz, axis=1)
    vrad_mcfost = np.repeat(Vrad[:, np.newaxis, :], nz, axis=1)
    vz_mcfost = np.zeros((220,50,200))

    velocities = np.array([vrad_mcfost.transpose(),vtheta_mcfost.transpose(),vz_mcfost.transpose()])
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
        hdul.writeto(fitsname + '.fits', overwrite=True)
        # --- TODO : add velocity : CP
    else:
        print("Fits file for mcfost was not created")

    return 'Done'






# Run with :
# import pymcfost as mcfost
# rho = mcfost.fargo2mcfost("/Users/cpinte/Downloads/4mcfost/",2,fitsname="test.fits")
# plot density with plt.imshow(rho[:,0,:])
# run mcfost with : mcfost_FARGO.para -df test.fits  # computes temperature & SED
#  mcfost_FARGO.para -df test.fits -img 1300 # creates an image at 1.3mm

import numpy as np
import matplotlib.pyplot as plt
from .image import Image
from .run import *
from .disc_structure import Disc
from astropy.io import fits
from scipy import constants
import os
import shutil
from scipy.ndimage import rotate
import scipy.interpolate
from .parameters import Params


def fit_sigma(para_file = None, fitsfile = None, distance = None, rmax = None, inc = 0.0, PA = -90.0, thresh=0.0, n_iter_max=10):
    # distance: distance to source in pc
    # rmax: maximum disk radius along major axis to include in fit (in au)
    # inc: disk inclinaiton in degrees
    # PA: disk PA in degrees east of north
    # thresh: threshold to use for masking data, in same units as fits file (default Jy/beam)
    # niter: number of iterations to run

    if para_file is None:
        raise ValueError("para_file is needed")

    if fitsfile is None:
        raise ValueError("fitsfile is needed")


    # Reading reference parameter file
    P  = Params(para_file)
    para_file_prefix = para_file.split('.para')[0]
    print(para_file_prefix)

    # Updating distance if needed
    if distance is None:
        distance = P.map.distance
    else:
        P.map.distance = distance

    # Reading data fits file
    fh = fits.open(fitsfile)
    header = fh[0].header

    bmaj = header['BMAJ'] * 3600.
    bmin = header['BMIN'] * 3600.
    BPA = header['BPA']

    beam_area = np.pi/(4.*np.log(2.))*bmaj*bmin

    pix_scale = header['CDELT2'] * 3600.
    pixel_area = pix_scale**2
    pix_scale_AU = pix_scale * distance

    # check order of axes - make sure CRVAL3 is freq and not stokes
    freq = header['CRVAL3']
    wl = constants.c/freq
    slambda = str(wl*1e6)
    print("wavelength=", slambda, "mum")

    image = fh[0].data[0, 0]
    # convert nan to zero
    image = np.nan_to_num(image, nan=0.0)
    fh.close()

    # Measure std and total flux in Jy
    std = np.nanstd([image[0:100,0:100],
                     image[0:100,-101:-1],
                     image[-101:-1,0:100],
                     image[-101:-1,-101:-1]
                     ])
    masked_image = np.ma.masked_where(image < 3*std, image)
    flux_Jy = np.sum(masked_image) * pixel_area/beam_area

    print("STD =", std, flux_Jy)

    # rotate image so that major axis is along x-axis
    # scipy.ndimage.rotate rotates image clockwise
    # this rotates image so that N side of majax in the positive x direction
    image = rotate(image, PA+90.0)

    # get center of image for now
    npix = image.shape[0]

    print(npix)

    center_x = int(npix/2)
    center_y = int(npix/2)

    # maximum radius where there is signal (provided by user or up to edge of image)
    if rmax is None:
        rmax = (int(npix/2)-1) * pix_scale_AU

    xmax = int((rmax/distance)/pix_scale)  # maximum x distance, in pixels

    x_profile = 1+np.arange(xmax) # distances along x axis in pixels, exclude center pixel
    x_profile_au = x_profile * pix_scale_AU

    # extract profile along x-axis (which should now be the major axis)
    profile_pos = image[center_y][center_x+1:center_x+xmax+1] # I values for positive x distances along majax
    profile_neg = image[center_y][center_x-xmax-1:center_x-1] # I values for negative x distances along majax
    # Average the 2 sides
    obs_profile = np.mean(np.array([profile_pos, np.flip(profile_neg)]), axis=0)

    obs_profile = obs_profile/beam_area # Convert to Jy/arcsec^2


    #mask = (image >= thresh).astype(int) # create mask that includes regions where we have signal in the data
    #npix = np.count_nonzero(mask) # number of pixels in mask
    #data_masked = image*mask # multiply data by mask


    # Updating parameter file pixelscale to the observed one
    # This makes it easier to have ratios and avoid an extra interpolation
    P.map.size= npix * pix_scale_AU
    P.map.nx = npix
    P.map.ny= P.map.nx

    # Turn off SED calculation
    P.simu.compute_SED = False

    P.simu.separate_contrib = False
    P.simu.separate_pola = False

    plt.figure(2)
    plt.clf()
    plt.figure(3)
    plt.clf()


    for i in range(n_iter_max):
        # Writing parameter file
        para_file = para_file_prefix+"_"+str(i)+'.para'
        P.writeto(para_file)

        # We create the directory before running mcfost to store the surface density file and keep things clean
        root_dir = f'iter_{i}'
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.makedirs(root_dir, exist_ok=True)
        options = "-root_dir "+root_dir

        # Surface density
        if i==0: # Reading surface density from initial model
            print('Running mcfost')
            run(para_file, options=options+" +disk_struct")
            run(para_file, options=options+' -casa -img '+slambda)
            print('Finished running mcfost')

            # Reading mcfost grid and initial surface density
            disk = Disc("iter_0/data_disk")
            r_mcfost = disk.r()[0][0]
            Sigma = disk.sigma()

        else: # Writing updated surface density

            Sigma *= correct_factor

            hdr = fits.Header()
            hdr['EXTNAME'] = 'IMAGE'
            fh = fits.PrimaryHDU(data=Sigma, header=hdr)
            sigma_file = os.path.join(root_dir,f'surface_density_{i}.fits.gz')
            fh.writeto(sigma_file, overwrite=True)
            options += ' -sigma '+sigma_file


            print('Running mcfost with'+options)
            run(para_file, options=options)
            run(para_file, options=options+' -casa -img '+slambda)
            print('Finished running mcfost')

        # Plotting surface density
        plt.figure(1)
        plt.plot(r_mcfost, Sigma)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('r (au)')
        plt.ylabel('Surface density (g cm$^{-2}$)')
        plt.savefig('sigma_'+str(i)+'.png')
        plt.close()

        # Reading model
        mod = Image(root_dir+"/data_"+slambda)

        max_change = 1.2

        # Flux_ratio
        mod_flux = np.sum(mod.image)
        flux_ratio = flux_Jy/mod_flux
        mass_ratio = np.minimum(np.maximum(flux_ratio,1/max_change),max_change)

        # Updating mass for next iteration
        P.zones[0].dust_mass *= mass_ratio

        plt.figure(2)
        plt.plot(i,mass_ratio,"o")

        plt.figure(3)
        plt.plot(i,mod_flux,"o")

        # SYnthetic profile
        profile_pos = mod.image[0,0,center_y,center_x+1:center_x+xmax+1]
        profile_neg = mod.image[0,0,center_y,center_x-xmax-1:center_x-1]

        mod_profile = np.mean(np.array([profile_pos, np.flip(profile_neg)]), axis=0)

        # Conversion from Jy/pix to Jy/arcsec**2
        mod_profile /= mod.pixelscale**2

        # Plotting brightness profiles
        plt.figure(1)
        plt.plot(x_profile, obs_profile, label='Observations')
        plt.plot(x_profile, mod_profile, label='mcfost')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('r (au)')
        plt.ylabel('I (Jy arcsec$^{-2}$)')
        plt.legend()
        plt.savefig('model_v_data_iteration_'+str(i)+'.png')
        plt.close()


        # ratio between obs and model (set to 1 if models do to 0)
        ratio = obs_profile/(mod_profile + (mod_profile < 1e-30) * obs_profile)

        # We just want the overall shape here as the normalisation is in the mass
        ratio -= np.mean(ratio)

        max_change = 3

        correct_factor = np.minimum(np.maximum(ratio,1/max_change),max_change)

        # We interpolate on the mcfost grid
        interp = scipy.interpolate.CubicSpline(x_profile_au, correct_factor)
        correct_factor = interp(r_mcfost)

        plt.figure(1)
        plt.plot(r_mcfost, correct_factor)
        plt.xscale('log')
        plt.ylabel('correction factor')
        plt.savefig('correct_factor_'+str(i)+'.png')
        plt.close()


    plt.figure(2)
    plt.ylabel('mass ratio')
    plt.xlabel('iteration')
    plt.savefig('mass_ratio.png')

    plt.figure(3)
    plt.ylabel('Flux [Jy]')
    plt.xlabel('iteration')
    plt.savefig('model_flux.png')

    #plt.plot(r_mcfost, CUT_data_interp, label='interp, rmcfost')
    #plt.plot(x_profile_au, CUT, label='no interp')
    #plt.legend()
    #plt.show()


    # We will do the mass corection later

    #Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    #mask_semimajor_au = rmax
    #mask_semimajor = mask_semimajor_au/pix_scale_AU
    #mask_semiminor = mask_semimajor*np.cos(np.deg2rad(inc))
    #Y = Y - center_y
    #X = X - center_x
    #ellipse_mask = ((X / mask_semimajor)**2 + (-Y / mask_semiminor)**2) <= 1
    #mask = ellipse_mask.astype(int)
    #data_masked = image*mask
    #npix = np.count_nonzero(mask)
    #print(npix)
    #pixel_area_arcsec = pix_scale**2
    #Integ_data = np.sum((data_masked/beam_area)*pixel_area_arcsec)
    #print(Integ_data)
    #
    #os.system('rm -r data_th_old')
    #os.system('rm -r data_'+slambda+'_old')
    #n_iter_max = niter
    #dust_masses = []
    #dust_mass_correction_factors = []



#
# # Rachel's code from here
#
#    for i in range(n_iter_max):
#        if i == 1:
#            # interpolate image values at radii in x_profile (in au) to radii in r_mcfost (also in au, but log spaced)
#            spl = scipy.interpolate.CubicSpline(x_profile_au, CUT)
#            Sigma = T_profile/1e6
#        else:
#            Sigma *= correct_factor
#
#        para_file = para_file_prefix+"_"+str(i)+'.para'
#        print(para_file)
#
#        print(i)
#
#        mydisk = Image("data_"+slambda)
#        mydisk.plot(type="I", bmaj=bmaj, bmin=bmin, bpa=PA, Jy=True)
#
#        plt.close()
#
#        model_pixsize_arcsec = mydisk.pixelscale
#        model_pixsize_au = model_pixsize_arcsec*distance
#        pix_area = model_pixsize_arcsec**2
#
#        model_Jy = mydisk.last_image/pix_area
#        model_center_x = int(1+mydisk.last_image.shape[1]/2)
#        model_center_y = int(1+mydisk.last_image.shape[0]/2)
#
#        xmax_model = int((rmax/distance)/model_pixsize_arcsec) # maximum x distance, in pixels
#        x_model_profile = np.linspace(1, xmax_model, xmax_model) # distances along x axis in pixels, exclude center pixel
#        x_model_profile_au = x_model_profile * model_pixsize_au
#        CUT_model = [model_Jy[model_center_y][x] for x in range(model_center_x+1, model_center_x+xmax_model+1)]
#        spl_model = scipy.interpolate.CubicSpline(x_model_profile_au, CUT_model)
#        CUT_model_interp = spl_model(r_mcfost)
#
#        correct_factor = [CUT_data_interp[j]/CUT_model_interp[j] if CUT_model_interp[j] > 0 else 1.0 for j in range(len(CUT_data_interp))]# already on r_mcfost grid
#
#
#
#        # if model is zero, set correct factor to 1
#        ratio = 1.2
#        correct_factor = [min(x, ratio) for x in correct_factor]
#        ratio2 = 0.8
#        correct_factor = [max(x, ratio2) for x in correct_factor]
#
#        semimajor_pix = mask_semimajor_au/model_pixsize_au # max extent of mask in model pixels
#        semiminor_pix = semimajor_pix*np.cos(np.deg2rad(41.))
#        Y, X = np.ogrid[:model_Jy.shape[0], :model_Jy.shape[1]]
#        Y = Y - model_center_y
#        X = X - model_center_x
#        ellipse_mask = ((X/semimajor_pix)**2 + (-Y/semiminor_pix)**2) <= 1
#        mask = ellipse_mask.astype(int)
#        model_masked = mydisk.last_image*mask
#        Integ_model = np.sum(model_masked)
#        print('model integrated flux: '+str(Integ_model))
#
#        with open(para_file, 'r') as file:
#            lines = file.readlines()
#        for line in enumerate(lines):
#            if 'gas-to-dust' in line:
#                parts = line.split()
#                dust_mass = float(parts[0])
#        for j, line in enumerate(lines):
#            if 'gas-to-dust' in line:
#                print('found dust mass line')
#                parts = line.split()
#                lines[j] = '    '.join(parts) + '\n'
#                print(lines[45])
#                dust_mass = float(parts[0])
#                dust_masses.append(dust_mass)
#                print(dust_mass)
#                break
#            #else:
#                #print('DID NOT find dust mass line')
#        print(dust_masses)
#        print('initial dust mass: '+str(dust_masses[i-1]))
#        flux_ratio = Integ_data/Integ_model
#        if flux_ratio > 1.1:
#            correction_dust_mass = 1.1
#        elif flux_ratio < 0.9:
#            correction_dust_mass = 0.9
#        else:
#            correction_dust_mass = flux_ratio
#        print('dust mass correction factor: '+str(correction_dust_mass))
#        dust_mass_corrected = dust_masses[i-1]*correction_dust_mass
#        dust_mass_correction_factors.append(correction_dust_mass)
#        print('corrected dust mass: '+str(dust_mass_corrected))
#
#        # Write the new content to new para file
#        new_para_file = para_file_prefix+str(i+1)+'.para'
#        #os.system('cp '+ para_file +' '+new_para_file)
#        with open(para_file_init, 'r') as file:
#            lines = file.readlines()
#        for j in range(len(lines)):
#            if 'dust mass' in lines[j]:
#                lines[j] = '  ' +str(dust_mass_corrected)+ '    100.    dust    mass,    gas-to-dust    mass    ratio \n'
#                print(lines[j])
#        with open(para_file_prefix+str(i+1)+'.para', 'w') as file:
#            file.writelines(lines)
#
#        # plot data vs. model brightness
#        plt.plot(r_mcfost, CUT_data_interp, label='data')
#        plt.plot(r_mcfost, CUT_model_interp, label='model, iteration '+str(i))
#        plt.xlabel('r (au)')
#        plt.ylabel('I (Jy arcsec$^{-2}$)')
#        plt.xscale('log')
#        plt.yscale('log')
#        plt.legend()
#        plt.savefig('model_v_data_iteration'+str(i)+'.png')
#        plt.close()
#
#        plt.plot(r_mcfost, correct_factor)
#        plt.xscale('log')
#        plt.yscale('log')
#        plt.axhline(y=1.0)
#        plt.ylabel('correction factor')
#        plt.title('iteration '+str(i))
#        plt.savefig('correction_factor_iter'+str(i)+'.png')
#        plt.close()
#
#        os.rename('data_th', 'data_th_iter'+str(i))
#        os.rename('data_'+slambda, 'data_'+slambda+'_iter'+str(i))
#
#        np.savetxt('dust_mass_correction_factors.txt', dust_mass_correction_factors)
#        plt.plot(dust_mass_correction_factors)
#        plt.xlabel('iteration')
#        plt.ylabel('dust mass correction factor')
#        plt.savefig('dust_mass_correction_factors.png')
#
#        np.savetxt('dust_masses.txt', dust_masses)
#        plt.plot(dust_masses)
#        plt.xlabel('iteration')
#        plt.ylabel('dust mass (Msun)')
#        plt.savefig('dust_masses.png')
#
#



def deconvolve_1d(profile, psf, deconv_threshold):
    """
    Deconvolve a 1D signal by a 1D PSF (point-spread function) without smoothing.

    profile            : 1D data array (your signal)
    psf                : 1D PSF/beam array (same length as profile)
    deconv_threshold   : threshold for noise in PSF’s FFT

    # Example:
    # raw_profile = np.loadtxt('profile.txt')
    # raw_psf     = np.loadtxt('psf.txt')
    # deconv, reconv = deconvolve_1d(raw_profile, raw_psf, deconv_threshold=0.01)

    """
    # Fourier transforms
    freq_profile = np.fft.fft(profile)
    freq_psf     = np.fft.fft(psf)

    # mask out frequencies where the PSF is too small (i.e. noisy divisions)
    valid = np.abs(freq_psf) > deconv_threshold * 0.1

    # allocate and perform division only on valid freq bins
    freq_deconv = np.zeros_like(freq_psf, dtype=complex)
    freq_deconv[valid] = freq_profile[valid] / freq_psf[valid]

    # back to real space and normalize
    deconv_profile = np.abs(np.fft.ifft(freq_deconv))
    deconv_profile /= deconv_profile.max()

    # (optional) reconvolve to check residuals
    freq_deconv_norm = np.fft.fft(deconv_profile)
    freq_reconv      = freq_deconv_norm * freq_psf
    reconvolved = np.abs(np.fft.ifft(freq_reconv))

    return deconv_profile, reconvolved

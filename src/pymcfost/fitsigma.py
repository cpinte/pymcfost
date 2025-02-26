import numpy as np
import matplotlib.pyplot as plt
from .image import Image
from .run import *
from .disc_structure import Disc
from astropy.io import fits
from scipy import constants
import os
from scipy.ndimage import rotate
import scipy.interpolate

def fitsigma(parafile = None, fitsfile = None, distance = None, rmax = None, inc = 0.0, PA = 0.0, thresh=0.0, niter=40):
    # distance: distance to source in pc
    # rmax: maximum disk radius along major axis to include in fit (in au)
    # inc: disk inclinaiton in degrees
    # PA: disk PA in degrees east of north
    # thresh: threshold to use for masking data, in same units as fits file (default Jy/beam)
    # niter: number of iterations to run 

    dir = os.getcwd()

    para_file_init = parafile
    para_file_prefix = para_file_init.split('1.para')[0]
    print(para_file_prefix)
    

    filename = fitsfile
    # read in parameters from fits file
    fh = fits.open(filename)
    header = fh[0].header

    bmaj = header['BMAJ'] * 3600.
    bmin = header['BMIN'] * 3600.
    PA = header['BPA']

    beam_area = np.pi/(4.*np.log(2.))*bmaj*bmin

    pix_scale = header['CDELT2'] * 3600.
    pix_scale_AU = pix_scale * distance

    # check order of axes - make sure CRVAL3 is freq and not stokes
    freq = header['CRVAL3']
    wl = constants.c/freq
    slambda = str(wl*1e6)
    print(slambda)

    image = fh[0].data[0, 0]
    print(np.max(image))
    # convert nan to zero
    image = np.nan_to_num(image, nan=0.0)
    print(np.max(image))
    # rotate image so that major axis is along x-axis
    # scipy.ndimage.rotate rotates image clockwise
    # this rotates image so that N side of majax in the positive x direction
    image = rotate(image, PA+90.0)
    print(np.max(image))
    # get center of image
    center = (int(image.shape[0]/2), int(image.shape[1]/2))# use center pixel of image
    center_x = center[1]
    center_y = center[0]

    xmax = int((rmax/distance)/pix_scale)  # maximum x distance, in pixels
    x_profile = np.linspace(1, xmax, xmax) # distances along x axis in pixels, exclude center pixel
    x_profile_au = x_profile * pix_scale_AU 
    profile_pos = np.array([image[center_y][i] for i in range(center_x+1, center_x+xmax+1)]) # I values for positive x distances along majax
    profile_neg = np.array([image[center_y][i] for i in range(center_x-xmax, center_x)]) # I values for negative x distances along majax
    profile = np.mean(np.array([profile_pos, np.flip(profile_neg)]), axis=0)
    CUT = profile/beam_area # Convert to Jy/arcsec^2

    mask = (image >= thresh).astype(int) # create mask that includes regions where we have signal in the data
    print(np.max(image))
    npix = np.count_nonzero(mask) # number of pixels in mask 
    print(npix)
    data_masked = image*mask # multiply data by mask    
    # compute disk structure
    run(parafile, options='-disk_struct')

    r_mcfost = Disc(dir+"/data_disk").r()[0][0]

    T_slope = -0.5
    T_profile = 150. * (r_mcfost/10)**T_slope

    # Plot initial sigma profile vs. cut along major axis

    spl_data = scipy.interpolate.CubicSpline(x_profile_au, CUT)
    CUT_data_interp = spl_data(r_mcfost)

    plt.plot(r_mcfost, CUT_data_interp, label='interp, rmcfost')
    plt.plot(x_profile_au, CUT, label='no interp')
    plt.legend()
    plt.show()

    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    mask_semimajor_au = rmax
    mask_semimajor = mask_semimajor_au/pix_scale_AU
    mask_semiminor = mask_semimajor*np.cos(np.deg2rad(inc))
    Y = Y - center_y
    X = X - center_x
    ellipse_mask = ((X / mask_semimajor)**2 + (-Y / mask_semiminor)**2) <= 1
    mask = ellipse_mask.astype(int)
    data_masked = image*mask
    npix = np.count_nonzero(mask) 
    print(npix)
    pixel_area_arcsec = pix_scale**2
    Integ_data = np.sum((data_masked/beam_area)*pixel_area_arcsec)
    print(Integ_data)

    os.system('rm -r data_th_old')
    os.system('rm -r data_'+slambda+'_old')
    n_iter_max = niter
    dust_masses = []
    dust_mass_correction_factors = []

    for i in range(1, n_iter_max):
        if i == 1:
            # interpolate image values at radii in x_profile (in au) to radii in r_mcfost (also in au, but log spaced)
            spl = scipy.interpolate.CubicSpline(x_profile_au, CUT)
            Sigma = T_profile/1e6
        else:
            Sigma *= correct_factor
        
        para_file = para_file_prefix+str(i)+'.para'
        print(para_file)
            
        print(i)
        fh = fits.PrimaryHDU(data = Sigma)
        fh.writeto('surface_density'+str(i)+'.fits.gz', overwrite=True)
        hdr = fits.open('surface_density'+str(i)+'.fits.gz')[0].header
        hdr['EXTNAME'] = 'IMAGE'
        fh = fits.PrimaryHDU(data = Sigma, header = hdr)
        fh.writeto('surface_density'+str(i)+'.fits.gz', overwrite=True)

        print('Running mcfost')
        
        os.system('rm -r data_th_iter'+str(i))
        options = options = '-sigma surface_density' + str(i) + '.fits.gz'
        run(para_file, options=options)
        os.system('rm -r data_'+slambda+'_iter'+str(i))
        options = '-sigma surface_density' + str(i) + '.fits.gz -img '+slambda
        run(para_file, options=options)

        print('Finished running mcfost')

        mydisk = Image("data_"+slambda)
        mydisk.plot(type="I", bmaj=bmaj, bmin=bmin, bpa=PA, Jy=True)
        
        plt.close()

        model_pixsize_arcsec = mydisk.pixelscale
        model_pixsize_au = model_pixsize_arcsec*distance
        pix_area = model_pixsize_arcsec**2

        model_Jy = mydisk.last_image/pix_area
        model_center_x = int(1+mydisk.last_image.shape[1]/2)
        model_center_y = int(1+mydisk.last_image.shape[0]/2)

        xmax_model = int((rmax/distance)/model_pixsize_arcsec) # maximum x distance, in pixels
        x_model_profile = np.linspace(1, xmax_model, xmax_model) # distances along x axis in pixels, exclude center pixel
        x_model_profile_au = x_model_profile * model_pixsize_au
        CUT_model = [model_Jy[model_center_y][x] for x in range(model_center_x+1, model_center_x+xmax_model+1)]
        spl_model = scipy.interpolate.CubicSpline(x_model_profile_au, CUT_model)
        CUT_model_interp = spl_model(r_mcfost)
        
        correct_factor = [CUT_data_interp[j]/CUT_model_interp[j] if CUT_model_interp[j] > 0 else 1.0 for j in range(len(CUT_data_interp))]# already on r_mcfost grid
        # if model is zero, set correct factor to 1
        ratio = 1.2
        correct_factor = [min(x, ratio) for x in correct_factor]
        ratio2 = 0.8
        correct_factor = [max(x, ratio2) for x in correct_factor]
        
        semimajor_pix = mask_semimajor_au/model_pixsize_au # max extent of mask in model pixels
        semiminor_pix = semimajor_pix*np.cos(np.deg2rad(41.))
        Y, X = np.ogrid[:model_Jy.shape[0], :model_Jy.shape[1]]
        Y = Y - model_center_y
        X = X - model_center_x
        ellipse_mask = ((X/semimajor_pix)**2 + (-Y/semiminor_pix)**2) <= 1
        mask = ellipse_mask.astype(int)
        model_masked = mydisk.last_image*mask
        Integ_model = np.sum(model_masked)
        print('model integrated flux: '+str(Integ_model))

        with open(para_file, 'r') as file:
            lines = file.readlines()
        for line in enumerate(lines):
            if 'gas-to-dust' in line:
                parts = line.split()
                dust_mass = float(parts[0]) 
        for j, line in enumerate(lines):
            if 'gas-to-dust' in line:
                print('found dust mass line')
                parts = line.split()
                lines[j] = '    '.join(parts) + '\n'
                print(lines[45])
                dust_mass = float(parts[0])
                dust_masses.append(dust_mass)
                print(dust_mass)
                break
            #else:
                #print('DID NOT find dust mass line')
        print(dust_masses)
        print('initial dust mass: '+str(dust_masses[i-1]))
        flux_ratio = Integ_data/Integ_model
        if flux_ratio > 1.1:
            correction_dust_mass = 1.1
        elif flux_ratio < 0.9:
            correction_dust_mass = 0.9
        else:
            correction_dust_mass = flux_ratio
        print('dust mass correction factor: '+str(correction_dust_mass))
        dust_mass_corrected = dust_masses[i-1]*correction_dust_mass
        dust_mass_correction_factors.append(correction_dust_mass)
        print('corrected dust mass: '+str(dust_mass_corrected))
    
        # Write the new content to new para file
        new_para_file = para_file_prefix+str(i+1)+'.para'
        #os.system('cp '+ para_file +' '+new_para_file)
        with open(para_file_init, 'r') as file:
            lines = file.readlines()
        for j in range(len(lines)):
            if 'dust mass' in lines[j]:
                lines[j] = '  ' +str(dust_mass_corrected)+ '    100.    dust    mass,    gas-to-dust    mass    ratio \n'
                print(lines[j])
        with open(para_file_prefix+str(i+1)+'.para', 'w') as file:
            file.writelines(lines)
        
        # plot data vs. model brightness 
        plt.plot(r_mcfost, CUT_data_interp, label='data')
        plt.plot(r_mcfost, CUT_model_interp, label='model, iteration '+str(i))
        plt.xlabel('r (au)')
        plt.ylabel('I (Jy arcsec$^{-2}$)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig('model_v_data_iteration'+str(i)+'.png')
        plt.close()
        
        plt.plot(r_mcfost, correct_factor)
        plt.xscale('log')
        plt.yscale('log')
        plt.axhline(y=1.0)
        plt.ylabel('correction factor')
        plt.title('iteration '+str(i))
        plt.savefig('correction_factor_iter'+str(i)+'.png')
        plt.close()

        os.rename('data_th', 'data_th_iter'+str(i))
        os.rename('data_'+slambda, 'data_'+slambda+'_iter'+str(i))

        np.savetxt('dust_mass_correction_factors.txt', dust_mass_correction_factors)
        plt.plot(dust_mass_correction_factors)
        plt.xlabel('iteration')
        plt.ylabel('dust mass correction factor')
        plt.savefig('dust_mass_correction_factors.png')

        np.savetxt('dust_masses.txt', dust_masses)
        plt.plot(dust_masses)
        plt.xlabel('iteration')
        plt.ylabel('dust mass (Msun)')
        plt.savefig('dust_masses.png')
        


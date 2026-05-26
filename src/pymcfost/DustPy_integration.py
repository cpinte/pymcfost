# =============================================================================
# DustPy to MCFOST PyMCFOST module developped by Angelino Blazère in May 2026
# This module is made to transform DustPy result data files into a format that MCFOST can read via a density file.
# ---IMPORTANT: THIS MODULE REQUIRES DUSTPY AND IT'S DEPENDENCIES TO FUNCTION---
# 
# It offers the following functionality:
#     A) make_sigma_fits: Creates a linear density fits file that MCFOST can read via it's -sigma_file option. !!! This does not yet work with multiple grains, and thus the function should just be avoided. !!!
#     B) make_density_x: Creates a 2D density fits file that MCFOST can read via it's -density_file option, taking into account the chosen vertical settling.
#     x can be respectively : no_settling, parametric_settling, Dubrulle_settling,Fromang_settling.
#     C) check_dustpy_MCFOST_correspondance_1D or 2D. This checks a fits file resulting from A) or B) with the obtained files from the -disk_struct command. Always run this to make sure you created
#     a correct MCFOST para file.
#     
# How to use:
#     1) Run your DustPy simulation, find the output folder path
#     2) Run A) or B) provinding this path followed by the number of data files you wish to convert, please note that B) functions will also ask
#     the number of vertical cells you wish to create. In the case of parametric_settling, you will be asked to input an array parametric_parameters, that contains
#     two values: [amix (microns), eta (no unit)].
#     If you wish to get assistance in building your MCFOST para file, add a 1 at the end of all make_density_x functions. This will output a 2D array containing for each data file a set of MCFOST parameters.
#     3) Run MCFOST using the option -sigma_file or -density_file, followed by the path of the fits file in question produced by this module, finally add to that
#     commande -disk_struct. MCFOST will produce a folder called data_disk, find it's path.
#     4) Run one of the C) functions, inputing the path of the dustpy fits file produced by this module, followed by the path of the data_disk folder.
#     5) If everything checks out, you can run your simulation in MCFOST without the -disk_struct option
# =============================================================================


'----------Dependencies----------'

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

'----------Utilities----------'

#This function calculates the dust scale height when parametric settling is considered
def height_parametric(pressure_scale_height,a,amix,eta_settl):
    return pressure_scale_height * (( a / amix ) ** eta_settl)

#This function calculates the density of a cell in the (r,Z) plane
def density(dust_midplane,Height,Z):
    return(
        dust_midplane * np.exp(-(Z ** 2) / (2 * (Height ** 2)))
    )

#This function calculates the density of a cell in the (r,Z) plane in the case of Fromang
def density_Fromang(dust_midplane,Height,D_tilde,Omega, Cs,rho_g,rho_s,a,Z):
    return(
        dust_midplane * np.exp(((- Omega * rho_s * a) / (rho_g * Cs * D_tilde)) * (np.exp((Z ** 2) / (2 * (Height ** 2)))  - 1) - ((Z ** 2) / (2 * (Height ** 2))))
    )

#This function normalizes the density calculated by the previous function in order to insure that the integral over the vertical direction of this density is equal to the linear density
def normalize_Fromang(dust_density_array,Sigma_array,Z_cscl,number_of_radial_cells,number_of_vertical_cells,number_of_size_bins):
    for size_cell in range (0,number_of_size_bins):
        for r_cell in range(0,number_of_radial_cells):
            norm=0
            for Z_cell in range (0,number_of_vertical_cells):
                norm+=2*Z_cscl[r_cell]*dust_density_array[size_cell,0,Z_cell,r_cell]
            if norm<1e-307: #This is the smallest value python can handle as a float. If lower, than we consider the grains are settled into the midplane.
                norm=Sigma_array[size_cell,0,0,r_cell]
                dust_density_array[size_cell,0,0,r_cell]=Sigma_array[size_cell,0,0,r_cell]/(2*Z_cscl[r_cell])
            dust_density_array[size_cell,0,:,r_cell]=dust_density_array[size_cell,0,:,r_cell] * (1 / (norm / Sigma_array[size_cell,0,0,r_cell]))
    return(dust_density_array)

#This function calculates the volume of an (r,Z) cell
def volume(ri_array,Z_cscl,number_of_vertical_cells,number_of_radial_cells):
    volume=np.zeros((number_of_vertical_cells,number_of_radial_cells))
    for Z_cell in range(0,number_of_vertical_cells):
        for r_cell in range(0,number_of_radial_cells):
            volume[Z_cell,r_cell]=Z_cscl[r_cell]*np.pi*(ri_array[r_cell+1]**2-ri_array[r_cell]**2)
    return(volume)

#This function calculates the gass and dust mass from a linear density
def linear_mass_intergrator(dust_sigma_array,gas_sigma_array,number_of_size_bins,r_array):
    dust_mass=0
    gas_mass=0
    grain_mass_array=np.zeros(number_of_size_bins)
    
    #Integration from 0 to max size
    for size_cell in range (0,number_of_size_bins):
        
        #Integration from 0 to r and 0 to 2 pi
        if size_cell==0: # The gas has only one "grain size"
            gas_mass = 2*np.pi * np.trapezoid(r_array*gas_sigma_array[0,0],r_array)
        grain_mass_array[size_cell] = 2*np.pi * np.trapezoid(r_array*dust_sigma_array[size_cell],r_array)
        dust_mass += grain_mass_array[size_cell]
    return (grain_mass_array,dust_mass,gas_mass)

#This function calculates the gass and dust mass from a 2D density
def mass_calculator(dust_density_array,gas_density_array,number_of_size_bins,number_of_vertical_cells,number_of_radial_cells,r_array,Z_array,Z_cscl,ri_array):
    dust_mass=0
    gas_mass=0
    grain_mass_array=np.zeros(number_of_size_bins)
    Volume=volume(ri_array,Z_cscl,number_of_vertical_cells,number_of_radial_cells)
    #Integration of gas
    #Integration from -Z to Z, from 0 to r, and  from 0 to 2 pi   
    for Z_cell in range(0,number_of_vertical_cells):
        for r_cell in range(0,number_of_radial_cells):
            gas_mass+=2*gas_density_array[0,Z_cell,r_cell]*Volume[Z_cell,r_cell] #We multiply by two the integration along 0 to Z to take into account the symetric -Z to 0
    
    #Integration from 0 to max size
    for size_cell in range (0,number_of_size_bins):
        #Integration from -Z to Z, from 0 to r, and  from 0 to 2 pi   
        for Z_cell in range(0,number_of_vertical_cells):
            for r_cell in range(0,number_of_radial_cells):
                grain_mass_array[size_cell]+=2*dust_density_array[size_cell,0,Z_cell,r_cell]*Volume[Z_cell,r_cell]
        dust_mass += grain_mass_array[size_cell]
    return (grain_mass_array,dust_mass,gas_mass)

#This function calculates the gass and dust mass from a 2D density
# def surface_mass_intergrator(dust_density_array,gas_density_array,number_of_size_bins,r_array,Z_array,nb_of_vertical_cells,nb_of_radial_cells,min_bin_size,max_bin_size):
#     dust_mass=0
#     gas_mass=0
#     grain_mass_array=np.zeros(number_of_size_bins)
    
#     #Integration from 0 to max size
#     for size_cell in range (0,number_of_size_bins):
#         #Integration from -Z to Z, from 0 to r, and  from 0 to 2 pi
#         if size_cell==0: # The gas has only one "grain size"
#             gas_mass = 2*np.pi * np.trapezoid(r_array * 2 * np.trapezoid(gas_density_array[0,:,:],Z_array,axis=0),r_array) #We multiply by two the integration along 0 to Z to take into account the symetric -Z to 0
#             # We can improve this integration by taking into account the fact that our Z and r values are at their bin center. We now do a simple rectangle numerical integration for half the lowest and half the highest bins.
#             for Z_cell in range (0,nb_of_vertical_cells):
#                 if Z_cell==0:
#                     gas_mass += (min_bin_size/2) * (r_array[0] - min_bin_size/2) * 2 * ((Z_array[1,0] - Z_array[0,0]) / 2) * gas_density_array[0,0,0]
#                 elif Z_cell==nb_of_vertical_cells-1:
#                     gas_mass += (min_bin_size/2) * (r_array[0] - min_bin_size/2) * 2 * ((Z_array[1,0] - Z_array[0,0]) / 2) * gas_density_array[0,nb_of_vertical_cells-1,0]
#                 else:
#                     gas_mass += 2 * ((Z_array[1,0] - Z_array[0,0]) / 2) * gas_density_array[0,Z_cell,0]
                    
#             for Z_cell in range (0,nb_of_vertical_cells):
#                 if Z_cell==0:
#                     gas_mass += (max_bin_size/2) * (r_array[nb_of_radial_cells-1] + min_bin_size/2) * 2 * ((Z_array[1,nb_of_radial_cells-1] - Z_array[0,nb_of_radial_cells-1]) / 2) * gas_density_array[0,0,nb_of_radial_cells-1]
#                 elif Z_cell==nb_of_vertical_cells-1:
#                     gas_mass += (max_bin_size/2) * (r_array[nb_of_radial_cells-1] + min_bin_size/2) * 2 * ((Z_array[1,nb_of_radial_cells-1] - Z_array[0,nb_of_radial_cells-1]) / 2) * gas_density_array[0,nb_of_vertical_cells-1,nb_of_radial_cells-1]
#                 else:
#                     gas_mass += 2 * ((Z_array[1,nb_of_radial_cells-1] - Z_array[0,nb_of_radial_cells-1]) / 2) * gas_density_array[0,nb_of_vertical_cells-1,nb_of_radial_cells-1]
#         grain_mass_array[size_cell] = 2*np.pi * np.trapezoid(r_array * 2 * np.trapezoid(dust_density_array[size_cell,0,:,:],Z_array,axis=0),r_array)
#         # We can improve this integration by taking into account the fact that our Z and r values are at their bin center. We now do a simple rectangle numerical integration for half the lowest and half the highest bins.
#         for Z_cell in range (0,nb_of_vertical_cells):
#             if Z_cell==0:
#                 grain_mass_array[size_cell] += (min_bin_size/2) * (r_array[0] - min_bin_size/2) * 2 * ((Z_array[1,0] - Z_array[0,0]) / 2) * dust_density_array[size_cell,0,0,0]
#             elif Z_cell==nb_of_vertical_cells-1:
#                 grain_mass_array[size_cell] += (min_bin_size/2) * (r_array[0] - min_bin_size/2) * 2 * ((Z_array[1,0] - Z_array[0,0]) / 2) * dust_density_array[size_cell,0,nb_of_vertical_cells-1,0]
#             else:
#                 grain_mass_array[size_cell] += 2 * ((Z_array[1,0] - Z_array[0,0]) / 2) * dust_density_array[size_cell,0,Z_cell,0]
                
#         for Z_cell in range (0,nb_of_vertical_cells):
#             if Z_cell==0:
#                 grain_mass_array[size_cell] += (max_bin_size/2) * (r_array[nb_of_radial_cells-1] + min_bin_size/2) * 2 * ((Z_array[1,nb_of_radial_cells-1] - Z_array[0,nb_of_radial_cells-1]) / 2) * dust_density_array[size_cell,0,0,nb_of_radial_cells-1]
#             elif Z_cell==nb_of_vertical_cells-1:
#                 grain_mass_array[size_cell] += (max_bin_size/2) * (r_array[nb_of_radial_cells-1] + min_bin_size/2) * 2 * ((Z_array[1,nb_of_radial_cells-1] - Z_array[0,nb_of_radial_cells-1]) / 2) * dust_density_array[size_cell,0,nb_of_vertical_cells-1,nb_of_radial_cells-1]
#             else:
#                 grain_mass_array[size_cell] += 2 * ((Z_array[1,nb_of_radial_cells-1] - Z_array[0,nb_of_radial_cells-1]) / 2) * dust_density_array[size_cell,0,nb_of_vertical_cells-1,nb_of_radial_cells-1]
#         dust_mass += grain_mass_array[size_cell]
#     return (grain_mass_array,dust_mass,gas_mass)
    
'----------MCFOST fits file creation functions----------'

#The following is a function to pass DustPy linear density into MCFOST and let MCFOST reconstruct the vertical density direction 
def make_sigma_fits(DustPy_data_folder :str,number_of_data_files:int,parameter_file_guide=0): #DOES NOT WORK AS MCFOST -SIGMA OPTION IS BROKEN FOR NOW

    #Initializing the reader module to the DustPy data folder
    from dustpy import hdf5writer
    wrtr = hdf5writer(DustPy_data_folder)
    if parameter_file_guide == 1:
        output = np.zeros((number_of_data_files,11))
    #Sequentially making sigma fits files for each data file, the data file must follow the default naming system of DustPy
    for data_file_number in range(0,number_of_data_files):
        print("--------Computing date file number "+str(data_file_number)+"--------")
        
        data=wrtr.read.output(data_file_number)
        mass_array=np.array(data.grid.m) #Gets the position of each mass bin center of DustPy grid
        r_array=np.array(data.grid.r) #Gets the position of each r bin center of DustPy grid
        number_of_radial_cells=r_array.shape[0] 
        
        '---Pressure scale height---'
        Hp_array=np.array(data.gas.Hp) #Gets the pressure scale height for each r cell
        
        '---Dust---'
        number_of_size_bins=data.grid.m.shape[0] #Getting the number of grain size/mass bins
        dust_sigma_array=np.array(data.dust.Sigma)*(1/10**4) #Opens Sigma array and converts from g/cm^2/cm to g/cm^2/micron
        dust_sigma_array=np.swapaxes(dust_sigma_array, 0, 1) #Changes the table from x axis = sizes and y axis = radius, to x axis = radius and y axis= sizes to correspond to MCFOST
        dust_sigma_array=np.reshape(dust_sigma_array,(number_of_size_bins,1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta,size)
        Grain_size=np.array(data.dust.a[0]*1e4) #Gets the sizes for an arbitrary line of the sizes file and converts to microns for MCFOST (!!! This may not be true if all grain sizes do not have the same fundamental density !!!)
        print("Attention! Found "+str(Grain_size.shape[0])+" dust grain sizes by assuming all dust grains have the same individual density regardless of size. This is wrong if this assumption is false!")
        
        '---Gas---'
        gas_sigma_array=np.array(data.gas.Sigma) #Opens Sigma array and divides by (mass bin width/mass bin center mass)
        gas_sigma_array=np.reshape(gas_sigma_array,(1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta)
        
        '---Mass---'
        Masses = linear_mass_intergrator(dust_sigma_array*10**4, gas_sigma_array, number_of_size_bins, r_array) #Multiplying by 10**4, to go back to g/cm^2/cm for simple integration
        print("Calculated a total dust mass of: "+str(Masses[1])+"g, and a total gas mass of: "+str(Masses[2])+"g. The dust to gas ratio is thus: "+str(Masses[1]/Masses[2]))
        distribution_array=Masses[0]/mass_array
        
        #The following block saves the Sigma array (1D density map), with Grain size as a secondary HDU. This can be used in MCFOST with -sigma_file option.
        '---MCFOST required headers---'
        dust_sigma_hdr = fits.Header()
        dust_sigma_hdr['UNIT'] = "g/cm^2/micron"
        dust_sigma_hdr['read_n_a']= 1
        dust_sigma_hdr['HIERARCH read_gas_density']= 1
        dust_sigma_hdr['HIERARCH read_gas_velocity']= 0 #We do not pass gas velocity to MCFOST here. MCFOST thus considers keplerian velocities.
        size_hdr = fits.Header()
        size_hdr['UNIT'] = "microns"
        distribution_hdr = fits.Header()
        distribution_hdr['UNIT'] = "Number of particles per size bin"
        gas_sigma_hdr = fits.Header()
        gas_sigma_hdr['UNIT'] = "g/cm^2"
        '---Bonus headers for this code---'
        mass_hdr = fits.Header()
        mass_hdr['UNIT'] = "grams"
        r_grid_hdr = fits.Header()
        r_grid_hdr['UNIT'] = "cm"
        '---MCFOST required HDUs---'
        dust_sigma_hdu = fits.PrimaryHDU(dust_sigma_array, header=dust_sigma_hdr)
        size_hdu = fits.ImageHDU(Grain_size,header=size_hdr)
        distribution_hdu = fits.ImageHDU(distribution_array,header=size_hdr)
        gas_sigma_hdu = fits.ImageHDU(gas_sigma_array,gas_sigma_hdr)
        '---Bonus HDUs for this code---'
        mass_hdu = fits.ImageHDU(mass_array,header=mass_hdr)
        r_grid_hdu = fits.ImageHDU(r_array,header=r_grid_hdr)
        '---Saving---'
        final = fits.HDUList([dust_sigma_hdu,size_hdu,distribution_hdu,gas_sigma_hdu,mass_hdu,r_grid_hdu]) #If one wanted to pass gas velocity, this information would need to be stored in the 5th HDU.
        final.writeto(DustPy_data_folder+"/data"+str(data_file_number)+"_sigma_file.fits",overwrite=True)
        
        if parameter_file_guide == 1:
            output[data_file_number,:]=[data.star.T,data.star.R, data.star.M, r_array[0],r_array[number_of_radial_cells-1],Masses[1],Masses[1]/Masses[2],Grain_size[0],Grain_size[number_of_size_bins-1], number_of_size_bins,number_of_radial_cells]
    if parameter_file_guide == 1:
        print("The following array will be outputed: [solar temperature, solar radius, solar mass, inner radius(cm), external radius(cm), dust mass(g),\n dust to gass ratio, min grain size(microns), max grain size(microns), number of grain size bins, number of radial cells]")
        return(output)
    
#The following is a function to extend DustPy linear density into the vertical direction considering no settling and pass it into MCFOST
def make_density_no_settling(DustPy_data_folder :str,number_of_data_files :int,number_of_vertical_cells :int, parameter_file_guide=0):
    
    #Initializing the reader module to the DustPy data folder
    from dustpy import hdf5writer
    wrtr = hdf5writer(DustPy_data_folder)
    if parameter_file_guide == 1:
        output = np.zeros((number_of_data_files,14))
    #Sequentially making sigma fits files for each data file, the data file must follow the default naming system of DustPy
    for data_file_number in range(0,number_of_data_files):
        print("--------Computing date file number "+str(data_file_number)+"--------")
        
        data=wrtr.read.output(data_file_number)
        mass_array=np.array(data.grid.m) #Gets the position of each mass bin center of DustPy grid
        r_array=np.array(data.grid.r) #Gets the position of each r bin center of DustPy grid
        ri_array=np.array(data.grid.ri) #Gets the position of each r bin edge of DustPy grid
        number_of_radial_cells=r_array.shape[0]
        
        '---Pressure scale height---'
        Hp_array=np.array(data.gas.Hp) #Gets the pressure scale height for each r cell
        
        '---Dust---'
        number_of_size_bins=data.grid.m.shape[0] #Getting the number of grain size/mass bins
        dust_midplane_array=np.array(data.dust.Sigma)*(1e-4)#Opens Sigma array and converts from g/cm^2/cm to g/cm^2/micron
        dust_midplane_array=np.swapaxes(dust_midplane_array, 0, 1) #Changes the table from x axis = sizes and y axis = radius, to x axis = radius and y axis= sizes to correspond to MCFOST
        dust_midplane_array=np.reshape(dust_midplane_array,(number_of_size_bins,1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta,size)
        dust_density_array=np.zeros((number_of_size_bins,1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map
        dust_density_array_particles=np.zeros((number_of_size_bins,1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map in particles
        Z_array=np.zeros((number_of_vertical_cells,number_of_radial_cells))
        Z_cscl=np.zeros(number_of_radial_cells)
        Grain_size=np.array(data.dust.a[0]*1e4) #Gets the sizes for an arbitrary line of the sizes file and converts to microns for MCFOST (!!! This may not be true if all grain sizes do not have the same fundamental density !!!)
        print("Attention! Found "+str(Grain_size.shape[0])+" dust grain sizes by assuming all dust grains have the same individual density regardless of size and radial distance. This is wrong if this assumption is false!")
        '-Dust density extension into the vertical dimension-'
        for size_cell in range (0,number_of_size_bins):
        
            for r_cell in range (0,number_of_radial_cells):
        
                Z_cscl[r_cell]=( Hp_array[r_cell] * 7) / number_of_vertical_cells #Defines the size of one vertical cell by deviding 7 times the local pressure scale height by the number of cells (Why 7 times? To match MCFOST)
                
                for Z_cell in range (1,number_of_vertical_cells+1):
                    Z_array[Z_cell-1,r_cell]=Z_cell*Z_cscl[r_cell]-0.5*Z_cscl[r_cell] #Calculates the vertical position in cm of the center of the cell
                    dust_density_array[size_cell,0,Z_cell-1,r_cell] = density(dust_midplane_array[size_cell,0,0,r_cell] / (np.sqrt(2*np.pi) * Hp_array[r_cell]),Hp_array[r_cell],Z_array[Z_cell-1,r_cell])
                    dust_density_array_particles[size_cell,0,Z_cell-1,r_cell] = dust_density_array[size_cell,0,Z_cell-1,r_cell] / mass_array[size_cell] #Changes from g/cm^3 to particles per cm^3 as this is what MCFOST expects
                    
        '---Gas---'
        gas_midplane_array=np.array(data.gas.rho) #Opens Sigma array and divides by (mass bin width/mass bin center mass)
        gas_midplane_array=np.reshape(gas_midplane_array,(1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta)
        gas_density_array=np.zeros((1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map
        '-Gas density extension into the vertical dimension-'
        for r_cell in range (0,number_of_radial_cells):
            
            Z_cscl[r_cell]=( Hp_array[r_cell] * 7) / number_of_vertical_cells #Defines the size of one vertical cell by deviding 7 times the local pressure scale height by the number of cells (Why 7 times? To match MCFOST)
            
            for Z_cell in range (1,number_of_vertical_cells+1):
                Z_array[Z_cell-1,r_cell]=Z_cell*Z_cscl[r_cell]-0.5*Z_cscl[r_cell] #Calculates the vertical position in cm of the center of the cell
                gas_density_array[0,Z_cell-1,r_cell] = density(gas_midplane_array[0,0,r_cell],Hp_array[r_cell],Z_array[Z_cell-1,r_cell])
                
        '---Mass---'
        Masses = mass_calculator(dust_density_array*1e4,gas_density_array,number_of_size_bins,number_of_vertical_cells,number_of_radial_cells,r_array,Z_array,Z_cscl,ri_array) #Multiplying by 1e4, to go back to g/cm^3/cm for simple integration
        print("Calculated a total dust mass of: "+str(Masses[1])+"g, and a total gas mass of: "+str(Masses[2])+"g. The dust to gas ratio is thus: "+str(Masses[1]/Masses[2]))
        distribution_array=Masses[0]/mass_array
        #The actual number of grains matters little to MCFOST. What's important is the proportions. Therefore we decreasse homogenously the number of grains, to avoid problems of simple/double precision in MCFOST's fortran code
        for value in range(0,number_of_size_bins): 
            if(distribution_array[value]>1e37):
                distribution_array=distribution_array*(1e37/np.max(distribution_array))
                break
            
        #The following block saves the Sigma array (1D density map), with Grain size as a secondary HDU. This can be used in MCFOST with -sigma_file option.
        '---MCFOST required headers---'
        dust_density_hdr = fits.Header()
        dust_density_hdr['UNIT'] = "particles/cm^3/micron"
        dust_density_hdr['read_n_a']= 1
        dust_density_hdr['HIERARCH read_gas_density']= 1
        dust_density_hdr['HIERARCH read_gas_velocity']= 0 #We do not pass gas velocity to MCFOST here. MCFOST thus considers keplerian velocities.
        size_hdr = fits.Header()
        size_hdr['UNIT'] = "microns"
        distribution_hdr = fits.Header()
        distribution_hdr['UNIT'] = "Number of particles per size bin"
        gas_density_hdr = fits.Header()
        gas_density_hdr['UNIT'] = "particles/cm^3"
        '---Bonus headers for this code---'
        mass_hdr = fits.Header()
        mass_hdr['UNIT'] = "grams"
        r_grid_hdr = fits.Header()
        r_grid_hdr['UNIT'] = "cm"
        Z_grid_hdr = fits.Header()
        Z_grid_hdr['UNIT'] = "cm"
        '---MCFOST required HDUs---'
        dust_density_hdu = fits.PrimaryHDU(dust_density_array_particles, header=dust_density_hdr)
        size_hdu = fits.ImageHDU(Grain_size,header=size_hdr)
        distribution_hdu = fits.ImageHDU(distribution_array,header=size_hdr)
        gas_density_hdu = fits.ImageHDU(((1e2)**3)*gas_density_array/3.84912530E-24,gas_density_hdr) #Converting to m^3 instead of cm^3, and from grams to gas particles as this is what MCFOST expects
        '---Bonus HDUs for this code---'
        mass_hdu = fits.ImageHDU(mass_array,header=mass_hdr)
        r_grid_hdu = fits.ImageHDU(r_array,header=r_grid_hdr)
        Z_grid_hdu = fits.ImageHDU(Z_array,header=Z_grid_hdr)
        '---Saving---'
        final = fits.HDUList([dust_density_hdu,size_hdu,distribution_hdu,gas_density_hdu,mass_hdu,r_grid_hdu, Z_grid_hdu]) #If one wanted to pass gas velocity, this information would need to be stored in the 5th HDU.
        final.writeto(DustPy_data_folder+"/data"+str(data_file_number)+"_density_no_settling_file.fits",overwrite=True)
        
        if parameter_file_guide == 1:
            output[data_file_number,:]=[data.star.T,data.star.R, data.star.M, r_array[0],r_array[number_of_radial_cells-1],Hp_array[number_of_radial_cells-1],5/4,Masses[1],Masses[1]/Masses[2],Grain_size[0],Grain_size[number_of_size_bins-1], number_of_size_bins,number_of_radial_cells,number_of_vertical_cells]
    if parameter_file_guide == 1:
        print("The following array will be outputed: [solar temperature, solar radius, solar mass, inner radius(cm), external radius(cm),pressure scale height at external radius (cm), flaring index (this is a DustPy constant), dust mass(g),\n dust to gass ratio, min grain size(microns), max grain size(microns), number of grain size bins, number of radial cells, number of vertical cells]")
        return(output)
    

#The following is a function to extend DustPy linear density into the vertical direction considering parametric settling and pass it into MCFOST
def make_density_parametric(DustPy_data_folder,number_of_data_files,number_of_vertical_cells,parametric_parameters,parameter_file_guide=0):
    
    '---Initialization---'
    #parametric_parameters is an array containing the two following values
    amix=parametric_parameters[0] #microns
    eta_settl=parametric_parameters[1] #No unit (usually negative)
    print("Read amix = "+str(amix)+" microns and eta = "+str(eta_settl))
    
    #Initializing the reader module to the DustPy data folder
    from dustpy import hdf5writer
    wrtr = hdf5writer(DustPy_data_folder)
    if parameter_file_guide == 1:
        output = np.zeros((number_of_data_files,14))
    
    #Sequentially making sigma fits files for each data file, the data file must follow the default naming system of DustPy
    for data_file_number in range(0,number_of_data_files):
        print("--------Computing date file number "+str(data_file_number)+"--------")
        
        data=wrtr.read.output(data_file_number)
        mass_array=np.array(data.grid.m) #Gets the position of each mass bin center of DustPy grid
        r_array=np.array(data.grid.r) #Gets the position of each r bin center of DustPy grid
        ri_array=np.array(data.grid.ri) #Gets the position of each r bin edge of DustPy grid
        number_of_radial_cells=r_array.shape[0]
        
        '---Scale height---'
        Hp_array=np.array(data.gas.Hp) #Gets the pressure scale height for each r cell
        Grain_size=np.array(data.dust.a[0]*1e4) #Gets the sizes for an arbitrary line of the sizes file and converts to microns for MCFOST (!!! This may not be true if all grain sizes do not have the same fundamental density !!!)
        print("Attention! Found "+str(Grain_size.shape[0])+" dust grain sizes by assuming all dust grains have the same individual density regardless of size and radial distance. This is wrong if this assumption is false!")
        
        '---Dust---'
        number_of_size_bins=data.grid.m.shape[0] #Getting the number of grain size/mass bins
        dust_midplane_array=np.array(data.dust.Sigma)*(1/10**4)#Opens Sigma array and converts from g/cm^2/cm to g/cm^2/micron
        dust_midplane_array=np.swapaxes(dust_midplane_array, 0, 1) #Changes the table from x axis = sizes and y axis = radius, to x axis = radius and y axis= sizes to correspond to MCFOST
        dust_midplane_array=np.reshape(dust_midplane_array,(number_of_size_bins,1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta,size)
        dust_density_array=np.zeros((number_of_size_bins,1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map
        dust_density_array_particles=np.zeros((number_of_size_bins,1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map in particles
        Z_array=np.zeros((number_of_vertical_cells,number_of_radial_cells))
        Z_cscl=np.zeros(number_of_radial_cells)
        '-Dust density extension into the vertical dimension-'
        for size_cell in range (0,number_of_size_bins):
            
            for r_cell in range (0,number_of_radial_cells):
                
                Z_cscl[r_cell]=( Hp_array[r_cell] * 7) / number_of_vertical_cells #Defines the size of one vertical cell by deviding 7 times the local pressure scale height by the number of cells (Why 7 times? To match MCFOST)
                H=height_parametric(Hp_array[r_cell], Grain_size[size_cell], amix, eta_settl) #Gets the dust scale height for each cell
                
                for Z_cell in range (1,number_of_vertical_cells+1):
                    Z_array[Z_cell-1,r_cell]=Z_cell*Z_cscl[r_cell]-0.5*Z_cscl[r_cell] #Calculates the vertical position in cm of the center of the cell
                    dust_density_array[size_cell,0,Z_cell-1,r_cell] = density(dust_midplane_array[size_cell,0,0,r_cell] / (np.sqrt(2*np.pi) * H),H,Z_array[Z_cell-1,r_cell])
                    dust_density_array_particles[size_cell,0,Z_cell-1,r_cell] = dust_density_array[size_cell,0,Z_cell-1,r_cell] / mass_array[size_cell] #Changes from g/cm^3 to particles per cm^3 as this is what MCFOST expects
                    
        '---Gas---'
        gas_midplane_array=np.array(data.gas.rho) #Opens Sigma array and divides by (mass bin width/mass bin center mass)
        gas_midplane_array=np.reshape(gas_midplane_array,(1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta)
        gas_density_array=np.zeros((1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map
        '-Gas density extension into the vertical dimension-'
        for r_cell in range (0,number_of_radial_cells):
            
            Z_cscl[r_cell]=( Hp_array[r_cell] * 7) / number_of_vertical_cells #Defines the size of one vertical cell by deviding 7 times the local pressure scale height by the number of cells (Why 7 times? To match MCFOST)
            
            for Z_cell in range (1,number_of_vertical_cells+1):
                Z_array[Z_cell-1,r_cell]=Z_cell*Z_cscl[r_cell]-0.5*Z_cscl[r_cell] #Calculates the vertical position in cm of the center of the cell
                gas_density_array[0,Z_cell-1,r_cell] = density(gas_midplane_array[0,0,r_cell],Hp_array[r_cell],Z_array[Z_cell-1,r_cell])
                
        '---Mass---'
        Masses = mass_calculator(dust_density_array*10**4,gas_density_array,number_of_size_bins,number_of_vertical_cells,number_of_radial_cells,r_array,Z_array,Z_cscl,ri_array) #Multiplying by 10**4, to go back to g/cm^3/cm for simple integration
        print("Calculated a total dust mass of: "+str(Masses[1])+"g, and a total gas mass of: "+str(Masses[2])+"g. The dust to gas ratio is thus: "+str(Masses[1]/Masses[2]))
        distribution_array=Masses[0]/mass_array
        for value in range(0,number_of_size_bins): 
            if(distribution_array[value]>1e37):
                distribution_array=distribution_array*(1e37/np.max(distribution_array))
                break

        #The following block saves the Sigma array (1D density map), with Grain size as a secondary HDU. This can be used in MCFOST with -sigma_file option.
        '---MCFOST required headers---'
        dust_density_hdr = fits.Header()
        dust_density_hdr['UNIT'] = "particles/cm^3/micron"
        dust_density_hdr['read_n_a']= 1
        dust_density_hdr['HIERARCH read_gas_density']= 1
        dust_density_hdr['HIERARCH read_gas_velocity']= 0 #We do not pass gas velocity to MCFOST here. MCFOST thus considers keplerian velocities.
        size_hdr = fits.Header()
        size_hdr['UNIT'] = "microns"
        distribution_hdr = fits.Header()
        distribution_hdr['UNIT'] = "Number of particles per size bin"
        gas_density_hdr = fits.Header()
        gas_density_hdr['UNIT'] = "particles/cm^3"
        '---Bonus headers for this code---'
        mass_hdr = fits.Header()
        mass_hdr['UNIT'] = "grams"
        r_grid_hdr = fits.Header()
        r_grid_hdr['UNIT'] = "cm"
        Z_grid_hdr = fits.Header()
        Z_grid_hdr['UNIT'] = "cm"
        '---MCFOST required HDUs---'
        dust_density_hdu = fits.PrimaryHDU(dust_density_array_particles, header=dust_density_hdr)
        size_hdu = fits.ImageHDU(Grain_size,header=size_hdr)
        distribution_hdu = fits.ImageHDU(distribution_array,header=size_hdr)
        gas_density_hdu = fits.ImageHDU(((1e2)**3)*gas_density_array/3.84912530E-24,gas_density_hdr) #Converting to m^3 instead of cm^3, and from grams to gas particles as this is what MCFOST expects
        '---Bonus HDUs for this code---'
        mass_hdu = fits.ImageHDU(mass_array,header=mass_hdr)
        r_grid_hdu = fits.ImageHDU(r_array,header=r_grid_hdr)
        Z_grid_hdu = fits.ImageHDU(Z_array,header=Z_grid_hdr)
        '---Saving---'
        final = fits.HDUList([dust_density_hdu,size_hdu,distribution_hdu,gas_density_hdu,mass_hdu,r_grid_hdu, Z_grid_hdu]) #If one wanted to pass gas velocity, this information would need to be stored in the 5th HDU.
        final.writeto(DustPy_data_folder+"/data"+str(data_file_number)+"_density_parametric_settling_file.fits",overwrite=True)
        
        if parameter_file_guide == 1:
            output[data_file_number,:]=[data.star.T,data.star.R, data.star.M, r_array[0],r_array[number_of_radial_cells-1],Hp_array[number_of_radial_cells-1],5/4,Masses[1],Masses[1]/Masses[2],Grain_size[0],Grain_size[number_of_size_bins-1], number_of_size_bins,number_of_radial_cells,number_of_vertical_cells]
    if parameter_file_guide == 1:
        print("The following array will be outputed: [solar temperature (K), solar radius (cm), solar mass (g), inner radius(cm), external radius(cm),pressure scale height at external radius (cm), flaring index (this is a DustPy constant), dust mass(g),\n dust to gas ratio, min grain size(microns), max grain size(microns), number of grain size bins, number of radial cells, number of vertical cells]")
        return(output)
        
#The following is a function to extend DustPy linear density into the vertical direction considering Dubrulle settling and pass it into MCFOST
def make_density_Dubrulle(DustPy_data_folder,number_of_data_files,number_of_vertical_cells,parameter_file_guide=0):
    
    #Initializing the reader module to the DustPy data folder
    from dustpy import hdf5writer
    wrtr = hdf5writer(DustPy_data_folder)
    if parameter_file_guide == 1:
        output = np.zeros((number_of_data_files,14))
    
    #Sequentially making sigma fits files for each data file, the data file must follow the default naming system of DustPy
    for data_file_number in range(0,number_of_data_files):
        print("--------Computing date file number "+str(data_file_number)+"--------")
        
        data=wrtr.read.output(data_file_number)
        mass_array=np.array(data.grid.m) #Gets the position of each mass bin center of DustPy grid
        r_array=np.array(data.grid.r) #Gets the position of each r bin center of DustPy grid
        ri_array=np.array(data.grid.ri) #Gets the position of each r bin edge of DustPy grid
        number_of_radial_cells=r_array.shape[0]
        
        '---Scale height---'
        Hp_array=np.array(data.gas.Hp) #Gets the pressure scale height for each r cell
        H_array=np.array(data.dust.H) #Gets the dust scale height for each r cell
        
        '---Dust---'
        number_of_size_bins=data.grid.m.shape[0] #Getting the number of grain size/mass bins
        dust_midplane_array=np.array(data.dust.rho)*(1/10**4)#Opens Sigma array and converts from g/cm^2/cm to g/cm^2/micron
        dust_midplane_array=np.swapaxes(dust_midplane_array, 0, 1) #Changes the table from x axis = sizes and y axis = radius, to x axis = radius and y axis= sizes to correspond to MCFOST
        dust_midplane_array=np.reshape(dust_midplane_array,(number_of_size_bins,1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta,size)
        dust_density_array=np.zeros((number_of_size_bins,1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map
        dust_density_array_particles=np.zeros((number_of_size_bins,1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map in particles
        Z_array=np.zeros((number_of_vertical_cells,number_of_radial_cells))
        Z_cscl=np.zeros(number_of_radial_cells)
        Grain_size=np.array(data.dust.a[0]*1e4) #Gets the sizes for an arbitrary line of the sizes file and converts to microns for MCFOST (!!! This may not be true if all grain sizes do not have the same fundamental density !!!)
        print("Attention! Found "+str(Grain_size.shape[0])+" dust grain sizes by assuming all dust grains have the same individual density regardless of size and radial distance. This is wrong if this assumption is false!")
        '-Dust density extension into the vertical dimension-'
        for size_cell in range (0,number_of_size_bins):
        
            for r_cell in range (0,number_of_radial_cells):
        
                Z_cscl[r_cell]=( Hp_array[r_cell] * 7) / number_of_vertical_cells #Defines the size of one vertical cell by deviding 7 times the local pressure scale height by the number of cells (Why 7 times? To match MCFOST)
                
                for Z_cell in range (1,number_of_vertical_cells+1):
                    Z_array[Z_cell-1,r_cell]=Z_cell*Z_cscl[r_cell]-0.5*Z_cscl[r_cell] #Calculates the vertical position in cm of the center of the cell
                    dust_density_array[size_cell,0,Z_cell-1,r_cell] = density(dust_midplane_array[size_cell,0,0,r_cell],H_array[r_cell,size_cell],Z_array[Z_cell-1,r_cell])
                    dust_density_array_particles[size_cell,0,Z_cell-1,r_cell] = dust_density_array[size_cell,0,Z_cell-1,r_cell] / mass_array[size_cell] #Changes from g/cm^3 to particles per cm^3 as this is what MCFOST expects
                    
        '---Gas---'
        gas_midplane_array=np.array(data.gas.rho) #Opens Sigma array and divides by (mass bin width/mass bin center mass)
        gas_midplane_array=np.reshape(gas_midplane_array,(1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta)
        gas_density_array=np.zeros((1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map
        '-Gas density extension into the vertical dimension-'
        for r_cell in range (0,number_of_radial_cells):
            
            Z_cscl[r_cell]=( Hp_array[r_cell] * 7) / number_of_vertical_cells #Defines the size of one vertical cell by deviding 7 times the local pressure scale height by the number of cells (Why 7 times? To match MCFOST)
            
            for Z_cell in range (1,number_of_vertical_cells+1):
                Z_array[Z_cell-1,r_cell]=Z_cell*Z_cscl[r_cell]-0.5*Z_cscl[r_cell] #Calculates the vertical position in cm of the center of the cell
                gas_density_array[0,Z_cell-1,r_cell] = density(gas_midplane_array[0,0,r_cell],Hp_array[r_cell],Z_array[Z_cell-1,r_cell])
                
        '---Mass---'
        Masses = mass_calculator(dust_density_array*10**4,gas_density_array,number_of_size_bins,number_of_vertical_cells,number_of_radial_cells,r_array,Z_array,Z_cscl,ri_array) #Multiplying by 10**4, to go back to g/cm^3/cm for simple integration
        print("Calculated a total dust mass of: "+str(Masses[1])+"g, and a total gas mass of: "+str(Masses[2])+"g. The dust to gas ratio is thus: "+str(Masses[1]/Masses[2]))
        distribution_array=Masses[0]/mass_array
        for value in range(0,number_of_size_bins): 
            if(distribution_array[value]>1e37):
                distribution_array=distribution_array*(1e37/np.max(distribution_array))
                break

        #The following block saves the Sigma array (1D density map), with Grain size as a secondary HDU. This can be used in MCFOST with -sigma_file option.
        '---MCFOST required headers---'
        dust_density_hdr = fits.Header()
        dust_density_hdr['UNIT'] = "particles/cm^3/micron"
        dust_density_hdr['read_n_a']= 1
        dust_density_hdr['HIERARCH read_gas_density']= 1
        dust_density_hdr['HIERARCH read_gas_velocity']= 0 #We do not pass gas velocity to MCFOST here. MCFOST thus considers keplerian velocities.
        size_hdr = fits.Header()
        size_hdr['UNIT'] = "microns"
        distribution_hdr = fits.Header()
        distribution_hdr['UNIT'] = "Number of particles per size bin"
        gas_density_hdr = fits.Header()
        gas_density_hdr['UNIT'] = "particles/cm^3"
        '---Bonus headers for this code---'
        mass_hdr = fits.Header()
        mass_hdr['UNIT'] = "grams"
        r_grid_hdr = fits.Header()
        r_grid_hdr['UNIT'] = "cm"
        Z_grid_hdr = fits.Header()
        Z_grid_hdr['UNIT'] = "cm"
        '---MCFOST required HDUs---'
        dust_density_hdu = fits.PrimaryHDU(dust_density_array_particles, header=dust_density_hdr)
        size_hdu = fits.ImageHDU(Grain_size,header=size_hdr)
        distribution_hdu = fits.ImageHDU(distribution_array,header=size_hdr)
        gas_density_hdu = fits.ImageHDU(((1e2)**3)*gas_density_array/3.84912530E-24,gas_density_hdr) #Converting to m^3 instead of cm^3, and from grams to gas particles as this is what MCFOST expects
        '---Bonus HDUs for this code---'
        mass_hdu = fits.ImageHDU(mass_array,header=mass_hdr)
        r_grid_hdu = fits.ImageHDU(r_array,header=r_grid_hdr)
        Z_grid_hdu = fits.ImageHDU(Z_array,header=Z_grid_hdr)
        '---Saving---'
        final = fits.HDUList([dust_density_hdu,size_hdu,distribution_hdu,gas_density_hdu,mass_hdu,r_grid_hdu, Z_grid_hdu]) #If one wanted to pass gas velocity, this information would need to be stored in the 5th HDU.
        final.writeto(DustPy_data_folder+"/data"+str(data_file_number)+"_density_Dubrulle_settling_file.fits",overwrite=True)
        
        if parameter_file_guide == 1:
            output[data_file_number,:]=[data.star.T,data.star.R, data.star.M, r_array[0],r_array[number_of_radial_cells-1],Hp_array[number_of_radial_cells-1],5/4,Masses[1],Masses[1]/Masses[2],Grain_size[0],Grain_size[number_of_size_bins-1], number_of_size_bins,number_of_radial_cells,number_of_vertical_cells]
    if parameter_file_guide == 1:
        print("The following array will be outputed: [solar temperature, solar radius, solar mass, inner radius(cm), external radius(cm),pressure scale height at external radius (cm), flaring index (this is a DustPy constant), dust mass(g),\n dust to gass ratio, min grain size(microns), max grain size(microns), number of grain size bins, number of radial cells, number of vertical cells]")
        return(output)
        
#The following is a function to extend DustPy linear density into the vertical direction considering Fromang settling and pass it into MCFOST
def make_density_Fromang(DustPy_data_folder,number_of_data_files,number_of_vertical_cells,parameter_file_guide=0):
    
    #Initializing the reader module to the DustPy data folder
    from dustpy import hdf5writer
    wrtr = hdf5writer(DustPy_data_folder)
    if parameter_file_guide == 1:
        output = np.zeros((number_of_data_files,14))
    
    #Sequentially making sigma fits files for each data file, the data file must follow the default naming system of DustPy
    for data_file_number in range(0,number_of_data_files):
        print("--------Computing date file number "+str(data_file_number)+"--------")
        
        data=wrtr.read.output(data_file_number)
        mass_array=np.array(data.grid.m) #Gets the position of each mass bin center of DustPy grid
        number_of_size_bins=data.grid.m.shape[0] #Getting the number of grain size/mass bins
        r_array=np.array(data.grid.r) #Gets the position of each r bin center of DustPy grid
        ri_array=np.array(data.grid.ri) #Gets the position of each r bin edge of DustPy grid
        number_of_radial_cells=r_array.shape[0]
        alpha_array=np.array(data.dust.delta.vert)
        # Stokes_array=np.array(data.dust.St)
        Omega_array=np.array(data.grid.OmegaK) #Gets the Omega K of each r bin of DustPy grid
        Cs_array=np.array(data.gas.cs) #Gets the gas sound speed of each r bin of DustPy grid
        
        '---Scale height---'
        Hp_array=np.array(data.gas.Hp) #Gets the pressure scale height for each r cell
                    
        '---Gas---'
        Z_array=np.zeros((number_of_vertical_cells,number_of_radial_cells))
        Z_cscl=np.zeros(number_of_radial_cells)
        gas_midplane_array=np.array(data.gas.rho) #Opens Sigma array and divides by (mass bin width/mass bin center mass)
        gas_midplane_array=np.reshape(gas_midplane_array,(1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta)
        gas_density_array=np.zeros((1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map
        '-Gas density extension into the vertical dimension-'
        for r_cell in range (0,number_of_radial_cells):
            
            Z_cscl[r_cell]=( Hp_array[r_cell] * 7) / number_of_vertical_cells #Defines the size of one vertical cell by deviding 7 times the local pressure scale height by the number of cells (Why 7 times? To match MCFOST)
            
            for Z_cell in range (1,number_of_vertical_cells+1):
                Z_array[Z_cell-1,r_cell]=Z_cell*Z_cscl[r_cell]-0.5*Z_cscl[r_cell] #Calculates the vertical position in cm of the center of the cell
                gas_density_array[0,Z_cell-1,r_cell] = density(gas_midplane_array[0,0,r_cell],Hp_array[r_cell],Z_array[Z_cell-1,r_cell])
        
        '---Dust---'
        Sigma_array=np.array(data.dust.Sigma)*(1/10**4)#Opens Sigma array and converts from g/cm^2/cm to g/cm^2/micron
        Sigma_array=np.swapaxes(Sigma_array, 0, 1) #Changes the table from x axis = sizes and y axis = radius, to x axis = radius and y axis= sizes to correspond to MCFOST
        Sigma_array=np.reshape(Sigma_array,(number_of_size_bins,1,1,number_of_radial_cells)) #Changes the array shape to match MCFOST requirements (r,Z,theta,size)
        dust_density_array=np.zeros((number_of_size_bins,1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map
        dust_density_array_particles=np.zeros((number_of_size_bins,1,number_of_vertical_cells,number_of_radial_cells)) #Builds the array to store the 2D density map in particles
        Grain_size=np.array(data.dust.a[0]*1e4) #Gets the sizes for an arbitrary line of the sizes file and converts to microns for MCFOST (!!! This may not be true if all grain sizes do not have the same fundamental density !!!)
        rho_s=float(data.dust.rhos[0,0]) #Gets individual density of grains
        print("Attention! Found "+str(Grain_size.shape[0])+" dust grain sizes by assuming all dust grains have the same individual density regardless of size and radial distance. This is wrong if this assumption is false!")
        '-Dust density extension into the vertical dimension-'
        for size_cell in range (0,number_of_size_bins):
        
            for r_cell in range (0,number_of_radial_cells):
        
                Z_cscl[r_cell]=( Hp_array[r_cell] * 7) / number_of_vertical_cells #Defines the size of one vertical cell by deviding 7 times the local pressure scale height by the number of cells (Why 7 times? To match MCFOST)
                
                for Z_cell in range (1,number_of_vertical_cells+1):
                    Z_array[Z_cell-1,r_cell]=Z_cell*Z_cscl[r_cell]-0.5*Z_cscl[r_cell] #Calculates the vertical position in cm of the center of the cell
                    #D_tilde=alpha_array[r_cell]/(1+Stokes_array[r_cell,size_cell]**2) #We adopt the expression D_tilde=alpha/Sc from Fromang and Nelson 2009, where Sc is the Schimdt number calculated as Sc=1+St, where St is the Stokes number, as in C. P. Dullemond and C. Dominik 2004
                    D_tilde=alpha_array[r_cell]/1.5
                    dust_density_array[size_cell,0,Z_cell-1,r_cell] = density_Fromang(Sigma_array[size_cell,0,0,r_cell], Hp_array[r_cell], D_tilde, Omega_array[r_cell], Cs_array[r_cell], gas_midplane_array[0,0,r_cell], rho_s, Grain_size[size_cell]*1e-4, Z_array[Z_cell-1,r_cell])
        dust_density_array=normalize_Fromang(dust_density_array,Sigma_array,Z_cscl,number_of_radial_cells,number_of_vertical_cells,number_of_size_bins)
        for size_cell in range (0,number_of_size_bins): #Changes from g/cm^3 to particles per cm^3 as this is what MCFOST expects
        
            for r_cell in range (0,number_of_radial_cells):
                
                for Z_cell in range (1,number_of_vertical_cells+1):
                        dust_density_array_particles[size_cell,0,Z_cell-1,r_cell] = dust_density_array[size_cell,0,Z_cell-1,r_cell] / mass_array[size_cell]
        
        '---Mass---'
        Masses = mass_calculator(dust_density_array*10**4,gas_density_array,number_of_size_bins,number_of_vertical_cells,number_of_radial_cells,r_array,Z_array,Z_cscl,ri_array) #Multiplying by 10**4, to go back to g/cm^3/cm for simple integration
        print("Calculated a total dust mass of: "+str(Masses[1])+"g, and a total gas mass of: "+str(Masses[2])+"g. The dust to gas ratio is thus: "+str(Masses[1]/Masses[2]))
        distribution_array=Masses[0]/mass_array
        for value in range(0,number_of_size_bins): 
            if(distribution_array[value]>1e37):
                distribution_array=distribution_array*(1e37/np.max(distribution_array))
                break

        #The following block saves the Sigma array (1D density map), with Grain size as a secondary HDU. This can be used in MCFOST with -sigma_file option.
        '---MCFOST required headers---'
        dust_density_hdr = fits.Header()
        dust_density_hdr['UNIT'] = "particles/cm^3/micron"
        dust_density_hdr['read_n_a']= 1
        dust_density_hdr['HIERARCH read_gas_density']= 1
        dust_density_hdr['HIERARCH read_gas_velocity']= 0 #We do not pass gas velocity to MCFOST here. MCFOST thus considers keplerian velocities.
        size_hdr = fits.Header()
        size_hdr['UNIT'] = "microns"
        distribution_hdr = fits.Header()
        distribution_hdr['UNIT'] = "Number of particles per size bin"
        gas_density_hdr = fits.Header()
        gas_density_hdr['UNIT'] = "particles/cm^3"
        '---Bonus headers for this code---'
        mass_hdr = fits.Header()
        mass_hdr['UNIT'] = "grams"
        r_grid_hdr = fits.Header()
        r_grid_hdr['UNIT'] = "cm"
        Z_grid_hdr = fits.Header()
        Z_grid_hdr['UNIT'] = "cm"
        '---MCFOST required HDUs---'
        dust_density_hdu = fits.PrimaryHDU(dust_density_array_particles, header=dust_density_hdr)
        size_hdu = fits.ImageHDU(Grain_size,header=size_hdr)
        distribution_hdu = fits.ImageHDU(distribution_array,header=size_hdr)
        gas_density_hdu = fits.ImageHDU(((1e2)**3)*gas_density_array/3.84912530E-24,gas_density_hdr) #Converting to m^3 instead of cm^3, and from grams to gas particles as this is what MCFOST expects
        '---Bonus HDUs for this code---'
        mass_hdu = fits.ImageHDU(mass_array,header=mass_hdr)
        r_grid_hdu = fits.ImageHDU(r_array,header=r_grid_hdr)
        Z_grid_hdu = fits.ImageHDU(Z_array,header=Z_grid_hdr)
        '---Saving---'
        final = fits.HDUList([dust_density_hdu,size_hdu,distribution_hdu,gas_density_hdu,mass_hdu,r_grid_hdu, Z_grid_hdu]) #If one wanted to pass gas velocity, this information would need to be stored in the 5th HDU.
        final.writeto(DustPy_data_folder+"/data"+str(data_file_number)+"_density_Fromang_settling_file.fits",overwrite=True)
        
        if parameter_file_guide == 1:
            output[data_file_number,:]=[data.star.T,data.star.R, data.star.M, r_array[0],r_array[number_of_radial_cells-1],Hp_array[number_of_radial_cells-1],5/4,Masses[1],Masses[1]/Masses[2],Grain_size[0],Grain_size[number_of_size_bins-1], number_of_size_bins,number_of_radial_cells,number_of_vertical_cells]
    if parameter_file_guide == 1:
        print("The following array will be outputed: [solar temperature, solar radius, solar mass, inner radius(cm), external radius(cm),pressure scale height at external radius (cm), flaring index (this is a DustPy constant), dust mass(g),\n dust to gass ratio, min grain size(microns), max grain size(microns), number of grain size bins, number of radial cells, number of vertical cells]")
        return(output)

'----------MCFOST fits file tests----------'
        
def check_dustpy_MCFOST_correspondance_1D(sigma_file_path,MCFOST_data_disk_folder_path):
    
    MCFOST_file = fits.open(MCFOST_data_disk_folder_path+"/grid.fits.gz")[0].data
    DustPy_file = fits.open(sigma_file_path)[5].data
    
    print("Checking the correspondance of the simulation grid...")
    print("Checking the radial dimension... ")
    
    plt.plot(DustPy_file, label="DustPy radial grid")
    plt.plot(MCFOST_file[0,0,0,:],label="MCFOST radial grid")
    plt.xlabel("Cell number")
    plt.ylabel("Radial value in AU")
    plt.yscale("log")
    plt.legend()
    plt.show()
    print("Tip : if these do not match, check if n_rad_in is 0 in your parameter file. A value =! from 0 leads to a departure from a log grid in MCFOST.")
    
    print("Checking the correspondance of the grain sizes...")
    
    MCFOST_file = fits.open(MCFOST_data_disk_folder_path+"/grain_sizes.fits.gz")[0].data
    DustPy_file = fits.open(sigma_file_path)[1].data
    plt.plot(DustPy_file, label="DustPy grain sizes")
    plt.plot(MCFOST_file,label="MCFOST grain sizes")
    plt.xlabel("Size bin")
    plt.ylabel("Size in microns")
    plt.yscale("log")
    plt.legend()
    plt.show()
    
    print("Checking the correspondance of the grain masses...")
    
    MCFOST_file = fits.open(MCFOST_data_disk_folder_path+"/grain_masses.fits.gz")[0].data
    DustPy_file = fits.open(sigma_file_path)[4].data
    plt.plot(DustPy_file, label="DustPy grain masses")
    plt.plot(MCFOST_file,label="MCFOST grain masses")
    plt.xlabel("Mass bin")
    plt.ylabel("Mass in microns")
    plt.yscale("log")
    plt.legend()
    plt.show()
    print("Tip : if these do not match, check that the material you are using in MCFOST parameter file has the same density as your grains in DustPy.")
    
def check_dustpy_MCFOST_correspondance_2D(density_file_path,MCFOST_data_disk_folder_path):
    
    MCFOST_file = fits.open(MCFOST_data_disk_folder_path+"/grid.fits.gz")[0].data
    DustPy_file = fits.open(density_file_path)[5].data
    
    print(">>>Checking the correspondance of the simulation grid...")
    print(">>>Checking the radial dimension... ")
    
    plt.plot(DustPy_file*6.68459e-14, label="DustPy radial grid")
    plt.plot(MCFOST_file[0,0,0,:],label="MCFOST radial grid")
    plt.xlabel("Cell number")
    plt.ylabel("Radial value in AU")
    plt.yscale("log")
    plt.legend()
    plt.show()
    print("Tip : if these do not match, check if n_rad_in is 0 in your parameter file. A value =! from 0 leads to a departure from a log grid in MCFOST.\n Do note that to run MCFOST images and avoid diffusion approximation, you will have to make this value =! 0, but do so without impacting the passage of the density file.\n This will depend on how large n_rad_in is (8 should be a max), please control with residual maps below, that no significant patern emerges.\n If despite this you still have the diffusion approximation, you will need to refine your DustPy grid. ")
    
    print(">>>Checking the vertical dimension... ")
    
    DustPy_file = fits.open(density_file_path)[6].data
    
    plt.figure(figsize=(80,80))
    for i in range (0,len(DustPy_file[0,:])):
        plt.subplot(11,round(len(DustPy_file[0,:])/10)+1,i+1)
        plt.plot(DustPy_file[:,i]*6.68459e-14,'-.',label="DustPy vertical grid for r cell="+str(i+1))
        plt.plot(MCFOST_file[1,0,:,i],label="MCFOST vertical grid for r cell="+str(i+1))
        plt.xlabel("Cell number")
        plt.ylabel("Radial value in AU")
        plt.legend()
    plt.show()
    print("Tip : if these do not match, make sure you have entered a correct reference scale height and radius as well as flaring index.\n If you have made n_rad_in =! 0, to avoid diffusion approximation, these will not perfectly match.")
    
    print(">>>Checking the correspondance of the grain sizes...")
    
    MCFOST_file = fits.open(MCFOST_data_disk_folder_path+"/grain_sizes.fits.gz")[0].data
    DustPy_file = fits.open(density_file_path)[1].data
    plt.plot(DustPy_file, label="DustPy grain sizes")
    plt.plot(MCFOST_file,label="MCFOST grain sizes")
    plt.xlabel("Size bin")
    plt.ylabel("Size in microns")
    plt.yscale("log")
    plt.legend()
    plt.show()
    
    print(">>>Checking the correspondance of the grain masses...")
    
    MCFOST_file = fits.open(MCFOST_data_disk_folder_path+"/grain_masses.fits.gz")[0].data
    DustPy_file = fits.open(density_file_path)[4].data
    plt.plot(DustPy_file, label="DustPy grain masses")
    plt.plot(MCFOST_file,label="MCFOST grain masses")
    plt.xlabel("Mass bin")
    plt.ylabel("Mass in grams")
    plt.yscale("log")
    plt.legend()
    plt.show()
    print("Tip : if these do not match, check that the material you are using in MCFOST parameter file has the same density as your grains in DustPy.")
    
    print(">>>Checking the correspondance of the gas density file...")
    MCFOST_file = fits.open(MCFOST_data_disk_folder_path+"/gas_density.fits.gz")[0].data
    DustPy_file = fits.open(density_file_path)[3].data
    
    plt.imshow(MCFOST_file[:,:]/np.max(MCFOST_file[:,:])-DustPy_file[0,:,:]/np.max(DustPy_file[0,:,:]),origin='lower')
    plt.xlabel("r bins")
    plt.ylabel("Z bins")
    plt.title("Residual of MCFOST gas density file - gas density file\nmade by this code at size bin ")
    plt.colorbar()
    plt.show()
    print("Tip : if the redsidual is one over the vast majority of the image, this probably means that the density was so low that it fell below MCFOST's single precision.\n This is a fine approximation as the lower limit for single precision is about 1e-40 part.m^-3 [per grain size bin N(a).da] ")
    
    print(">>>Checking the correspondance of the dust density file...")
    MCFOST_file = fits.open(MCFOST_data_disk_folder_path+"/dust_particle_density.fits.gz")[0].data
    DustPy_file = fits.open(density_file_path)[0].data
    
    plt.figure(figsize=(80,80))
    for i in range (0,len(DustPy_file[:,0,0,0])):
        plt.subplot(10,round(len(DustPy_file[:,0,0,0])/10)+1,i+1)
        plt.imshow(MCFOST_file[i,:,:]/np.max(MCFOST_file[i,:,:])-DustPy_file[i,0,:,:]/np.max(DustPy_file[i,0,:,:]),origin='lower')
        plt.xlabel("r bins")
        plt.ylabel("Z bins")
        plt.title("Residual of MCFOST dust density file - dust density file\nmade by this code at size bin "+str(i))
        plt.colorbar()
    plt.show()
    print("Tip : if after a certain size bin the redsidual is one over the vast majority of the image, this probably means that the density was so low that it fell below MCFOST's single precision.\n This is a fine approximation as the lower limit for single precision is about 1e-40 part.m^-3 [per grain size bin N(a).da] ")

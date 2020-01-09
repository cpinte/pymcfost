import glob
import numpy as np


class Photons:
    pass


class Wavelengths:
    pass


class Physics:
    pass


class Dust:
    component = []
    pass


class DustComponent:
    pass


class Grid:
    pass


class Map:
    pass


class Zone:
    dust = []
    pass


class Mol:
    molecule = []
    pass


class Molecule:
    pass


class Star:
    pass


class Simu:
    version = float()
    pass


class Params:

    simu = Simu()
    phot = Photons()
    wavelengths = Wavelengths()
    map = Map()
    grid = Grid()
    zones = []
    mol = Mol()
    stars = []

    _minimum_version = 3.0

    def __init__(self, filename=None, **kwargs):
        self.filename = filename
        self._read(**kwargs)

    def _read(self):

        with open(self.filename, 'r') as file:
            f = []
            # Reading file and removing comments
            for line in file:
                # Skipping comments and empty lines
                if (not line.startswith("#")) and (len(line) > 1):
                    f += [line]
            f = iter(f)

        # -- Version of the parameter file --
        line = next(f).split()
        self.simu.version = float(line[0])
        if self.simu.version < self._minimum_version - 1e-3:
            print("Parameter file version is ", self.simu.version)
            raise Exception(
                'Parameter file version must be at least {ver:.2f}'.format(
                    ver=self._minimum_version
                )
            )

        # -- Number of photon packages --
        line = next(f).split()
        self.phot.nphot_T = float(line[0])
        line = next(f).split()
        self.phot.nphot_SED = float(line[0])
        line = next(f).split()
        self.phot.nphot_image = float(line[0])

        # -- Wavelengths --
        line = next(f).split()
        self.wavelengths.n_wl = int(line[0])
        self.wavelengths.wl_min = float(line[1])
        self.wavelengths.wl_max = float(line[2])

        line = next(f).split()
        self.simu.compute_T = line[0][0] == "T"
        self.simu.compute_SED = line[1][0] == "T"
        self.simu.use_default_wl = line[2][0] == "T"

        line = next(f).split()
        self.wavelengths.file = line[0]

        line = next(f).split()
        self.simu.separate_contrib = line[0][0] == "T"
        self.simu.separate_pola = line[1][0] == "T"

        # -- Grid --
        line = next(f).split()
        self.grid.type = int(line[0])
        line = next(f).split()
        self.grid.n_rad = int(line[0])
        self.grid.nz = int(line[1])
        self.grid.n_az = int(line[2])
        self.grid.n_rad_in = int(line[3])

        # -- Maps --
        line = next(f).split()
        self.map.nx = int(line[0])
        self.map.ny = int(line[1])
        self.map.size = float(line[2])

        line = next(f).split()
        self.map.RT_imin = float(line[0])
        self.map.RT_imax = float(line[1])
        self.map.RT_ntheta = int(line[2])
        self.map.lRT_centered = line[3][0] == "T"

        line = next(f).split()
        self.map.RT_az_min = float(line[0])
        self.map.RT_az_max = float(line[1])
        self.map.RT_n_az = int(line[2])

        line = next(f).split()
        self.map.distance = float(line[0])

        line = next(f).split()
        self.map.PA = float(line[0])

        # -- Scattering method --
        line = next(f).split()
        self.simu.scattering_method = int(line[0])

        line = next(f).split()
        self.simu.phase_function_method = int(line[0])

        # -- Symetries --
        line = next(f).split()
        self.simu.image_symmetry = line[0][0] == "T"
        line = next(f).split()
        self.simu.central_symmetry = line[0][0] == "T"
        line = next(f).split()
        self.simu.axial_symmetry = line[0][0] == "T"

        # -- Disk physics --
        line = next(f).split()
        self.simu.dust_settling_type = int(line[0])
        self.simu.dust_settling_exp = float(line[1])
        self.simu.a_settling = float(line[2])

        line = next(f).split()
        self.simu.radial_migration = line[0][0] == "True"

        line = next(f).split()
        self.simu.dust_sublimation = line[0][0] == "True"

        line = next(f).split()
        self.simu.hydrostatic_eq = line[0][0] == "True"

        line = next(f).split()
        self.simu.viscous_heating = line[0][0] == "True"
        self.simu.viscosity = float(line[1])

        # -- Number of zones --
        line = next(f).split()
        n_zones = int(line[0])
        self.simu.n_zones = n_zones

        # -- Density structure --
        z = Zone()
        for k in range(n_zones):
            self.zones.append(z)

            line = next(f).split()
            self.zones[k].geometry = int(line[0])

            line = next(f).split()
            self.zones[k].dust_mass = float(line[0])
            self.zones[k].gas_to_dust_ratio = float(line[1])

            line = next(f).split()
            self.zones[k].h0 = float(line[0])
            self.zones[k].Rref = float(line[1])
            self.zones[k].vertical_exp = float(line[2])

            line = next(f).split()
            self.zones[k].Rin = float(line[0])
            self.zones[k].edge = float(line[1])
            self.zones[k].Rout = float(line[2])
            self.zones[k].Rc = float(line[3])

            line = next(f).split()
            self.zones[k].flaring_exp = float(line[0])

            line = next(f).split()
            self.zones[k].surface_density_exp = float(line[0])
            self.zones[k].m_gamma_exp = float(line[1])

        # -- Grain properties --
        d = Dust
        for k in range(n_zones):
            line = next(f).split()
            n_species = int(line[0])
            self.zones[k].n_species = n_species

            for j in range(n_species):
                self.zones[k].dust.append(d)

                line = next(f).split()
                self.zones[k].dust[j].type = line[0]
                n_components = int(line[1])
                self.zones[k].dust[j].n_components = n_components
                self.zones[k].dust[j].mixing_rule = int(line[2])
                self.zones[k].dust[j].porosity = float(line[3])
                self.zones[k].dust[j].mass_fraction = float(line[4])
                self.zones[k].dust[j].DHS_Vmax = float(line[5])

                c = DustComponent()
                for l in range(n_components):
                    self.zones[k].dust[j].component.append(c)

                    line = next(f).split()
                    self.zones[k].dust[j].component[l].file = line[0]
                    self.zones[k].dust[j].component[l].volume_fraction = float(line[1])

                line = next(f).split()
                self.zones[k].dust[j].heating_method = int(line[0])

                line = next(f).split()
                self.zones[k].dust[j].amin = float(line[0])
                self.zones[k].dust[j].amax = float(line[1])
                self.zones[k].dust[j].aexp = float(line[2])
                self.zones[k].dust[j].n_grains = int(line[3])

        # -- Molecular settings --
        line = next(f).split()
        self.mol.compute_pop = line[0][0] == "T"
        self.mol.compute_pop_accurate = line[1][0] == "T"
        self.mol.LTE = line[2][0] == "T"
        self.mol.profile_width = float(line[3])

        line = next(f).split()
        self.mol.v_turb = float(line[0])

        line = next(f).split()
        n_mol = int(line[0])
        self.mol.n_mol = n_mol

        m = Molecule()
        for k in range(n_mol):
            self.mol.molecule.append(m)

            line = next(f).split()
            self.mol.molecule[k].file = line[0]
            self.mol.molecule[k].level_max = int(line[1])

            line = next(f).split()
            self.mol.molecule[k].v_max = float(line[0])
            self.mol.molecule[k].nv = int(line[1])

            line = next(f).split()
            self.mol.molecule[k].cst_abundance = line[0][0] == "T"
            self.mol.molecule[k].abundance = line[1]
            self.mol.molecule[k].abundance_file = line[2]

            line = next(f).split()
            self.mol.molecule[k].ray_tracing = line[0][0] == "T"
            nTrans = int(line[1])
            self.mol.molecule[k].n_trans = nTrans

            line = next(f).split()
            self.mol.molecule[k].transitions = list(
                map(int, line[0:nTrans])
            )  # convert list of str to int

        # -- Star properties --
        line = next(f).split()
        n_stars = int(line[0])
        self.simu.n_stars = n_stars
        s = Star()
        for k in range(n_stars):
            self.stars.append(s)

            line = next(f).split()
            self.stars[k].Teff = float(line[0])
            self.stars[k].R = float(line[1])
            self.stars[k].M = float(line[2])
            self.stars[k].x = float(line[3])
            self.stars[k].y = float(line[4])
            self.stars[k].z = float(line[5])
            self.stars[k].is_bb = line[6][0] == "T"

            line = next(f).split()
            self.stars[k].file = line[0]

            line = next(f).split()
            self.stars[k].fUV = float(line[0])
            self.stars[k].slope_UV = float(line[1])

        # -- Command line options --
        for line in f:
            if (len(line) > 0):
                line = line.split()
                if (len(line) > 0): # we test again in case there were only spaces
                    if (line[0] == "Executed"):
                        self.options = " ".join(line[6:])

                        if (line[0] == "sha"):
                            self.mcfost_sha = line[2]



    def __str__(self):
        """ Return a formatted parameter file. Currently returns v3.0 format
        """

        # -- Photon packets --
        txt = f"""3.0                       mcfost version\n
#-- Number of photon packages --
  {self.phot.nphot_T:<10.5g}              nbr_photons_eq_th  : T computation
  {self.phot.nphot_SED:<10.5g}              nbr_photons_lambda : SED computation
  {self.phot.nphot_image:<10.5g}              nbr_photons_image : images computation\n\n"""

        # -- Wavelengths --
        txt += f"""#-- Wavelength --
  {self.wavelengths.n_wl:<4d} {self.wavelengths.wl_min:<5.1f} {self.wavelengths.wl_max:<7g}      n_lambda, lambda_min, lambda_max [microns]
  {self.simu.compute_T} {self.simu.compute_SED} {self.simu.use_default_wl}         compute temperature?, compute sed?, use default wavelength grid ?
  {self.wavelengths.file}            wavelength file (if previous parameter is F)
  {self.simu.separate_contrib} {self.simu.separate_pola}              separation of different contributions?, stokes parameters?\n\n"""

        # -- Grid --
        txt += f"""#-- Grid geometry and size --
  {self.grid.type:>1d}                       1 = cylindrical, 2 = spherical
  {self.grid.n_rad} {self.grid.nz} {self.grid.n_az} {self.grid.n_rad_in}             n_rad (log distribution), nz (or n_theta), n_az, n_rad_in\n\n"""

        # -- Maps --
        txt += f"""#-- Maps --
  {self.map.nx} {self.map.ny} {self.map.size:5.1f}           grid (nx,ny), size [au]
  {self.map.RT_imin:<4.1f}  {self.map.RT_imax:<4.1f}  {self.map.RT_ntheta:>2d} {self.map.lRT_centered}    RT: imin, imax, n_incl, centered ?
  {self.map.RT_az_min:<4.1f}  {self.map.RT_az_max:<4.1f}  {self.map.RT_n_az:>2d}          RT: az_min, az_max, n_az
  {self.map.distance:<6.2f}                  distance (pc)
  {self.map.PA:<6.2f}                  disk PA\n\n"""

        # -- Scattering method --
        txt += f"""#-- Scattering method --
  {self.simu.scattering_method}                       0=auto, 1=grain prop, 2=cell prop
  {self.simu.phase_function_method}                       1=Mie, 2=hg (2 implies the loss of polarizarion)\n\n"""

        # -- Symetries --
        txt += f"""#-- Symmetries --
  {self.simu.image_symmetry}                    image symmetry
  {self.simu.central_symmetry}                    central symmetry
  {self.simu.axial_symmetry}                    axial symmetry (important only if N_phi > 1)\n\n"""

        # -- Disk physics --
        txt += f"""#Disk physics
  {self.simu.dust_settling_type}  {self.simu.dust_settling_exp:<6.2f}  {self.simu.a_settling:<6.2f}       dust_settling (0=no settling, 1=parametric, 2=Dubrulle, 3=Fromang), exp_strat, a_strat (for parametric settling)
  {self.simu.radial_migration}                   dust radial migration
  {self.simu.dust_sublimation}                   sublimate dust
  {self.simu.hydrostatic_eq}                   hydrostatic equilibrium
  {self.simu.viscous_heating}  {self.simu.viscosity:4.1g}            viscous heating, alpha_viscosity\n\n"""

        # -- Number of zones --
        txt += f"""#-- Number of zones --   1 zone = 1 density structure + corresponding grain properties
  {self.simu.n_zones}\n\n"""

        # -- Density structure --
        txt += f"#-- Density structure --\n"
        for k in range(self.simu.n_zones):
            txt += f"""  {self.zones[k].geometry}                        zone type : 1 = disk, 2 = tapered-edge disk, 3 = envelope, 4 = debris disk, 5 = wall
  {self.zones[k].dust_mass:<10.2e} {self.zones[k].gas_to_dust_ratio:<5.1f}         dust mass,  gas-to-dust mass ratio
  {self.zones[k].h0:<5.1f}  {self.zones[k].Rref:<6.1f} {self.zones[k].vertical_exp:<6.1f}     scale height, reference radius (AU), unused for envelope, vertical profile exponent (only for debris disk)
  {self.zones[k].Rin:<6.1f}  {self.zones[k].edge:<6.1f} {self.zones[k].Rout:<6.1f} {self.zones[k].Rc:<6.1f}  Rin, edge, Rout, Rc (AU) Rc is only used for tappered-edge & debris disks (Rout set to 8*Rc if Rout==0)
  {self.zones[k].flaring_exp:<8.3f}                 flaring exponent, unused for envelope
  {self.zones[k].surface_density_exp} {self.zones[k].m_gamma_exp}                 surface density exponent (or -gamma for tappered-edge disk or volume density for envelope), usually < 0, -gamma_exp (or alpha_in & alpha_out for debris disk)\n\n"""
        txt += f"\n"

        # -- Grain properties --
        txt += f"#-- Grain properties --\n"
        for k in range(self.simu.n_zones):
            txt += (
                f"  {self.zones[k].n_species}                      Number of species\n"
            )
            for j in range(self.zones[k].n_species):
                txt += f"  Mie {self.zones[k].dust[j].n_components} {self.zones[k].dust[j].mixing_rule} {self.zones[k].dust[j].porosity:<5.2f} {self.zones[k].dust[j].mass_fraction:<5.2f} {self.zones[k].dust[j].DHS_Vmax}    Grain type (Mie or DHS), N_components, mixing rule (1 = EMT or 2 = coating),  porosity, mass fraction, Vmax (for DHS)\n"
                for l in range(self.zones[k].dust[j].n_components):
                    txt += f"  {self.zones[k].dust[j].component[l].file}  {self.zones[k].dust[j].component[l].volume_fraction}     Optical indices file, volume fraction\n"
                txt += f"""  {self.zones[k].dust[j].heating_method}                          Heating method : 1 = RE + LTE, 2 = RE + NLTE, 3 = NRE
  {self.zones[k].dust[j].amin} {self.zones[k].dust[j].amax}  {self.zones[k].dust[j].aexp} {self.zones[k].dust[j].n_grains}       amin, amax, aexp, nbr_grains\n\n"""

        # -- Molecular settings --
        txt += f"""#-- Molecular RT settings --
  {self.mol.compute_pop}  {self.mol.compute_pop_accurate}  {self.mol.LTE} {self.mol.profile_width}      lpop, laccurate_pop, LTE, profile width
  {self.mol.v_turb}                        v_turb [km/s]
  {self.mol.n_mol}                          nmol\n"""
        for k in range(self.mol.n_mol):
            txt += f"""  {self.mol.molecule[k].file} {self.mol.molecule[k].level_max}          molecular data filename, level_max
  {self.mol.molecule[k].v_max} {self.mol.molecule[k].nv}                 vmax (km.s-1), n_speed
  {self.mol.molecule[k].cst_abundance} {self.mol.molecule[k].abundance} {self.mol.molecule[k].abundance_file}   cst molecule abundance ?, abundance, abundance file
  {self.mol.molecule[k].ray_tracing}  {self.mol.molecule[k].n_trans}                   ray tracing ?,  number of lines in ray-tracing\n """
            for j in range(self.mol.molecule[k].n_trans):
                txt += f" {self.mol.molecule[k].transitions[j]}"
            txt += f" transition numbers\n"
        txt += f"\n"

        # -- Star properties --
        txt += f"""#-- Star properties --
  {self.simu.n_stars}  Number of stars\n"""
        for k in range(self.simu.n_stars):
            txt += f"""  {self.stars[k].Teff} {self.stars[k].R} {self.stars[k].M} {self.stars[k].x} {self.stars[k].y} {self.stars[k].x} {self.stars[k].is_bb}  Temp, radius (solar radius),M (solar mass),x,y,z (AU), is a blackbody?
  {self.stars[k].file}
  {self.stars[k].fUV} {self.stars[k].slope_UV}     fUV, slope_UV\n"""

        return txt

    def writeto(self, outname):
        """ Write an MCFOST parameter file to disk.  """
        outfile = open(outname, 'w')
        outfile.write(str(self))
        outfile.close()

    def calc_inclinations(self):
        # Calculate the inclinations for the ray-traced SEDs and images
        if self.map.RT_ntheta == 1:
            return self.map.RT_imin
        else:
            cos_min, cos_max = np.cos(np.deg2rad([self.map.RT_imin, self.map.RT_imax]))
            if self.map.lRT_centered:
                return (
                    np.rad2deg(np.arccos(
                        cos_min
                        + (np.arange(self.map.RT_ntheta) + 0.5)
                        / self.map.RT_ntheta
                        * (cos_max - cos_min)
                    ))
                )
            else:
                return (
                    np.rad2deg(np.arccos(
                        cos_min
                        + (np.arange(self.map.RT_ntheta))
                        / (self.map.RT_ntheta - 1)
                        * (cos_max - cos_min)
                    ))
                )


def find_parameter_file(directory="./"):

    list = glob.glob(directory + "/*.par*")
    if len(list) == 1:
        return list[0]
    elif len(list) > 1:
        raise ValueError("Multiple parameter files found in " + directory)
    else:
        raise ValueError("No parameter files found in " + directory)

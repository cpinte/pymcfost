import glob

class McfostPhotons:
    pass

class McfostWavelengths:
    pass

class McfostPhysics:
    pass

class McfostDust:
    component = []
    pass

class McfostDustComponent:
    pass

class McfostGrid:
    pass

class McfostMap:
    pass

class McfostZone:
    dust = []
    pass

class McfostMol:
    molecule = []
    pass

class McfostMolecule:
    pass

class McfostStar:
    pass

class McfostSimu:
    version = float()
    pass

class McfostParams:

    simu = McfostSimu()
    phot = McfostPhotons()
    wavelengths = McfostWavelengths()
    map = McfostMap()
    grid = McfostGrid()
    zones = []
    mol = McfostMol()
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
                if ((not line.startswith("#")) and (len(line)>1)): # Skipping comments and empty lines
                   f += [line]
            f = iter(f)

        #-- Version of the parameter file --
        line = next(f).split()
        self.simu.version = float(line[0])
        if (self.simu.version < self._minimum_version):
            raise Exception('Parameter file version must be at least {ver:.2f}'.format(ver=self._minimum_version))


        #-- Number of photon packages --
        line = next(f).split()
        self.phot.nphot_T = float(line[0])
        line = next(f).split()
        self.phot.nphot_SED = float(line[0])
        line = next(f).split()
        self.phot.nphot_image = float(line[0])

        #-- Wavelengths --
        line = next(f).split()
        self.wavelengths.n_wl = int(line[0])
        self.wavelengths.wl_min = float(line[1])
        self.wavelengths.wl_max = float(line[2])

        line = next(f).split()
        self.simu.compute_T = line[0] == "T"
        self.simu.compute_SED = line[1] == "T"
        self.simu.use_default_wl = line[2] == "T"

        line = next(f).split()
        self.wavelengths.file = line[0]

        line = next(f).split()
        self.simu.separate_contrib = line[0] == "T"
        self.simu.separate_pola = line[1] == "T"

        #-- Grid --
        line = next(f).split()
        self.grid.type = int(line[0])
        line = next(f).split()
        self.grid.n_rad = int(line[0])
        self.grid.nz = int(line[1])
        self.grid.n_az = int(line[2])
        self.grid.n_rad_in = int(line[3])

        #-- Maps --
        line = next(f).split()
        self.map.nx = int(line[0])
        self.map.ny = int(line[1])
        self.map.size = float(line[0])

        line = next(f).split()
        self.map.RT_imin = float(line[0])
        self.map.RT_imax = float(line[1])
        self.map.RT_ntheta = int(line[2])
        self.map.lRT_centered = line[3] == "T"

        line = next(f).split()
        self.map.RT_az_min = float(line[0])
        self.map.RT_az_max = float(line[1])
        self.map.RT_n_az = int(line[2])

        line = next(f).split()
        self.map.distance = float(line[0])

        line = next(f).split()
        self.map.PA = float(line[0])

        #-- Scattering method --
        line = next(f).split()
        self.simu.scattering_method = int(line[0])

        line = next(f).split()
        self.simu.phase_function_method = int(line[0])

        #-- Symetries --
        line = next(f).split()
        self.simu.image_symmetry = line[0] == "T"
        line = next(f).split()
        self.simu.central_symmetry = line[0] == "T"
        line = next(f).split()
        self.simu.axial_symmetry = line[0] == "T"

        #-- Disk physics --
        line = next(f).split()
        self.simu.dust_settling_type = int(line[0])
        self.simu.dust_settling_exp = float(line[1])
        self.simu.a_settling = float(line[2])

        line = next(f).split()
        self.simu.radial_migration = line[0] == "True"

        line = next(f).split()
        self.simu.dust_sublimation = line[0] == "True"

        line = next(f).split()
        self.simu.hydrostatic_eq = line[0] == "True"

        line = next(f).split()
        self.simu.viscous_heating = line[0] == "True"
        self.simu.viscosity = float(line[1])

        #-- Number of zones --
        line = next(f).split()
        n_zones = int(line[0])
        self.simu.n_zones = n_zones

        #-- Density structure --
        z = McfostZone()
        for k in range(n_zones):
            self.zones.append(z)

            line = next(f).split()
            self.zones[k].geometry = int(line[0])

            line = next(f).split()
            self.zones[k].dust_mass = float(line[0])
            self.zones[k].gas_dust_radtio = float(line[1])

            line = next(f).split()
            self.zones[k].h0 = float(line[0])
            self.zones[k].Rref = float(line[1])
            self.zones[k].vertical_exp = int(line[2])

            line = next(f).split()
            self.zones[k].Rin = float(line[0])
            self.zones[k].edge = float(line[0])
            self.zones[k].Rout = float(line[0])
            self.zones[k].Rc = float(line[0])

            line = next(f).split()
            self.zones[k].beta = float(line[0])

            line = next(f).split()
            self.zones[k].p1 = float(line[0])
            self.zones[k].p2 = float(line[1])

        #-- Grain properties --
        d = McfostDust
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

                c = McfostDustComponent()
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

        #-- Molecular settings --
        line = next(f).split()
        self.mol.compute_pop = line[0] == "T"
        self.mol.compute_pop_accurate = line[1] == "T"
        self.mol.LTE = line[2] == "T"
        self.mol.profile_width = float(line[3])

        line = next(f).split()
        self.mol.v_turb = float(line[0])

        line = next(f).split()
        n_mol = int(line[0])
        self.mol.n_mol = n_mol

        m = McfostMolecule()
        for k in range(n_mol):
            self.mol.molecule.append(m)

            line = next(f).split()
            self.mol.molecule[k].file = line[0]
            self.mol.molecule[k].level_max = int(line[1])

            line = next(f).split()
            self.mol.molecule[k].v_max = float(line[0])
            self.mol.molecule[k].nv = int(line[1])

            line = next(f).split()
            self.mol.molecule[k].cst_abundance = line[0] == "T"
            self.mol.molecule[k].abundance_file = line[1]

            line = next(f).split()
            self.mol.molecule[k].ray_tracing = line[0] == "T"
            nTrans = int(line[1])
            self.mol.molecule[k].n_trans = nTrans

            line = next(f).split()
            self.mol.molecule[k].transitions = list(map(int, line[0:nTrans])) # convert list of str to int

        #-- Star properties --
        line = next(f).split()
        n_stars = int(line[0])
        self.simu.n_stars = n_stars
        s = McfostStar()
        for k in range(n_stars):
            self.stars.append(s)

            line = next(f).split()
            self.stars[k].Teff = float(line[0])
            self.stars[k].R = float(line[1])
            self.stars[k].M = float(line[2])
            self.stars[k].x = float(line[3])
            self.stars[k].y = float(line[4])
            self.stars[k].z = float(line[5])
            self.stars[k].is_bb = line[6] == "T"

            line = next(f).split()
            self.stars[k].file = line[0]

            line = next(f).split()
            self.stars[k].fUV = float(line[0])
            self.stars[k].slope_UV = float(line[1])


    def write():
        pass



def find_parameter_file(directory="./"):
    list = glob.glob(directory+"/*.par*")

    if len(list) == 1:
        return list[0]
    elif len(list) > 1:
        raise ValueError("Multiple parameter files found in "+directory)
    else:
        raise ValueError("No parameter files found in "+directory)

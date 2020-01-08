import glob
import numpy as np
from abc import ABC, abstractmethod

def _word_to_bool(word):
    """convert a string to boolean according the first 2 characters."""
    _accepted_bool_prefixes = ("T", ".T")
    return word.upper().startswith(_accepted_bool_prefixes)


class ParafileSection:
    """writeme"""
    def __init__(self, blocks: list):
        if isinstance(blocks, AbstractParameterBlock):
            # force iterability
            blocks = [blocks]

        for b in blocks[1:]:
            assert isinstance(b, type(blocks[0]))

        self._blocks = blocks

    def __str__(self):
        txt = "# -- " + self._blocks[0].__class__.header + " --\n"
        txt += "\n".join([str(b) for b in self._blocks])
        return txt


class AbstractParameterBlock(ABC):
    """writeme"""
    def __str__(self):
        txt = ""
        for line in self.lines:
            txt += (
                "  "
                + "  ".join(line.values()).ljust(36)
                + "  "
                + ", ".join(line.keys())
                + "\n"
            )
        return txt

    def _link_simu_block(self, simu):
        self.simu = simu

    @property
    @abstractmethod
    def lines(self):
        """this method should return a list of lines in a parafile.
        A line is represented as a dictionnary where keys are comments and
        values are formatted strings representing the value of a parameter"""
        pass


class Photons(AbstractParameterBlock):
    header = "Number of photon packages"

    @property
    def lines(self):
        return [
            {"nbr_photons_eq_th : T computation": f"{self.nphot_T:<10.5g}"},
            {"nbr_photons_lambda : SED computation": f"{self.nphot_SED:<10.5g}"},
            {"nbr_photons_image : images computation": f"{self.nphot_image:<10.5g}"},
        ]


class Wavelengths(AbstractParameterBlock):
    header = "Wavelengths"

    @property
    def lines(self):
        assert hasattr(self, "simu")
        return [
            {
                "n_lambda": f"{self.n_wl:<4d}",
                "lambda_min": f"{self.wl_min:<5.1f}",
                "lambda_max [microns]": f"{self.wl_max:<7g}",
            },
            {
                "compute temperature?": f"{self.simu.compute_T}",
                "compute sed?": f"{self.simu.compute_SED}",
                "use default wavelength grid ?": f"{self.simu.use_default_wl}",
            },
            {"wavelength file (if previous parameter is F)": f"{self.file}"},
            {
                "separation of different contributions?": f"{self.simu.separate_contrib}",
                "stokes parameters?": f"{self.simu.separate_pola}",
            },
        ]


class Grid(AbstractParameterBlock):
    header = "Grid geometry and size"

    @property
    def lines(self):
        return [
            {"1 = cylindrical, 2 = spherical": f"{self.type:>1d}"},
            {
                "n_rad (log distribution)": f"{self.n_rad}",
                "nz (or n_theta)": f"{self.nz}",
                "n_az": f"{self.n_az}",
                "n_rad_in": f"{self.n_rad_in}",
            },
        ]


class Map(AbstractParameterBlock):
    header = "Maps"

    @property
    def lines(self):
        return [
            {
                "grid (nx)": f"{self.nx:d}",
                "grid (ny)": f"{self.ny:d}",
                "size [au]": f"{self.size:5.1f}",
            },
            {
                "RT: imin": f"{self.RT_imin:<4.1f}",
                "imax": f"{self.RT_imax:<4.1f}",
                "n_incl": f"{self.RT_ntheta:>2d}",
                "centered ?": f"{self.lRT_centered}",
            },
            {
                "RT: az_min": f"{self.RT_az_min:<4.1f}",
                "az_max": f"{self.RT_az_max:<4.1f}",
                "n_az": f"{self.RT_n_az:>2d}",
            },
            {"distance [pc]": f"{self.distance:<6.2f}"},
            {"disk PA": f"{self.PA:<6.2f}"},
        ]


class Scattering(AbstractParameterBlock):
    header = "Scattering method"

    @property
    def lines(self):
        assert hasattr(self, "simu")
        return [
            {"0=auto, 1=grain prop, 2=cell prop": f"{self.simu.scattering_method}"},
            {
                "1=Mie, 2=hg (2 implies the loss of polarizarion)": f"{self.simu.phase_function_method}"
            },
        ]


class Symmetries(AbstractParameterBlock):
    header = "Symmetries"

    @property
    def lines(self):
        assert hasattr(self, "simu")
        return [
            {"image symmetry": f"{self.simu.image_symmetry}"},
            {"central symmetry": f"{self.simu.central_symmetry}"},
            {
                "axial symmetry (important only if N_phi > 1)": f"{self.simu.axial_symmetry}"
            },
        ]


class Physics(AbstractParameterBlock):
    header = "Disk physics"

    @property
    def lines(self):
        assert hasattr(self, "simu")
        return [
            {
                "dust_settling (0=no settling, 1=parametric, 2=Dubrulle, 3=Fromang)": f"{self.simu.dust_settling_type}",
                "exp_strat": f"{self.simu.dust_settling_exp:<6.2f}",
                "a_strat (for parametric settling)": f"{self.simu.a_settling:<6.2f}",
            },
            {"dust radial migration": f"{self.simu.radial_migration}"},
            {"sublimate dust": f"{self.simu.dust_sublimation}"},
            {"hydrostatic equilibrium": f"{self.simu.hydrostatic_eq}"},
            {
                "viscous heating": f"{self.simu.viscous_heating}",
                "alpha_viscosity": f"{self.simu.viscosity:4.1g}",
            },
        ]


class Nzone(AbstractParameterBlock):
    header = "Number of zones"

    @property
    def lines(self):
        assert hasattr(self, "simu")
        return [
            {
                "1 zone = 1 density structure + corresponding grain properties": f"{self.simu.n_zones}"
            }
        ]

class Zone(AbstractParameterBlock):
    header = "Density structure"
    dust = []

    @property
    def lines(self):
        return [
            {"zone type : 1 = disk, 2 = tapered-edge disk, 3 = envelope, 4 = debris disk, 5 = wall": f"{self.geometry}"},
            {"dust mass": f"{self.dust_mass:<10.2e}",
            "gas-to-dust mass ratio": f"{self.gas_to_dust_ratio:<5.1f}"},
            {"scale height": f"{self.h0:<5.1f}",
            "reference radius [au] (unused for envelope)": f"{self.Rref:<6.1f}",
            "vertical profile exponent (only for debris disk)": f"{self.vertical_exp:<6.1f}"
            },
            {"Rin": f"{self.Rin:<6.1f}",
            "edge": f"{self.edge:<6.1f}",
            "Rout": f"{self.Rout:<6.1f}",
            "Rc [AU] - only used for tappered-edge & debris disks (Rout set to 8*Rc if Rout==0)": f"{self.Rc:<6.1f}",
            },
            {"flaring exponent - unused for envelope": f"{self.flaring_exp:<8.3f}"},
            {"surface density exponent (or -gamma for tappered-edge disk or volume density for envelope) - usually < 0": f"{self.surface_density_exp}",
            "-gamma_exp (or alpha_in & alpha_out for debris disk)": f"{self.m_gamma_exp}"}
        ]



class Dust:
    component = []
    pass


class DustComponent:
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

        with open(self.filename, mode="rt") as file:
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
        # to support float notations (e.g. "1.28e8" or "64000.0"),
        # we read as float but convert to int
        line = next(f).split()
        self.phot.nphot_T = int(float(line[0]))
        line = next(f).split()
        self.phot.nphot_SED = int(float(line[0]))
        line = next(f).split()
        self.phot.nphot_image = int(float(line[0]))

        # -- Wavelengths --
        line = next(f).split()
        self.wavelengths.n_wl = int(line[0])
        self.wavelengths.wl_min = float(line[1])
        self.wavelengths.wl_max = float(line[2])

        line = next(f).split()
        self.simu.compute_T = _word_to_bool(line[0])
        self.simu.compute_SED = _word_to_bool(line[1])
        self.simu.use_default_wl = _word_to_bool(line[2])

        line = next(f).split()
        self.wavelengths.file = line[0]

        line = next(f).split()
        self.simu.separate_contrib = _word_to_bool(line[0])
        self.simu.separate_pola = _word_to_bool(line[1])

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
        self.map.lRT_centered = _word_to_bool(line[3])

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
        self.simu.image_symmetry = _word_to_bool(line[0])
        line = next(f).split()
        self.simu.central_symmetry = _word_to_bool(line[0])
        line = next(f).split()
        self.simu.axial_symmetry = _word_to_bool(line[0])

        # -- Disk physics --
        line = next(f).split()
        self.simu.dust_settling_type = int(line[0])
        self.simu.dust_settling_exp = float(line[1])
        self.simu.a_settling = float(line[2])

        line = next(f).split()
        self.simu.radial_migration = _word_to_bool(line[0])

        line = next(f).split()
        self.simu.dust_sublimation = _word_to_bool(line[0])

        line = next(f).split()
        self.simu.hydrostatic_eq = _word_to_bool(line[0])

        line = next(f).split()
        self.simu.viscous_heating = _word_to_bool(line[0])
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
        self.mol.compute_pop = _word_to_bool(line[0])
        self.mol.compute_pop_accurate = _word_to_bool(line[1])
        self.mol.LTE = _word_to_bool(line[2])
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
            self.mol.molecule[k].cst_abundance = _word_to_bool(line[0])
            self.mol.molecule[k].abundance = line[1]
            self.mol.molecule[k].abundance_file = line[2]

            line = next(f).split()
            self.mol.molecule[k].ray_tracing = _word_to_bool(line[0])
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
            self.stars[k].is_bb = _word_to_bool(line[6])

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

        # -- Header --
        txt = "".join(["3.0", 23*" ", "mcfost version", "\n"])

        # -- Photon packets --
        txt += str(ParafileSection(blocks=self.phot)) + "\n"

        # -- Wavelengths --
        self.wavelengths._link_simu_block(self.simu)
        txt += str(ParafileSection(blocks=self.wavelengths)) + "\n"

        # -- Grid --
        txt += str(ParafileSection(blocks=self.grid)) + "\n"

        # -- Maps --
        txt += str(ParafileSection(blocks=self.map)) + "\n"

        # -- Scattering method --
        self.scattering = Scattering()
        self.scattering._link_simu_block(self.simu)
        txt += str(ParafileSection(blocks=self.scattering)) + "\n"

        # -- Symetries --
        self.symmetries = Symmetries()
        self.symmetries._link_simu_block(self.simu)
        txt += str(ParafileSection(blocks=self.symmetries)) + "\n"

        # -- Disk physics --
        self.physics = Physics()
        self.physics._link_simu_block(self.simu)
        txt += str(ParafileSection(blocks=self.physics)) + "\n"

        # -- Number of zones --
        self.n_zone = Nzone()
        self.n_zone._link_simu_block(self.simu)
        txt += str(ParafileSection(blocks=self.n_zone)) + "\n"

        # -- Density structure --
        txt += str(ParafileSection(blocks=self.zones[:self.simu.n_zones])) + "\n"

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
        with open(outname, mode="wt") as file:
            file.write(str(self))

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

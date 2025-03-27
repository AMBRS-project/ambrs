"""ambrs.mam4 -- data types and functions related to the MAM4 box model"""

from .aerosol import AerosolProcesses, AerosolModalSizePopulation, \
                     AerosolModalSizeState
from .aerosol_model import BaseAerosolModel
from .analysis import Output
from .gas import GasSpecies
from .scenario import Scenario
from .ppe import Ensemble

from dataclasses import dataclass, make_dataclass
import netCDF4
import numpy as np
import os.path
import scipy.stats

@dataclass
class Input:
    """ambrs.mam4.Input -- an input dataclass for the MAM4 box model"""
    # all fields here are named identically to their respective namelist
    # parameters in the MAM4 box model

    # scenario
    scenario: Scenario

    # CAMP mass fractions
    mfs: list

    # timestepping parameters
    mam_dt: float
    mam_nstep: int

    # aerosol processes
    mdo_gaschem: bool
    mdo_gasaerexch: bool
    mdo_rename: bool
    mdo_newnuc: bool
    mdo_coag: bool

    # atmospheric state
    temp: float
    press: float
    RH_CLEA: float

    #------------------------------------
    # modal number concentrations [#/m3]
    #------------------------------------

    numc1: float
    numc2: float
    numc3: float
    numc4: float

    #---------------------------------------------------
    # mode-specific aerosol specific mass fractions [-]
    #---------------------------------------------------

    # mode 1 (accumulation mode)
    mfso41: float
    mfpom1: float
    mfsoa1: float
    mfbc1: float
    mfdst1: float
    mfncl1: float

    # mode 2 (aitken mode)
    mfso42: float
    mfsoa2: float
    mfncl2: float

    # mode 3 (coarse mode)
    mfdst3: float
    mfncl3: float
    mfso43: float
    mfbc3: float
    mfpom3: float
    mfsoa3: float

    # mode 4 (primary carbon mode)
    mfpom4: float
    mfbc4: float

    #---------------------------------------
    # gas mixing ratios [kg gas/kg dry air]
    #---------------------------------------

    qso2: float
    qh2so4: float
    qsoag: float

# this type handles the mapping of AMBRS aerosol species to MAM4 species
# within all aerosol modes
class AerosolMassFractions:
    # @dataclass
    # class AccumMode:
    #     SO4: float
    #     POM: float
    #     SOA: float
    #     BC: float
    #     DST: float
    #     NCL: float

    # @dataclass
    # class AitkenMode:
    #     SO4: float
    #     SOA: float
    #     NCL: float

    # @dataclass
    # class CoarseMode:
    #     DST: float
    #     NCL: float
    #     SO4: float
    #     BC: float
    #     POM: float
    #     SOA: float

    # @dataclass
    # class PCarbonMode:
    #     POM: float
    #     BC: float

    def __init__(self,
                 scenario: Scenario):
        # self.accum = self.AccumMode(
        #     SO4 = scenario.size.modes[0].mass_fraction('ASO4'),
        #     POM = scenario.size.modes[0].mass_fraction('APOC'),
        #     SOA = scenario.size.modes[0].mass_fraction('SOA'), # FIXME: using MSA as a placeholder for SOA
        #     BC  = scenario.size.modes[0].mass_fraction("AEC"),
        #     DST = scenario.size.modes[0].mass_fraction("ASOIL"),
        #     NCL = scenario.size.modes[0].mass_fraction("ANA"), # FIXME: and "Cl"? Assuming 1:1 for now
        # )
        self.AccumMode = make_dataclass('AccumMode', [(p.aliases, float) if p.aliases else (p.name, float) for p in scenario.size.modes[0].species])
        self.accum = self.AccumMode(
            **{p.aliases if p.aliases else p.name : scenario.size.modes[0].mass_fraction(p.name) for p in scenario.size.modes[0].species}
        )
                
        
        # self.aitken = self.AitkenMode(
        #     SO4 = scenario.size.modes[1].mass_fraction("ASO4"),
        #     SOA = scenario.size.modes[1].mass_fraction("SOA"), # FIXME: using MSA as a placeholder for SOA
        #     NCL = scenario.size.modes[1].mass_fraction("ANA"), # FIXME: assuming 1:1 with Cl for now
        # )
        # self.coarse = self.CoarseMode(
        #     DST = scenario.size.modes[2].mass_fraction("ASOIL"),
        #     NCL = scenario.size.modes[2].mass_fraction("ANA"), # FIXME: see above
        #     SO4 = scenario.size.modes[2].mass_fraction("ASO4"),
        #     BC  = scenario.size.modes[2].mass_fraction("AEC"),
        #     POM = scenario.size.modes[2].mass_fraction("APOC"),
        #     SOA = scenario.size.modes[2].mass_fraction("SOA"), # FIXME: using MSA as a placeholder for SOA
        # )
        # self.pcarbon = self.PCarbonMode(
        #     POM = scenario.size.modes[3].mass_fraction("APOC"),
        #     BC  = scenario.size.modes[3].mass_fraction("AEC"),
        # )
        self.AitkenMode = make_dataclass('AitkenMode', [(p.aliases, float) if p.aliases else (p.name, float) for p in scenario.size.modes[1].species])
        self.aitken = self.AitkenMode(
            **{p.aliases if p.aliases else p.name : scenario.size.modes[1].mass_fraction(p.name) for p in scenario.size.modes[1].species}
        )

        self.CoarseMode = make_dataclass('CoarseMode', [(p.aliases, float) if p.aliases else (p.name, float) for p in scenario.size.modes[2].species])
        self.coarse = self.CoarseMode(
            **{p.aliases if p.aliases else p.name : scenario.size.modes[2].mass_fraction(p.name) for p in scenario.size.modes[2].species}
        )
        
        self.PCarbonMode = make_dataclass('PCarbonMode', [(p.aliases, float) if p.aliases else (p.name, float) for p in scenario.size.modes[3].species])
        self.pcarbon = self.PCarbonMode(
            **{p.aliases if p.aliases else p.name : scenario.size.modes[3].mass_fraction(p.name) for p in scenario.size.modes[3].species}
        )
        
# this type handles the mapping of AMBRS gas species to MAM4 species
class GasMixingRatios:
    def __init__(self,
                 scenario: Scenario):
        iso2 = GasSpecies.find(scenario.gases, 'SO2')
        if iso2 == -1:
            raise ValueError("SO2 gas not found in gas species")
        ih2so4 = GasSpecies.find(scenario.gases, 'H2SO4')
        if ih2so4 == -1:
            raise ValueError("H2SO4 gas not found in gas species")
        isoag = GasSpecies.find(scenario.gases, 'SOAG')
        self.SO2 = scenario.gas_concs[iso2]
        self.H2SO4 = scenario.gas_concs[ih2so4]
        self.SOAG = 0.0 if isoag == -1 else scenario.gas_concs[isoag]
    
class AerosolModel(BaseAerosolModel):
    def __init__(self,
                 processes: AerosolProcesses):
        BaseAerosolModel.__init__(self, 'mam4', processes)

    def create_input(self,
                     scenario: Scenario,
                     dt: float,
                     nstep: int) -> Input:
        """ambrs.mam4.AerosolModel.create_input(processes, scenario, dt, nstep) ->
ambrs.mam4.Input object that can create input files for a MAM4 box model
simulation

Parameters:
    * scenario: an ambrs.Scenario object created by sampling a modal particle
      size distribution
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation"""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if nstep <= 0:
            raise ValueError("nstep must be positive")
        if not isinstance(scenario.size, AerosolModalSizeState):
            raise TypeError('Non-modal aerosol particle size state cannot be used to create MAM4 input!')
        if len(scenario.size.modes) != 4:
            raise TypeError(f'{len(scenario.size.mode)}-mode aerosol particle size state cannot be used to create MAM4 input!')
        
        # translate the scenario's aerosol mass fractions to MAM4-ese
        aero_mass_fracs = AerosolMassFractions(scenario)
        
        # translate the scenario's gas mixing ratios to MAM4-ese
        gas_mixing_ratios = GasMixingRatios(scenario)

        mfs1 = {p.name : scenario.size.modes[0].mass_fraction(p.name) for p in scenario.size.modes[0].species}# if p.name in ['ASO4','AH2O','AH3OP','SOA','APOC','AEC','ANA','ASOIL','H2SO4_aq']}
        mftot1 = sum(mfs1.values())
        mfs1 = {p.name : mfs1[p.name]/mftot1 for p in scenario.size.modes[0].species} # if p.name in ['ASO4','AH2O','AH3OP','SOA','APOC','AEC','ANA','ASOIL','H2SO4_aq']}

        mfs2 = {p.name : scenario.size.modes[1].mass_fraction(p.name) for p in scenario.size.modes[1].species} # if p.name in ['ASO4','AH2O','AH3OP','SOA','ANA','H2SO4_aq']}
        mftot2 = sum(mfs2.values())
        mfs2 = {p.name : mfs2[p.name]/mftot2 for p in scenario.size.modes[1].species} # if p.name in ['ASO4','AH2O','AH3OP','SOA','ANA','H2SO4_aq']}
        
        mfs3 = {p.name : scenario.size.modes[2].mass_fraction(p.name) for p in scenario.size.modes[2].species} # if p.name in ['ASO4','AH2O','AH3OP','SOA','APOC','AEC','ANA','ASOIL','H2SO4_aq']}
        mftot3 = sum(mfs3.values())
        mfs3 = {p.name : mfs3[p.name]/mftot3 for p in scenario.size.modes[2].species} # if p.name in ['ASO4','AH2O','AH3OP','SOA','APOC','AEC','ANA','ASOIL','H2SO4_aq']}

        mfs4 = {p.name : scenario.size.modes[3].mass_fraction(p.name) for p in scenario.size.modes[3].species} # if p.name in ['APOC','AEC','AH2O','AH3OP','H2SO4_aq']}
        mftot4 = sum(mfs4.values())
        mfs4 = {p.name : mfs4[p.name]/mftot4 for p in scenario.size.modes[3].species} # if p.name in ['APOC','AEC','AH2O','AH3OP','H2SO4_aq']}

        mfs = [mfs1, mfs2, mfs3, mfs4]
        
        return Input(
            scenario = scenario,

            mfs = mfs,
            
            mam_dt = dt,
            mam_nstep = nstep,

            mdo_gaschem = 1 if self.processes.gas_phase_chemistry else 0,
            mdo_gasaerexch = 1 if self.processes.condensation else 0,
            mdo_rename = 1,
            mdo_newnuc = 1 if self.processes.nucleation else 0,
            mdo_coag = 1 if self.processes.coagulation else 0,

            temp = scenario.temperature,
            press = scenario.pressure,
            RH_CLEA = scenario.relative_humidity,

            numc1 = scenario.size.modes[0].number,
            numc2 = scenario.size.modes[1].number,
            numc3 = scenario.size.modes[2].number,
            numc4 = scenario.size.modes[3].number,
            
            mfso41 = np.floor(aero_mass_fracs.accum.SO4/mftot1 * 10**12) / 10**12,
            mfpom1 = np.floor(aero_mass_fracs.accum.POM/mftot1 * 10**12) / 10**12,
            mfsoa1 = np.floor(aero_mass_fracs.accum.SOA/mftot1 * 10**12) / 10**12,
            mfbc1  = np.floor(aero_mass_fracs.accum.BC/mftot1 * 10**12) / 10**12,
            mfdst1 = np.floor(aero_mass_fracs.accum.DST/mftot1 * 10**12) / 10**12,
            mfncl1 = np.floor(aero_mass_fracs.accum.NCL/mftot1 * 10**12) / 10**12,

            mfso42 = np.floor(aero_mass_fracs.aitken.SO4/mftot2 * 10**12) / 10**12,
            mfsoa2 = np.floor(aero_mass_fracs.aitken.SOA/mftot2 * 10**12) / 10**12,
            mfncl2 = np.floor(aero_mass_fracs.aitken.NCL/mftot2 * 10**12) / 10**12,
            
            mfdst3 = np.floor(aero_mass_fracs.coarse.DST/mftot3 * 10**12) / 10**12,
            mfncl3 = np.floor(aero_mass_fracs.coarse.NCL/mftot3 * 10**12) / 10**12,
            mfso43 = np.floor(aero_mass_fracs.coarse.SO4/mftot3 * 10**12) / 10**12,
            mfbc3  = np.floor(aero_mass_fracs.coarse.BC/mftot3 * 10**12) / 10**12,
            mfpom3 = np.floor(aero_mass_fracs.coarse.POM/mftot3 * 10**12) / 10**12,
            mfsoa3 = np.floor(aero_mass_fracs.coarse.SOA/mftot3 * 10**12) / 10**12,

            mfpom4 = np.floor(aero_mass_fracs.pcarbon.POM/mftot4 * 10**12) / 10**12,
            mfbc4  = np.floor(aero_mass_fracs.pcarbon.BC/mftot4 * 10**12) / 10**12,

            qso2 = gas_mixing_ratios.SO2 * 1.e-06 * 64.0648 / 28.966,
            qh2so4 = gas_mixing_ratios.H2SO4 * 1.e-06 * 98.0784 / 28.966,
            qsoag = gas_mixing_ratios.SOAG * 1.e-06 * 12.011 / 28.966,
        )
    
    def invocation(self, exe: str, prefix: str) -> str:
        """input.invocation(exe, prefix) -> a string defining the command invoking
the input with the given executable and input prefix, assuming that the current
working directory contains any needed input files."""
        return f'{exe}'

    def write_input_files(self, input, dir, prefix) -> None:
        content = f"""! generated by ambrs.mam4.AerosolModel.write_input_files
&time_input
  mam_dt         = {input.mam_dt},
  mam_nstep      = {input.mam_nstep},
/
&cntl_input
  mdo_gaschem    = {input.mdo_gaschem},
  mdo_gasaerexch = {input.mdo_gasaerexch},
  mdo_rename     = {input.mdo_rename},
  mdo_newnuc     = {input.mdo_newnuc},
  mdo_coag       = {input.mdo_coag},
/
&met_input
  temp           = {input.temp},
  press          = {input.press},
  RH_CLEA        = {input.RH_CLEA},
/
&chem_input
  numc1          = {input.numc1}, ! unit: #/m3
  numc2          = {input.numc2},
  numc3          = {input.numc3},
  numc4          = {input.numc4},
  !
  ! mfABCx: mass fraction of species ABC in mode x.
  ! 
  ! The mass fraction of mom is calculated by
  ! 1 - sum(mfABCx). If sum(mfABCx) > 1, an error
  ! is issued by the test driver. number of species
  ! ABC in each mode x comes from the MAM4 with mom.
  ! 
  mfso41         = {input.mfso41},
  mfpom1         = {input.mfpom1},
  mfsoa1         = {input.mfsoa1},
  mfbc1          = {input.mfbc1},
  mfdst1         = {input.mfdst1},
  mfncl1         = {input.mfncl1},
  mfso42         = {input.mfso42},
  mfsoa2         = {input.mfsoa2},
  mfncl2         = {input.mfncl2},
  mfdst3         = {input.mfdst3},
  mfncl3         = {input.mfncl3},
  mfso43         = {input.mfso43},
  mfbc3          = {input.mfbc3},
  mfpom3         = {input.mfpom3},
  mfsoa3         = {input.mfsoa3},
  mfpom4         = {input.mfpom4},
  mfbc4          = {input.mfbc4},
  qso2           = {input.qso2},
  qh2so4         = {input.qh2so4},
  qsoag          = {input.qsoag},
/
"""
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')
        filename = os.path.join(dir, 'namelist')
        with open(filename, 'w') as f:
            print(content)
            f.write(content)

        content = """"""
        for mode, species, mf, dens in [('accumulation', p.name, input.mfs[0][p.name], p.density) for p in input.scenario.size.modes[0].species]: #  \
                                        # if p.name in ['ASO4','AH2O','AH3OP','SOA','APOC','AEC','ANA','ASOIL','H2SO4_aq'] \
                                        # else ('accumulation', p.name, 0., p.density) for p in input.scenario.aerosols]:
            content += \
f"""
{mode},{species},{mf},{dens}
"""
        for mode, species, mf, dens in [('aitken', p.name, input.mfs[1][p.name], p.density) for p in input.scenario.size.modes[1].species]: #  \
                                        # if p.name in ['ASO4','AH2O','AH3OP','SOA','ANA','H2SO4_aq'] \
                                        # else ('aitken', p.name, 0., p.density) for p in input.scenario.aerosols]:
            content += \
f"""
{mode},{species},{mf},{dens}
"""
        for mode, species, mf, dens in [('coarse', p.name, input.mfs[2][p.name], p.density) for p in input.scenario.size.modes[2].species]: #  \
                                        # if p.name in ['ASO4','AH2O','AH3OP','SOA','APOC','AEC','ANA','ASOIL','H2SO4_aq'] \
                                        # else ('coarse', p.name, 0., p.density) for p in input.scenario.aerosols]:
            content += \
f"""
{mode},{species},{mf},{dens}
"""
        for mode, species, mf, dens in [('primary_carbon', p.name, input.mfs[3][p.name], p.density) for p in input.scenario.size.modes[3].species]: # \
                                        #if p.name in ['APOC','AEC','AH2O','AH3OP','H2SO4_aq'] \
                                        #else ('primary_carbon', p.name, 0., p.density) for p in input.scenario.aerosols]:
            content += \
f"""
{mode},{species},{mf},{dens}
"""
        filename = os.path.join(dir, 'aero_mass_fracs.dat')
        with open(filename, 'w') as f:
            # print(content)
            f.write(content)

           
        content = ''''''

        for g, y in zip(input.scenario.gases, input.scenario.gas_concs):
            content += \
f'''
{g.name}, {y}
'''
        filename = os.path.join(dir, 'ic.dat')
        with open(filename, 'w') as f:
            # print(content)
            f.write(content)

        content = \
'''
&camp_config
    config_key = '/Users/duncancq/Research/AMBRS/aero_unit_tests/sulfate_condensation/mam4_config.json',
/
&camp_mech
    mech_key = 'sulfate_condensation',
/
'''
        filename = os.path.join(dir, 'camp_nml')
        with open(filename, 'w') as f:
            # print(content)
            f.write(content)


    def read_output_files(self,
                          input,
                          dir: str,
                          prefix: str) -> Output:
        # FIXME: we can parameterize our output processing to adjust the number
        # FIXME: of bins, etc (but not till things are more settled)
        num_modes = 4
        timestep = -1 # FIXME: for now, we analyze only the most recent timestep

        output_file = os.path.join(dir, 'mam_output.nc')
        nc_output = netCDF4.Dataset(output_file)

        # bin particles in each mode
        bins = np.logspace(-10, -5, 1000)
        lnDs = np.log(bins)
        Ns = nc_output['num_aer'][:,timestep]        # particle numbers
        mus = np.log(nc_output['dgn_a'][:,timestep]) # log of mean geometric diameters
        sigmas = [1.8, 1.6, 1.8, 1.6] # geometric diameter stddevs (hardwired into MAM4)
        dNdlnD_by_mode = np.zeros([num_modes, len(lnDs)])
        for k, (N, mu, sigma) in enumerate(zip(Ns,mus,sigmas)):
            dNdlnD_by_mode[k,:] = N * scipy.stats.norm(loc=mu, scale=sigma).pdf(lnDs)

        # FIXME: computing CCN seems a little complicated for now, but we'll get back to it
        return Output(
            model = self.name,
            input = input,
            bins = bins,
            dNdlnD = np.sum(dNdlnD_by_mode, axis=0),
        )
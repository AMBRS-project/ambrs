"""ambrs.mam4 -- data types and functions related to the MAM4 box model"""

from .aerosol import AerosolProcesses, AerosolModalSizePopulation, \
                     AerosolModalSizeState
from .aerosol_model import BaseAerosolModel
from .analysis import Output
from .gas import GasSpecies, build_gas_mixture

from .scenario import Scenario
from .ppe import Ensemble

from .camp import CampConfig
from typing import Optional
import json
from pathlib import Path


# fixme: put this in wrapper? load wrapper with .aerosol? 
from pyparticle import build_population
from dataclasses import dataclass
from netCDF4 import Dataset
import numpy as np
import os.path
import scipy.stats

@dataclass
class Input:
    """ambrs.mam4.Input -- an input dataclass for the MAM4 box model"""
    # all fields here are named identically to their respective namelist
    # parameters in the MAM4 box model

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

    # fixme: LMF addition -- better way to handle this?
    # --- CAMP helper fields (names only; used when writing CAMP config) ---
    aero_spec_names: list[str]
    gas_spec_names: list[str]

# this type handles the mapping of AMBRS aerosol species to MAM4 species
# within all aerosol modes
class AerosolMassFractions:
    @dataclass
    class AccumMode:
        SO4: float
        POM: float
        SOA: float
        BC: float
        DST: float
        NCL: float

    @dataclass
    class AitkenMode:
        SO4: float
        SOA: float
        NCL: float

    @dataclass
    class CoarseMode:
        DST: float
        NCL: float
        SO4: float
        BC: float
        POM: float
        SOA: float

    @dataclass
    class PCarbonMode:
        POM: float
        BC: float

    def __init__(self,
                 scenario: Scenario):
        self.accum = self.AccumMode(
            SO4 = scenario.size.modes[0].mass_fraction('SO4'),
            POM = scenario.size.modes[0].mass_fraction("OC"),
            SOA = scenario.size.modes[0].mass_fraction('MSA'), # FIXME: using MSA as a placeholder for SOA
            BC  = scenario.size.modes[0].mass_fraction("BC"),
            DST = scenario.size.modes[0].mass_fraction("OIN"),
            NCL = scenario.size.modes[0].mass_fraction("Na"), # FIXME: and "Cl"? Assuming 1:1 for now
        )
        
        self.aitken = self.AitkenMode(
            SO4 = scenario.size.modes[1].mass_fraction("SO4"),
            SOA = scenario.size.modes[1].mass_fraction("MSA"), # FIXME: using MSA as a placeholder for SOA
            NCL = scenario.size.modes[1].mass_fraction("Na"), # FIXME: assuming 1:1 with Cl for now
        )
        self.coarse = self.CoarseMode(
            DST = scenario.size.modes[2].mass_fraction("OIN"),
            NCL = scenario.size.modes[2].mass_fraction("Na"), # FIXME: see above
            SO4 = scenario.size.modes[2].mass_fraction("SO4"),
            BC  = scenario.size.modes[2].mass_fraction("BC"),
            POM = scenario.size.modes[2].mass_fraction("OC"),
            SOA = scenario.size.modes[2].mass_fraction("MSA"), # FIXME: using MSA as a placeholder for SOA
        )
        self.pcarbon = self.PCarbonMode(
            POM = scenario.size.modes[3].mass_fraction("OC"),
            BC  = scenario.size.modes[3].mass_fraction("BC"),
        )
        
AIR_MW   = 28.9647   # g/mol, dry air
H2SO4_MW = 98.079    # g/mol
SO2_MW   = 64.066    # g/mol

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
        isoag = GasSpecies.find(scenario.gases, 'soag')
        # fixme: double-check MAM units
        self.SO2 = scenario.gas_concs[iso2] #* SO2_MW / AIR_MW
        self.H2SO4 = scenario.gas_concs[ih2so4] #* H2SO4_MW / AIR_MW
        self.SOAG = 0.0 if isoag == -1 else scenario.gas_concs[isoag]
    
class AerosolModel(BaseAerosolModel):
    def __init__(self,
                 processes: AerosolProcesses,
                 # FIXME: LMF addition -- better way to handle this?
                 camp: Optional[CampConfig] = None):
        BaseAerosolModel.__init__(self, 'mam4', processes)
        self.camp = camp  # None means “don’t use CAMP”

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
        
        mftot1 = (
            aero_mass_fracs.accum.SO4 + aero_mass_fracs.accum.POM + 
            aero_mass_fracs.accum.SOA + aero_mass_fracs.accum.BC + 
            aero_mass_fracs.accum.DST + aero_mass_fracs.accum.NCL)

        mftot2 = (
            aero_mass_fracs.aitken.SO4 + 
            aero_mass_fracs.aitken.SOA + 
            aero_mass_fracs.aitken.NCL)
        
        mftot3 = (
            aero_mass_fracs.coarse.SO4 + aero_mass_fracs.coarse.POM + 
            aero_mass_fracs.coarse.SOA + aero_mass_fracs.coarse.BC + 
            aero_mass_fracs.coarse.DST + aero_mass_fracs.coarse.NCL)

        mftot4 = (
            aero_mass_fracs.pcarbon.POM + aero_mass_fracs.pcarbon.BC)
        
        aero_names = []
        for mode in scenario.size.modes:
            for sp in mode.species:
                aero_names.append(getattr(sp, "name", str(sp)))

        gas_names = [getattr(gs, "name", str(gs)) for gs in scenario.gases]

        return Input(
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
            mfso41 = aero_mass_fracs.accum.SO4/mftot1,
            mfpom1 = aero_mass_fracs.accum.POM/mftot1,
            mfsoa1 = aero_mass_fracs.accum.SOA/mftot1,
            mfbc1  = aero_mass_fracs.accum.BC/mftot1,
            mfdst1 = aero_mass_fracs.accum.DST/mftot1,
            mfncl1 = aero_mass_fracs.accum.NCL/mftot1,

            mfso42 = aero_mass_fracs.aitken.SO4/mftot2,
            mfsoa2 = aero_mass_fracs.aitken.SOA/mftot2,
            mfncl2 = aero_mass_fracs.aitken.NCL/mftot2,
            
            mfdst3 = aero_mass_fracs.coarse.DST/mftot3,
            mfncl3 = aero_mass_fracs.coarse.NCL/mftot3,
            mfso43 = aero_mass_fracs.coarse.SO4/mftot3,
            mfbc3  = aero_mass_fracs.coarse.BC/mftot3,
            mfpom3 = aero_mass_fracs.coarse.POM/mftot3,
            mfsoa3 = aero_mass_fracs.coarse.SOA/mftot3,

            mfpom4 = aero_mass_fracs.pcarbon.POM/mftot4,
            mfbc4  = aero_mass_fracs.pcarbon.BC/mftot4,

            qso2 = gas_mixing_ratios.SO2,
            qh2so4 = gas_mixing_ratios.H2SO4,
            qsoag = gas_mixing_ratios.SOAG,

            aero_spec_names = aero_names,
            gas_spec_names = gas_names
        )
    
    def invocation(self, exe: str, prefix: str) -> str:
        """input.invocation(exe, prefix) -> a string defining the command invoking
the input with the given executable and input prefix, assuming that the current
working directory contains any needed input files."""
        return f'{exe}'

    def write_input_files(self, input, dir, prefix) -> None:

        # FIXME: LMF addition; double-check
        # create CAMP block if CAMP config is provided
        
        self._camp_env = None
#         if self.camp is not None:
#             # use scenario gases if available; fall back to default superset
#             try:
#                 gases = [g.name for g in input.scenario.gases]  # optional convenience
#             except Exception:
#                 gases = None

#             from pathlib import Path
#             # # fixme: LMF addition -- better way to handle this?
#             # file_list = self.camp.write_common_files(dir)
#             # self._camp_env = self.camp.runtime_env()

#             # Write under <dir>/camp and return .../camp/camp_files.json
#             camp_files_json = self.camp.write_for_model(Path(dir), model_name="mam4", gases=gases)
#             # Path to place in the namelist should be relative to run directory
#             rel_camp_path = os.path.relpath(camp_files_json, start=dir)
#             # MAM4 CAMP hook (branch `features/laura-camp-sync`): the driver reads this block
#             camp_block = f"""
# &camp_input
#   do_camp_chem   = 1,
#   camp_config    = "{rel_camp_path}",
# /
# """
        # --- CAMP integration (model-agnostic, no env tricks needed) ---
        if self.camp is not None:
            # Create/refresh <dir>/camp/* and get ABS path to .../camp_file_list.json
            # camp_list_path = self.camp.write_common_files(dir)
            camp_list_path = self.camp.write_for_model(dir, model_name="mam4")
            
            # Tell the MAM box driver where to find the CAMP config via namelist
            camp_block = f"""
&camp_input
  do_camp_chem = 1,
  camp_config  = "{camp_list_path}",
/
"""
        else:
            camp_block = ""

        
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
{camp_block}"""
        
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')
        filename = os.path.join(dir, 'namelist')
        with open(filename, 'w') as f:
            f.write(content)
    
    
def get_mam_input(
        varname,
        mam_input='../mam_refactor-main/standalone/tests/smoke_test.nl'):
    f_input = open(mam_input,'r')
    input_lines = f_input.readlines()
    yep = 0
    for oneline in input_lines:
        if varname in oneline:
            yep += 1 
            idx1,=np.where([hi == '=' for hi in oneline])
            idx2,=np.where([hi == ',' for hi in oneline])
            vardat = float(''.join([oneline[ii] for ii in range(idx1[0]+1,idx2[0])]))
    if yep == 0:
        print(varname,'is not a MAM input parameter')
    elif yep > 1:
        print('more than one line in ', mam_input, 'starts with', varname)
    return vardat

def retrieve_model_state(
        scenario_name: str, 
        scenario: Scenario, 
        timestep: int, 
        # t_eval: float, 
        # model_times: np.array,
        # fixme: remove this next entry? quick fix for now
        repeat_num: int=1, # option for Partmc; set to 1 for MAM4
        species_modifications: dict={},
        ensemble_output_dir: str='mam4_runs', 
        N_bins:int = 1000) -> Output: # data structure that allows species modifications in post-processing (e.g., treat some organics as light-absorbing)
    
    GMDs = []
    GSDs = []
    Ns = []
    aero_spec_names = []
    aero_spec_fracs = []
    for mode in scenario.size.modes:
        GMDs.append(mode.geom_mean_diam)
        GSDs.append(10.**mode.log10_geom_std_dev)
        Ns.append(mode.number)
        aero_spec_names_onemode = []
        aero_spec_fracs.append(mode.mass_fractions)
        for one_spec in scenario.size.modes[0].species:
            aero_spec_names_onemode.append(one_spec.name)
        aero_spec_names.append(aero_spec_names_onemode)
    
    if timestep == 0:
        raise ValueError('timestep=0 is invalid. Specify timestep = 1 for initial conditions')
    elif timestep == 1:
        scenario_dir = ensemble_output_dir + '/' + scenario_name + '/'
        # mam_input = scenario_dir + 'mam_input.nl'
        mam_input = scenario_dir + 'namelist'
        Ns = np.zeros([len(GSDs)])
        for kk in range(len(Ns)):
            Ns[kk] = get_mam_input(
                'numc' + str(kk+1),
                mam_input=mam_input)
        
        binned_lognormal_cfg = {
            'type': 'binned_lognormals',
            'D_min':1e-10, #fixme: option for +/- sigmas
            'D_max':1e-4,
            'N_bins': N_bins,
            'N': Ns,
            'GMD': GMDs,
            'GSD': GSDs,
            'aero_spec_names': aero_spec_names,
            'aero_spec_fracs': aero_spec_fracs,
            }
        particle_population = build_population(binned_lognormal_cfg)
        
        gas_cfg = {
            'H2SO4':get_mam_input(
                    'qh2so4',
                    mam_input=mam_input),
            'SO2':get_mam_input(
                    'qso2',
                    mam_input=mam_input),
            'units':'mole_ratio' # fixme: double-check
            }
        gas_mixture = build_gas_mixture(gas_cfg)
        
        thermodynamics = { 
            'T':scenario.temperature,
            'p':scenario.temperature,
            'RH':scenario.relative_humidity}
        
    else:
        output_filename = ensemble_output_dir + '/' + scenario_name + '/mam_output.nc'
        currnc = Dataset(output_filename)
        timestep = timestep - 2
        mam4_population_cfg = {
            'type':'mam4',
            'mam4_dir': ensemble_output_dir + '/' + scenario_name + '/',
            #'output_filename': output_filename,
            'timestep':timestep,
            'GSD':GSDs, #fixme: put in the correct GSD values!
            'D_min':1e-10, #fixme: option for +/- sigmas
            'D_max':1e-4,
            'N_bins':N_bins,
            'T':scenario.temperature,
            'p':scenario.pressure}
        
        particle_population = build_population(mam4_population_cfg)
        gas_cfg = {'H2SO4':currnc.variables['h2so4_gas'][timestep]}        
        # gas_cfg = {'SO2':currnc.variables['so2_gas'][timestep]}
        # gas_cfg = {'SOAG':currnc.variables['soa_gas'][timestep]}
        #gas_cfg['units'] = 'kg_per_kg' # todo: double-check
        gas_cfg['units'] = 'mole_ratio' # todo: double-check
        gas_mixture = build_gas_mixture(gas_cfg)
        
        thermodynamics = { 
            'T':scenario.temperature,
            'p':scenario.temperature,
            'RH':scenario.relative_humidity}
        
    # fixme: update model state
    return Output(
        model_name='mam4',
        scenario_name=scenario_name, 
        scenario=scenario,
        # time=t_eval,
        timestep=timestep,
        particle_population=particle_population,
        gas_mixture=gas_mixture,
        thermodynamics=thermodynamics,
        )
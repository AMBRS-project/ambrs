"""ambrs.mam4 -- data types and functions related to the MAM4 box model"""

from .aerosol import AerosolProcesses, AerosolModalSizePopulation, \
                     AerosolModalSizeState
from .aerosol_model import BaseAerosolModel
from .gas import GasSpecies
from .scenario import Scenario
from .ppe import Ensemble

from dataclasses import dataclass
import os.path

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

class AerosolModel(BaseAerosolModel):
    def __init__(self, processes: AerosolProcesses):
        BaseAerosolModel.__init__(self, processes)

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
        iso2 = GasSpecies.find(scenario.gases, 'so2')
        if iso2 == -1:
            raise ValueError("SO2 gas ('so2') not found in gas species")
        ih2so4 = GasSpecies.find(scenario.gases, 'h2so4')
        if ih2so4 == -1:
            raise ValueError("H2SO4 gas ('h2so4') not found in gas species")
        isoag = GasSpecies.find(scenario.gases, 'soag')
        if isoag == -1:
            raise ValueError("SOAG gas ('soag') not found in gas species")
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

            mfso41 = scenario.size.modes[0].mass_fraction("so4"),
            mfpom1 = scenario.size.modes[0].mass_fraction("pom"),
            mfsoa1 = scenario.size.modes[0].mass_fraction("soa"),
            mfbc1  = scenario.size.modes[0].mass_fraction("bc"),
            mfdst1 = scenario.size.modes[0].mass_fraction("dst"),
            mfncl1 = scenario.size.modes[0].mass_fraction("ncl"),

            mfso42 = scenario.size.modes[1].mass_fraction("so4"),
            mfsoa2 = scenario.size.modes[1].mass_fraction("soa"),
            mfncl2 = scenario.size.modes[1].mass_fraction("ncl"),

            mfdst3 = scenario.size.modes[2].mass_fraction("dst"),
            mfncl3 = scenario.size.modes[2].mass_fraction("ncl"),
            mfso43 = scenario.size.modes[2].mass_fraction("so4"),
            mfbc3  = scenario.size.modes[2].mass_fraction("bc"),
            mfpom3 = scenario.size.modes[2].mass_fraction("pom"),
            mfsoa3 = scenario.size.modes[2].mass_fraction("soa"),

            mfpom4 = scenario.size.modes[3].mass_fraction("pom"),
            mfbc4  = scenario.size.modes[3].mass_fraction("bc"),

            qso2 = scenario.gas_concs[iso2],
            qh2so4 = scenario.gas_concs[ih2so4],
            qsoag = scenario.gas_concs[isoag],
        )

    def invocation(self, exe: str, prefix: str) -> str:
        """input.invocation(exe, prefix) -> a string defining the command invoking
the input with the given executable and input prefix, assuming that the current
working directory contains any needed input files."""
        return f'{exe}'

    def read_output_files(self, dir: str, prefix: str) -> None:
        pass

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
            f.write(content)

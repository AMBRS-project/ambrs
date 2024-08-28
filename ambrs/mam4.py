"""ambrs.mam4 -- data types and functions related to the MAM4 box model"""

from dataclasses import dataclass

from .aerosol import AerosolProcesses, AerosolModalSizePopulation, \
                     AerosolModalSizeState
from .gas import GasSpecies
from .scenario import Scenario
from .ppe import Ensemble

import os.path

@dataclass
class MAM4Input:
    """MAM4Input -- represents input for a single MAM4 box model simulation"""
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

    def write_files(self, dir, prefix):
        """input.write_files(dir, prefix) -> writes MAM4 box model input files
to the given directory with the given prefix"""
        content = f"""
&time_input
  mam_dt         = {self.mam_dt},
  mam_nstep      = {self.mam_nstep},
/
&cntl_input
  mdo_gaschem    = {self.mdo_gaschem},
  mdo_gasaerexch = {self.mdo_gasaerexch},
  mdo_rename     = {self.mdo_rename},
  mdo_newnuc     = {self.mdo_newnuc},
  mdo_coag       = {self.mdo_coag},
/
&met_input
  temp           = {self.temp},
  press          = {self.press},
  RH_CLEA        = {self.RH_CLEA},
/
&chem_input
  numc1          = {self.numc1}, ! unit: #/m3
  numc2          = {self.numc2},
  numc3          = {self.numc3},
  numc4          = {self.numc4},
  !
  ! mfABCx: mass fraction of species ABC in mode x.
  ! 
  ! The mass fraction of mom is calculated by
  ! 1 - sum(mfABCx). If sum(mfABCx) > 1, an error
  ! is issued by the test driver. number of species
  ! ABC in each mode x comes from the MAM4 with mom.
  ! 
  mfso41         = {self.mfso41},
  mfpom1         = {self.mfpom1},
  mfsoa1         = {self.mfsoa1},
  mfbc1          = {self.mfbc1},
  mfdst1         = {self.mfdst1},
  mfncl1         = {self.mfncl1},
  mfso42         = {self.mfso42},
  mfsoa2         = {self.mfsoa2},
  mfncl2         = {self.mfncl2},
  mfdst3         = {self.mfdst3},
  mfncl3         = {self.mfncl3},
  mfso43         = {self.mfso43},
  mfbc3          = {self.mfbc3},
  mfpom3         = {self.mfpom3},
  mfsoa3         = {self.mfsoa3},
  mfpom4         = {self.mfpom4},
  mfbc4          = {self.mfbc4},
  qso2           = {self.qso2},
  qh2so4         = {self.qh2so4},
  qsoag          = {self.qsoag},
/
"""
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')
        filename = os.path.join(dir, f'{prefix}.nl')
        with open(filename, 'w') as f:
            f.write(content)

def _mam4_input(processes: AerosolProcesses,
                scenario: Scenario,
                dt: float,
                nstep: int) -> MAM4Input:
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
    return MAM4Input(
        mam_dt = dt,
        mam_nstep = nstep,

        mdo_gaschem = False,
        mdo_gasaerexch = False,
        mdo_rename = False,
        mdo_newnuc = processes.nucleation,
        mdo_coag = processes.coagulation,

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

def create_mam4_input(processes: AerosolProcesses,
                      scenario: Scenario,
                      dt: float,
                      nstep: int) -> MAM4Input:
    """create_mam4_input(processes, scenario, dt, nstep) -> MAM4Input object
that can create a namelist input file for a MAM4 box model simulation

Parameters:
    * processes: an ambrs.AerosolProcesses object that defines the aerosol
      processes under consideration
    * scenario: an ambrs.Scenario object created by sampling a modal particle
      size distribution
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation"""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nstep <= 0:
        raise ValueError("nstep must be positive")
    return _mam4_input(processes, scenario, dt, nstep)

def create_mam4_inputs(processes: AerosolProcesses,
                       ensemble: Ensemble,
                       dt: float,
                       nstep: int) -> list[MAM4Input]:
    """create_mam4_inputs(processes, ensemble, dt, nstep) -> list of MAM4Input
objects that can create namelist input files for MAM4 box model simulations

Parameters:
    * processes: an ambrs.AerosolProcesses object that defines the aerosol
      processes under consideration
    * ensemble: a ppe.Ensemble object created by sampling a modal particle size
      distribution
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation"""
    if not isinstance(ensemble.size, AerosolModalSizePopulation):
        raise TypeError("ensemble must have a modal size distribution!")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nstep <= 0:
        raise ValueError("nstep must be positive")
    inputs = []
    for scenario in ensemble:
        inputs.append(_mam4_input(processes, scenario, dt, nstep))
    return inputs

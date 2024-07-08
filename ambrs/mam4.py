"""ambrs.mam4 -- data types and functions related to the MAM4 box model"""

from dataclasses import dataclass

from .aerosol import AerosolProcesses, AerosolModalSizePopulation
from .scenario import Scenario
from .ppe import Ensemble

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
    mfsoa4: float

    # mode 4 (primary carbon mode)
    mfpom4: float
    mfbc4: float

    #---------------------------------------
    # gas mixing ratios [kg gas/kg dry air]
    #---------------------------------------

    qso2: float
    qh2so4: float
    qsoag: float

    def write_namelist(self, filename):
        """input.write_namelist(filename) -> writes a MAM4 box model namelist
file with the given name"""
        content = f"""
&time_input
  mam_dt         = {self.dt},
  mam_nstep      = {self.nstep},
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
        with open(filename, 'w') as f:
            f.write(content)

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
        pressure = 10e5 # FIXME: obtain from temperature, relative humidity?
        inputs.append(MAM4Input(
            mam_dt = dt,
            mam_nstep = nstep,

            mdo_gaschem = False,
            mdo_gasaerexch = False,
            mdo_rename = False,
            mdo_newnuc = processes.nucleation,
            mdo_coag = processes.coagulation,

            temp = scenario.temperature,
            press = pressure,
            RH_CLEA = scenario.relative_humidity,

            numc1 = scenario.modes[0].number,
            numc2 = scenario.modes[1].number,
            numc3 = scenario.modes[2].number,
            numc4 = scenario.modes[3].number,

            mfso41 = scenario.modes[0].mass_fraction("so4"),
            mfpom1 = scenario.modes[0].mass_fraction("pom"),
            mfsoa1 = scenario.modes[0].mass_fraction("soa"),
            mfbc1  = scenario.modes[0].mass_fraction("bc"),
            mfdst1 = scenario.modes[0].mass_fraction("dst"),
            mfncl1 = scenario.modes[0].mass_fraction("ncl"),

            mfso42 = scenario.modes[1].mass_fraction("so4"),
            mfsoa2 = scenario.modes[1].mass_fraction("soa"),
            mfncl2 = scenario.modes[1].mass_fraction("ncl"),

            mfdst3 = scenario.modes[2].mass_fraction("dst"),
            mfncl3 = scenario.modes[2].mass_fraction("ncl"),
            mfso43 = scenario.modes[2].mass_fraction("so4"),
            mfbc3  = scenario.modes[2].mass_fraction("bc"),
            mfpom3 = scenario.modes[2].mass_fraction("pom"),
            mfsoa4 = scenario.modes[2].mass_fraction("soa"),

            mfpom4 = scenario.modes[3].mass_fraction("pom"),
            mfbc4  = scenario.modes[3].mass_fraction("bc"),

            # FIXME: what to do about gases?
            qso2 = 0,
            qh2so4 = 0,
            qsoag = 0,
        ))
    return inputs

"""ambrs.aero_model -- Tools for the integration of aerosol models with the
AMBRS framework"""

from .aerosol import AerosolProcesses
from .ppe import Ensemble
from .scenario import Scenario

class NotImplementedError(Exception):
    """This exception is emitted when an ambrs.AerosolModel method is called
that has not been overridden by a derived class."""

class BaseAerosolModel:
    """ambrs.BaseAerosolModel -- a base class that defines an interface to be
overridden to incorporate an aerosol model into the AMBRS framework.

To add an aerosol model to AMBRS:

1. Derive a class from this one and override its methods, including the
   constructor (__init__)
2. Define an input dataclass that holds all necessary parameters for defining
   input for your aerosol model. An instance of this dataclass is created from
   AerosolModel.create_input and passed to Aerosol.write_input_files.
"""

    def __init__(self):
        """ambrs.BaseAerosolModel.__init__(self) - base class constructor
Call this constructor in your derived aerosol model class's __init__ method in
the usual Python way."""
        pass

    def create_input(self,
                     processes: AerosolProcesses,
                     scenario: Scenario,
                     dt: float,
                     nstep: int):
        """ambrs.BaseAerosolModel.create_input(processes, scenario, dt, nstep) ->
an instance of the relevant input dataclass for an aerosol model that holds data
used to create input files for a scenario

Parameters:
    * processes: an ambrs.AerosolProcesses object that defines the aerosol
      processes under consideration
    * scenario: an ambrs.Scenario object defining an individual scenario
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation"""
        raise NotImplementedError('BaseAerosolModel.create_input not overridden!')

    def create_inputs(self,
                      processes: AerosolProcesses,
                      ensemble: Ensemble,
                      dt: float,
                      nstep: int) -> list:
        """ambrs.BaseAerosolModel.create_inputs(processes, ensemble, dt, nstep) -> list of
relevant input dataclasses that can create input files for an entire ensemble

Parameters:
    * processes: an ambrs.AerosolProcesses object that defines the aerosol
      processes under consideration
    * ensemble: a ppe.Ensemble object created by sampling a modal particle size
      distribution
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation

You don't have to override this method--it uses self.create_input() to create
input dataclasses for each individual scenario."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if nstep <= 0:
            raise ValueError("nstep must be positive")
        inputs = []
        for scenario in ensemble:
            inputs.append(self.create_input(processes, scenario, dt, nstep))
        return inputs

    def invocation(self, exe: str, prefix: str) -> str:
        """ambrs.BaseAerosolModel.invocation(exe, prefix) -> command string
Override this method to provide the command invoking the aerosol model on the
command line given
    * exe: a path to the aerosol model executable
    * prefix: a prefix that identifies the main input file

You may assume that this command is issued in the directory in which all necessary
input files reside."""
        raise NotImplementedError('BaseAerosolModel.invocation not overridden!')

    def read_output_files(self, dir: str, prefix: str) -> dict:
        """ambrs.BaseAerosolModel.read_output_files(dir, prefix) -> dict
containing diagnostic information about the final state of the aerosol particle
distribution

Override this method to read the contents of output files for an aerosol model
and compute diagnostic quantities for the aerosol model in its final state.

Input parameters:
    * dir: an absolute path to a directory containing the output files
    * prefix: a prefix used to identify the output files (if any)

Output diagnostics (key/value pairs in the dict returned by this method):
    -------------------------------------------------------------------------
    Key:        Value:
    -------------------------------------------------------------------------
    'dNdlnD'    a rank-3 numpy.array containing the logarithmic derivative of
                the number of particles of diameter D w.r.t. D
                (FIXME: describe indices here)
    -------------------------------------------------------------------------
"""
        raise NotImplementedError('BaseAerosolModel.read_output_files not overridden!')

    def write_input_files(self, input, dir: str, prefix: str) -> None:
        """ambrs.BaseAerosolModel.write_input_files(input, dir, prefix) -> None
Override this method to write input files for an aerosol model. Arguments:
    * input: an instance of your aerosol model's dataclass that defines the
             input parameters to be written to the files
    * dir: an absolute path to a directory where the files are written
    * prefix: a prefix used to identify the main input file"""
        raise NotImplementedError('BaseAerosolModel.write_input_files not overridden!')

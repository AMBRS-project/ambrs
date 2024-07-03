"""ambrs.scenario - types and functions supporting the creation of single
simulation scenarios.
"""

from dataclasses import dataclass
from .aerosol import AerosolModalSizeState

@dataclass
class Scenario:
    """Scenario: an abstract description of an aerosol configuration, expressed
in terms of state information"""
    size: AerosolModalSizeState # could also be sectional, stochastic, etc
    flux: float
    relative_humidity: float
    temperature: float


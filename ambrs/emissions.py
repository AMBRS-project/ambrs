"""emissions -- aerosol (and potentially gas) emissions and background inputs

We use frozen AerosolModelSizeState objects to define the emitted size distributions.

At this point, we are treating aerosol background and emissions the same way, which works for PartMC, but we may want a separate class for background later.
"""

from dataclasses import dataclass
from .aerosol import AerosolModalSizeState

@dataclass(frozen=True)
class AerosolEmissions:
    """AerosolEmissions: defines an aerosol emission or background input"""
    time: float
    rate: float
    size: AerosolModalSizeState

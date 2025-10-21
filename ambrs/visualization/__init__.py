"""Refactored visualization subpackage for AMBRS.

Legacy ridge/FastGrid plotting has been replaced by PyParticle-based helpers.
Only the new PyParticle line/grid comparison utilities are exported.

If legacy functions are still needed temporarily, import from
`ambrs.visualization_legacy.output_plot_legacy` explicitly.
"""

from .output_pyparticle import (
    plot_ensemble_size_distributions,
    plot_ensemble_variables,
)
from .input_ranges import plot_range_bars  # still useful for input range summaries

__all__ = [
    'plot_ensemble_size_distributions',
    'plot_ensemble_variables',
    'plot_range_bars',
]

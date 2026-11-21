"""The visualization helper's requested coordinates reach actual analyses."""

import numpy as np
import pytest
from part2pop import build_population
from part2pop.analysis import build_variable

from ambrs import viz
from ambrs.analysis import Output


@pytest.mark.parametrize("method", ["hist", "kde"])
@pytest.mark.parametrize("count", [7, 50])
def test_requested_diameter_grid_reaches_part2pop_and_output(method, count):
    population = build_population({
        "type": "monodisperse", "N": [1.0e8, 2.0e8],
        "D": [80.0e-9, 150.0e-9], "D_is_wet": False,
        "aero_spec_names": [["SO4"], ["SO4"]],
        "aero_spec_fracs": [[1.0], [1.0]],
    })
    config = viz.make_dNdlnD_cfg(
        D_range=(20.0e-9, 500.0e-9), N_bins=count, method=method,
    )
    result = build_variable("dNdlnD", var_cfg=config).compute(population, as_dict=True)
    expected = np.logspace(np.log10(20.0e-9), np.log10(500.0e-9), count)
    np.testing.assert_allclose(result["D"], expected, rtol=1.0e-13, atol=0.0)
    output = Output("synthetic", "case", None, 1, population, None, {})
    actual = output.compute_variable("dNdlnD", config)
    assert np.asarray(actual).shape == (count,)
    np.testing.assert_allclose(actual, result["dNdlnD"])

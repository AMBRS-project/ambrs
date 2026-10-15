"""Regression tests for writing aerosol emissions and background events."""
from pathlib import Path
from tempfile import TemporaryDirectory

from ambrs.aerosol import AerosolModeState, AerosolModalSizeState, AerosolProcesses, AerosolSpecies
from ambrs.emissions import AerosolEmissions
from ambrs.partmc import AerosolModel


def test_nonempty_aerosol_time_series_writes_distribution_and_composition():
    bc = AerosolSpecies(name="BC", molar_mass=12.01, density=1000.0)
    state = AerosolModalSizeState(modes=(AerosolModeState(
        name="emitted", species=(bc,), number=2e8, geom_mean_diam=1e-7,
        log10_geom_std_dev=0.2, mass_fractions=(1.0,),
    ),))
    event = AerosolEmissions(time=30.0, rate=0.01, size=state)
    model = AerosolModel(processes=AerosolProcesses(), n_part=10)
    for prefix in ("aero_emit", "aero_back"):
        with TemporaryDirectory() as tmp:
            model._write_aerosol_time_series(tmp, prefix, [event])
            contents = (Path(tmp) / f"{prefix}.dat").read_text()
            dist = Path(tmp) / f"{prefix}_dist_1_dist.dat"
            composition = Path(tmp) / f"{prefix}_dist_1_comp_1.dat"
            assert "time\t30.0" in contents
            assert "rate\t0.01" in contents
            assert dist.name in contents and dist.is_file()
            assert "mode_name emitted" in dist.read_text()
            assert "BC\t1.0" in composition.read_text()

"""Workflow-oriented AMBRS helpers.

These helpers provide two capabilities needed by coupled aerosol workflows while
keeping the existing AMBRS API unchanged:

* ``lhs``: seeded Latin-hypercube sampling with one distinct factor column for
  every modal size/composition variable, gas concentration, and environmental
  scalar.
* ``PartMCAerosolModel``: a thin PartMC adapter that forwards gas background
  data and writes ``AerosolEmissions.size.modes`` correctly.

The functions are intentionally additive so coupled workflows can use them now
without changing established AMBRS behavior. They can be folded into the core
``ppe`` and ``partmc`` implementations in a later compatibility release.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyDOE

from .aerosol import AerosolModalSizeDistribution, AerosolModalSizePopulation, AerosolModePopulation
from .partmc import AerosolModel as CorePartMCAerosolModel
from .ppe import Ensemble, EnsembleSpecification


def _sample_lhs(n_factors: int, n: int, criterion=None, iterations=None,
                seed: Optional[int] = None) -> np.ndarray:
    """Create an LHS without changing the caller-visible NumPy RNG state."""
    if seed is None:
        return pyDOE.lhs(n_factors, n, criterion, iterations)

    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        return pyDOE.lhs(n_factors, n, criterion, iterations)
    finally:
        np.random.set_state(rng_state)


def lhs(specification: EnsembleSpecification,
        n: int,
        criterion=None,
        iterations=None,
        seed: Optional[int] = None) -> Ensemble:
    """Generate a reproducible Latin-hypercube AMBRS ensemble.

    Unlike the legacy ``ambrs.ppe.lhs`` implementation, this function advances
    a cumulative factor index. Consequently each of the following occupies a
    distinct Latin-hypercube dimension:

    * modal number concentration,
    * modal geometric-mean diameter,
    * modal geometric standard deviation,
    * every modal species mass-fraction coordinate,
    * every gas initial concentration,
    * flux, relative humidity, temperature, and pressure.

    Mass-fraction coordinates retain AMBRS's existing behavior: NaNs are mapped
    to zero and the independently sampled coordinates are normalized to sum to
    one for each ensemble member.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if not isinstance(specification.size, AerosolModalSizeDistribution):
        raise TypeError("Latin-hypercube sampling currently requires modal aerosol size distributions")

    n_factors = len(specification.gas_concs) + 4
    for mode in specification.size.modes:
        n_factors += 3 + len(mode.mass_fractions)

    design = _sample_lhs(n_factors, n, criterion, iterations, seed)
    column = 0
    modes = []

    for mode in specification.size.modes:
        number = np.asarray(mode.number.ppf(design[:, column])) * np.ones(n)
        column += 1
        geom_mean_diam = np.asarray(mode.geom_mean_diam.ppf(design[:, column])) * np.ones(n)
        column += 1
        log10_geom_std_dev = np.asarray(
            mode.log10_geom_std_dev.ppf(design[:, column])
        ) * np.ones(n)
        column += 1

        mass_fractions = []
        for distribution in mode.mass_fractions:
            values = np.asarray(distribution.ppf(design[:, column])) * np.ones(n)
            values[np.isnan(values)] = 0.0
            mass_fractions.append(values)
            column += 1

        normalizer = sum(mass_fractions)
        if np.any(normalizer <= 0.0):
            raise ValueError("Sampled aerosol mass fractions sum to zero for one or more members")
        mass_fractions = tuple(values / normalizer for values in mass_fractions)

        modes.append(
            AerosolModePopulation(
                name=mode.name,
                species=mode.species,
                number=number,
                geom_mean_diam=geom_mean_diam,
                log10_geom_std_dev=log10_geom_std_dev,
                mass_fractions=mass_fractions,
            )
        )

    size = AerosolModalSizePopulation(modes=tuple(modes))

    gas_concs = []
    for gas_conc in specification.gas_concs:
        gas_concs.append(np.asarray(gas_conc.ppf(design[:, column])) * np.ones(n))
        column += 1

    flux = np.asarray(specification.flux.ppf(design[:, column])) * np.ones(n)
    column += 1
    relative_humidity = np.asarray(
        specification.relative_humidity.ppf(design[:, column])
    ) * np.ones(n)
    column += 1
    temperature = np.asarray(specification.temperature.ppf(design[:, column])) * np.ones(n)
    column += 1
    pressure = np.asarray(specification.pressure.ppf(design[:, column])) * np.ones(n)
    column += 1

    if column != n_factors:
        raise RuntimeError(
            f"Internal LHS indexing error: consumed {column} of {n_factors} factors"
        )

    return Ensemble(
        aerosols=specification.aerosols,
        gases=specification.gases,
        specification=specification,
        size=size,
        gas_concs=tuple(gas_concs),
        flux=flux,
        relative_humidity=relative_humidity,
        temperature=temperature,
        pressure=pressure,
        height=specification.height,
        gas_emissions=specification.gas_emissions,
        gas_background=specification.gas_background,
        aerosol_emissions=specification.aerosol_emissions,
        aerosol_background=specification.aerosol_background,
    )


class PartMCAerosolModel(CorePartMCAerosolModel):
    """PartMC adapter with source/background plumbing needed by coupled workflows."""

    def create_input(self, scenario, dt: float, nstep: int, t_output: float | None = None):
        input_data = super().create_input(scenario, dt, nstep, t_output=t_output)
        input_data.gas_background = scenario.gas_background
        return input_data

    def _write_aerosol_time_series(self, dir: str, prefix: str, events):
        """Write aerosol source/background events using ``AerosolEmissions.size``."""
        import os

        file_path = os.path.join(dir, f"{prefix}.dat")
        if not events:
            with open(file_path, "w") as handle:
                handle.write("# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n")
                handle.write("time\t0.0\n")
                handle.write("rate\t0.0\n")
                handle.write("dist\taero_init_dist.dat\n")
            return

        mode_prefixes = [f"{prefix}_dist_{i + 1}" for i in range(len(events))]
        dist_files = [f"{mode_prefix}_dist.dat" for mode_prefix in mode_prefixes]
        with open(file_path, "w") as handle:
            handle.write("# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n")
            handle.write("\t".join(["time"] + [str(event.time) for event in events]) + "\n")
            handle.write("\t".join(["rate"] + [str(event.rate) for event in events]) + "\n")
            handle.write("\t".join(["dist"] + dist_files) + "\n")

        for i, event in enumerate(events):
            self._write_aero_modes(dir, mode_prefixes[i], event.size.modes)

"""Refactored visualization utilities using part2pop.viz

This module replaces legacy seaborn ridge/FacetGrid plotting with
declarative part2pop population + plotting helpers.

Public Functions
----------------
plot_ensemble_size_distributions(...):
    Per-scenario PartMC vs MAM4 dNdlnD line comparisons.

plot_ensemble_variables(...):
    Multi-variable (e.g., dNdlnD + frac_ccn) overlay per scenario.

Notes
-----
* No synthetic data creation. Missing files raise FileNotFoundError.
* Styling kept minimal; callers can post-process axes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np

from part2pop import build_population  # generic factory
from part2pop.viz.plotting import plot_lines
from part2pop.viz.grids import make_grid_scenarios_models

try:
    # MAM4 specific builder (provisional path per user instructions)
    from part2pop.population.factory.mam4 import build as build_mam4
except ImportError:  # pragma: no cover
    build_mam4 = None  # type: ignore

__all__ = [
    'plot_ensemble_size_distributions',
    'plot_ensemble_variables',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_dir(path: Path, kind: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} path not found: {path}. Ensure model outputs are generated before plotting."
        )


def _partmc_builder(cfg: Dict) -> 'ParticlePopulation':  # type: ignore[name-defined]
    needed = {'type', 'partmc_dir', 'timestep', 'repeat', 'species_modifications'}
    part_cfg = {k: cfg[k] for k in needed if k in cfg}
    return build_population(part_cfg)


def _mam4_builder(cfg: Dict) -> 'ParticlePopulation':  # type: ignore[name-defined]
    if build_mam4 is None:
        raise ImportError("part2pop MAM4 builder not available; install part2pop with MAM4 extras.")
    # Accept either pre-built keys or minimal set if embedded in output file
    allowed = {'type', 'output_filename', 'timestep', 'GSD', 'D_min', 'D_max', 'N_bins', 'p', 'T'}
    mam_cfg = {k: cfg[k] for k in allowed if k in cfg}
    mam_cfg.setdefault('type', 'mam4')
    return build_mam4(mam_cfg)  # type: ignore[no-any-return]


def _scenario_cfgs(
    scenario_names: Sequence[str],
    partmc_root: Path,
    mam4_root: Path,
    timestep: int,
    repeat: int,
    mam4_extra: Optional[Dict] = None,
) -> List[Dict]:
    cfgs: List[Dict] = []
    for sid in scenario_names:
        part_dir = partmc_root / sid
        mam_dir = mam4_root / sid
        _validate_dir(part_dir, 'PartMC scenario')
        # MAM4 may store a single NetCDF file per scenario
        mam_nc = mam_dir / 'mam_output.nc'
        if not mam_nc.exists():  # allow initial condition case
            # fall back to directory presence, but still require file for timestep>1
            if timestep != 1:
                raise FileNotFoundError(f"Missing MAM4 NetCDF for scenario {sid}: {mam_nc}")
        cfg: Dict = {
            'type': 'partmc',  # base type used by partmc builder
            'scenario': sid,
            'partmc_dir': str(part_dir),
            'timestep': timestep,
            'repeat': repeat,
            'species_modifications': {},
            'output_filename': str(mam_nc),
        }
        if mam4_extra:
            cfg.update(mam4_extra)
        cfgs.append(cfg)
    return cfgs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_ensemble_size_distributions(
    partmc_dir: str | Path,
    mam4_dir: str | Path,
    scenario_names: Sequence[str],
    *,
    timestep: int = 1,
    repeat: int = 0,
    dNdlnD_cfg: Optional[Dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    mam4_extra: Optional[Dict] = None,
):
    """Create a grid of size-distribution (dNdlnD) comparisons (PartMC vs MAM4).

    Each row = scenario; single column variable = dNdlnD.
    Returns (fig, axes).
    """
    partmc_root = Path(partmc_dir)
    mam4_root = Path(mam4_dir)
    scenario_cfgs = _scenario_cfgs(
        scenario_names, partmc_root, mam4_root, timestep=timestep, repeat=repeat, mam4_extra=mam4_extra
    )

    variables = ['dNdlnD']
    var_cfg = {'dNdlnD': dNdlnD_cfg or {}}

    fig, axes = make_grid_scenarios_models(
        scenario_cfgs,
        variables,
        model_cfg_builders=[_partmc_builder, _mam4_builder],
        var_cfg=var_cfg,
        figsize=figsize or (6, 2.2 * len(scenario_cfgs)),
    )

    # Add legend labels (assumes two lines per axis)
    for ax in np.atleast_1d(axes).ravel():
        lines = ax.lines
        if len(lines) >= 2:
            lines[0].set_label('PartMC')
            lines[1].set_label('MAM4')
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_ensemble_variables(
    partmc_dir: str | Path,
    mam4_dir: str | Path,
    scenario_names: Sequence[str],
    variables: Sequence[str],
    *,
    timestep: int = 1,
    repeat: int = 0,
    var_cfg: Optional[Dict[str, Dict]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    mam4_extra: Optional[Dict] = None,
):
    """General multi-variable PartMC vs MAM4 comparison grid.

    Parameters
    ----------
    variables : list of variable names accepted by part2pop.viz.plot_lines
        e.g., ['dNdlnD', 'frac_ccn']
    var_cfg : optional mapping variable -> config dict
    """
    partmc_root = Path(partmc_dir)
    mam4_root = Path(mam4_dir)
    scenario_cfgs = _scenario_cfgs(
        scenario_names, partmc_root, mam4_root, timestep=timestep, repeat=repeat, mam4_extra=mam4_extra
    )
    cfg_map = var_cfg or {}
    # ensure every variable has a dictionary (even empty) so downstream code doesn't fail
    prepared_cfg = {v: cfg_map.get(v, {}) for v in variables}

    fig, axes = make_grid_scenarios_models(
        scenario_cfgs,
        list(variables),
        model_cfg_builders=[_partmc_builder, _mam4_builder],
        var_cfg=prepared_cfg,
        figsize=figsize or (4 * len(variables), 2.2 * len(scenario_cfgs)),
    )
    for ax in np.atleast_1d(axes).ravel():
        lines = ax.lines
        if len(lines) >= 2:
            lines[0].set_label('PartMC')
            lines[1].set_label('MAM4')
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes

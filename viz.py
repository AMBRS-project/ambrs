"""Reusable visualization helpers for AMBRS ensembles.

Expose a single high-level function `visualize_ensemble` that builds
PartMC and MAM4 outputs (using the repository retrieval helpers) and
produces a PyParticle comparison grid and a range-bar summary. This
extracts the visualization code previously embedded in
`demo_revised.py` so it can be reused for other ensembles.
"""
from pathlib import Path
import warnings
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# These imports are optional at import-time because PyParticle may not
# be available in all environments; import errors should propagate when
# the function is actually used so callers see the failure early.
from PyParticle.viz.grids import make_grid_scenarios_models
from PyParticle.analysis import compute_variable

from ambrs.partmc import retrieve_model_state as retrieve_partmc
from ambrs.mam4 import retrieve_model_state as retrieve_mam4
from ambrs.visualization.input_ranges import plot_range_bars


logger = logging.getLogger(__name__)


def _default_var_cfg(v):
    if v == "dNdlnD":
        return {"wetsize": True, "N_bins": 40, "D_min": 1e-8, "D_max": 2e-6}
    if v in ("frac_ccn", "Nccn"):
        return {"s_eval": np.logspace(-2, 1, 40)}
    return {}


def visualize_ensemble(
    ensemble,
    partmc_dir: str,
    mam4_dir: str,
    scenario_names: Optional[Iterable[str]] = None,
    timestep_to_plot: int = 1,
    out_dir: Optional[str] = None,
    variables: Optional[Iterable[str]] = None,
    s_target: float = 0.01,
):
    """Create visualization outputs for an ensemble.

    Parameters
    - ensemble: the AMBRS ensemble object (must support .member and .flux)
    - partmc_dir, mam4_dir: directories where model outputs are stored
    - scenario_names: iterable of scenario name strings (defaults to 1..N)
    - timestep_to_plot: timestep index to retrieve
    - out_dir: directory to write figures (defaults to ./reports)
    - variables: list of variable names for the grid (defaults to ["dNdlnD","frac_ccn"]) 
    - s_target: supersaturation used for Nccn metric in range-bars

    Returns a dict with paths to saved figures and the generated matplotlib figures
    where available: {"grid": Path(...), "range_bars": Path(...), "fig": fig}
    """
    # defaults
    if variables is None:
        variables = ["dNdlnD", "frac_ccn"]

    if scenario_names is None:
        # attempt to infer number of scenarios from ensemble.flux
        try:
            n = len(ensemble.flux)
        except Exception:
            # fallback: try to iterate members until failure (unlikely)
            n = 0
            try:
                while True:
                    ensemble.member(n + 1)
                    n += 1
            except Exception:
                pass
        scenario_names = [str(ii).zfill(1) for ii in range(1, n)]

    out_path = Path(out_dir) if out_dir is not None else Path.cwd() / "reports"
    out_path.mkdir(parents=True, exist_ok=True)

    partmc_outputs = []
    mam4_outputs = []
    for scenario_name in scenario_names:
        # retrieve PartMC output
        p_out = retrieve_partmc(
            scenario_name=scenario_name,
            scenario=ensemble.member(int(scenario_name)),
            timestep=timestep_to_plot,
            repeat_num=1,
            species_modifications={},
            ensemble_output_dir=partmc_dir,
        )
        partmc_outputs.append(p_out)

        # retrieve MAM4 output
        m_out = retrieve_mam4(
            scenario_name=scenario_name,
            scenario=ensemble.member(int(scenario_name)),
            timestep=timestep_to_plot,
            repeat_num=1,
            species_modifications={},
            ensemble_output_dir=mam4_dir,
        )
        mam4_outputs.append(m_out)

    # Build scenario config dicts
    scenario_cfgs = []
    for sid, (p_out, m_out) in zip(scenario_names, zip(partmc_outputs, mam4_outputs)):
        scenario_cfgs.append({
            "scenario_name": sid,
            "partmc_output": p_out,
            "mam4_output": m_out,
        })

    var_cfg_mapping = {v: _default_var_cfg(v) for v in variables}

    def partmc_builder(cfg):
        pop = cfg["partmc_output"].particle_population
        try:
            pop.origin = "PartMC"
        except Exception:
            pass
        return pop

    def mam4_builder(cfg):
        pop = cfg["mam4_output"].particle_population
        try:
            pop.origin = "MAM4"
        except Exception:
            pass
        return pop

    # Suppress known benign warning from PyParticle hygroscopic growth
    warnings.filterwarnings(
        "ignore",
        message="Surface tension not implemented",
        category=UserWarning,
        module="PyParticle",
    )

    # Create grid figure
    fig, axes = make_grid_scenarios_models(
        scenario_cfgs,
        variables,
        model_cfg_builders=[partmc_builder, mam4_builder],
        var_cfg=var_cfg_mapping,
        figsize=(4 * len(variables), 3 * len(scenario_cfgs)),
    )

    fig.suptitle("PartMC vs MAM4 scenario comparison")
    fig.tight_layout()
    grid_out = out_path / "out_grid_partmc_mam4.png"
    fig.savefig(grid_out, dpi=180)
    logger.info(f"Wrote: {grid_out}")

    # -------------------------
    # Range-bar ensemble summary
    # -------------------------
    metrics = []
    for idx, p_out in enumerate(partmc_outputs):
        try:
            sd = compute_variable(population=p_out.particle_population, varname="dNdlnD", var_cfg={"N_bins": 40})
            dNdlnD = sd.get("dNdlnD") if isinstance(sd, dict) else sd
            total_N = float(dNdlnD.sum())
        except Exception as e:
            total_N = float('nan')
            logger.warning(f"Warning computing dNdlnD for scenario {scenario_names[idx]}: {e}")

        try:
            nccn = compute_variable(population=p_out.particle_population, varname="Nccn", var_cfg={"s_eval": [s_target]})
            Nccn_val = float(nccn.get("Nccn")[0]) if (isinstance(nccn, dict) and "Nccn" in nccn) else float('nan')
        except Exception as e:
            Nccn_val = float('nan')
            logger.warning(f"Warning computing Nccn for scenario {scenario_names[idx]}: {e}")

        metrics.append({"scenario": scenario_names[idx], "total_N": total_N, "Nccn_1pct": Nccn_val})

    # Build DataFrame in long format and drop NaNs
    rows = []
    for i, m in enumerate(metrics):
        rows.append({"variable": "total_N", "value": m["total_N"], "sample": i})
        rows.append({"variable": "Nccn_1pct", "value": m["Nccn_1pct"], "sample": i})
    df_metrics = pd.DataFrame(rows)
    before = len(df_metrics)
    df_metrics = df_metrics.dropna(subset=["value"]).reset_index(drop=True)
    dropped = before - len(df_metrics)
    if dropped:
        logger.info(f"Dropped {dropped} metric entries with NaN values before plotting range bars")

    range_rb_out = None
    fig_rb = None
    if df_metrics.empty:
        logger.info("No valid metric data available for range-bars; skipping.")
    else:
        present_samples = sorted(df_metrics['sample'].unique().tolist())
        highlight_colors = {s: f"C{ii % 10}" for ii, s in enumerate(present_samples)}
        fig_rb = plot_range_bars(df_metrics, ["total_N", "Nccn_1pct"], var_col="variable", value_col="value",
                                 highlight_idx=present_samples, highlight_colors=highlight_colors, figsize=(8, 4))
        range_rb_out = out_path / "out_range_bars.png"
        fig_rb.savefig(range_rb_out, dpi=180)
        logger.info(f"Wrote range-bars: {range_rb_out}")

    return {
        "grid": grid_out,
        "range_bars": range_rb_out,
        "fig": fig,
        "fig_range_bars": fig_rb,
    }

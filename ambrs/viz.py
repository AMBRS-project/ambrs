"""Package visualization helpers (moved inside `ambrs`).

This file mirrors the refactored visualization code and provides the
flexible highlighting and line-plot features requested.
"""
from pathlib import Path
import warnings
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from PyParticle.viz.grids import make_grid_scenarios_models
from PyParticle.analysis import compute_variable

from ambrs.partmc import retrieve_model_state as retrieve_partmc
from ambrs.mam4 import retrieve_model_state as retrieve_mam4


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
    highlight_count: int = 8,
    highlight_seed: Optional[int] = None,
    show_line_plots: bool = True,
    line_var: str = "dNdlnD",
    line_var_cfg: Optional[dict] = None,
):
    """Create visualization outputs for an ensemble.

    See module-level docs and previous implementation for parameter details.
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
        # retrieve PartMC output (best-effort)
        try:
            p_out = retrieve_partmc(
                scenario_name=scenario_name,
                scenario=ensemble.member(int(scenario_name)),
                timestep=timestep_to_plot,
                repeat_num=1,
                species_modifications={},
                ensemble_output_dir=partmc_dir,
            )
        except Exception as e:
            logger.warning(f"Could not retrieve PartMC output for scenario {scenario_name}: {e}")
            p_out = None
        partmc_outputs.append(p_out)

        # retrieve MAM4 output (best-effort)
        try:
            m_out = retrieve_mam4(
                scenario_name=scenario_name,
                scenario=ensemble.member(int(scenario_name)),
                timestep=timestep_to_plot,
                repeat_num=1,
                species_modifications={},
                ensemble_output_dir=mam4_dir,
            )
        except Exception as e:
            logger.warning(f"Could not retrieve MAM4 output for scenario {scenario_name}: {e}")
            m_out = None
        mam4_outputs.append(m_out)

    # Build scenario config dicts
    scenario_cfgs = []
    for sid, (p_out, m_out) in zip(scenario_names, zip(partmc_outputs, mam4_outputs)):
        # Only include scenarios that have both outputs available for the grid
        if (p_out is not None) and (m_out is not None):
            scenario_cfgs.append({
                "scenario_name": sid,
                "partmc_output": p_out,
                "mam4_output": m_out,
            })
        else:
            logger.info(f"Skipping grid plotting for scenario {sid} (missing outputs)")

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

    # Create grid figure using PyParticle helper (keeps previous behaviour)
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
    # Range-bar / input-ranges ensemble summary (customized)
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
        # Determine samples and pick highlighted subset
        samples = sorted(df_metrics['sample'].unique().tolist())
        n_samples = len(samples)
        rng = random.Random(highlight_seed)
        highlighted = rng.sample(samples, min(highlight_count, n_samples)) if n_samples else []

        # Assign colors to highlighted samples
        base_colors = [f"C{i % 10}" for i in range(len(highlighted))]
        highlight_colors = {s: base_colors[i] for i, s in enumerate(highlighted)}

        # Plot a simple input_ranges-like figure: for each variable show all
        # sample values (light/white markers) and overlay highlighted ones in color.
        fig_rb, axes_rb = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4), squeeze=False)
        axes_rb = axes_rb[0]
        for ax, var in zip(axes_rb, variables):
            # collect values per sample in order
            vals = []
            for s in samples:
                v = df_metrics[(df_metrics['sample'] == s) & (df_metrics['variable'] == var)]['value']
                if v.empty:
                    vals.append(float('nan'))
                else:
                    vals.append(float(v.iloc[0]))

            x = np.arange(len(samples))
            # plot all points as small white markers with gray edge
            ax.scatter(x, vals, color='white', edgecolor='lightgray', s=25, zorder=1)

            # overlay highlighted points in color
            for s in highlighted:
                idx = samples.index(s)
                v = vals[idx]
                ax.scatter(idx, v, color=highlight_colors[s], edgecolor='k', s=40, zorder=2, label=f"{s}")

            ax.set_title(var)
            ax.set_xlabel('scenario')
            ax.set_xticks(x)
            # improve layout for many samples
            if len(samples) > 20:
                ax.set_xticklabels([str(i) for i in samples], rotation=90, fontsize=6)
            else:
                ax.set_xticklabels([str(i) for i in samples])

        range_rb_out = out_path / "out_range_bars_custom.png"
        fig_rb.tight_layout()
        fig_rb.savefig(range_rb_out, dpi=180)
        logger.info(f"Wrote custom range-bars: {range_rb_out}")

        # Optionally produce line plots for highlighted scenarios using the same colors
        fig_lines = None
        fig_lines_out = None
        if show_line_plots and highlighted:
            # Choose var_cfg for line plotting (dNdlnD)
            if line_var_cfg is None:
                line_var_cfg = _default_var_cfg(line_var)

            fig_lines, ax_lines = plt.subplots(1, 1, figsize=(6, 4))
            for s in highlighted:
                # retrieve corresponding partmc output
                idx = samples.index(s)
                p_out = partmc_outputs[idx]
                try:
                    sd = compute_variable(population=p_out.particle_population, varname=line_var, var_cfg=line_var_cfg)
                    d = sd.get(line_var) if isinstance(sd, dict) else sd
                except Exception as e:
                    logger.warning(f"Error computing {line_var} for scenario {s}: {e}")
                    continue

                # derive x-axis (diameter) from var_cfg if possible
                N_bins = int(line_var_cfg.get('N_bins', 40))
                D_min = float(line_var_cfg.get('D_min', 1e-8))
                D_max = float(line_var_cfg.get('D_max', 2e-6))
                D_centers = np.logspace(np.log10(D_min), np.log10(D_max), N_bins)

                ax_lines.plot(D_centers, d, color=highlight_colors[s], label=f"scenario {s}")

            ax_lines.set_xscale('log')
            ax_lines.set_xlabel('D (m)')
            ax_lines.set_ylabel(line_var)
            ax_lines.legend()
            fig_lines.tight_layout()
            fig_lines_out = out_path / "highlighted_lines.png"
            fig_lines.savefig(fig_lines_out, dpi=180)
            logger.info(f"Wrote highlighted lines: {fig_lines_out}")

        fig_rb = fig_rb
        fig_range_bars = fig_lines

    return {
        "grid": grid_out,
        "range_bars": range_rb_out,
        "fig": fig,
        "fig_range_bars": fig_rb,
    }

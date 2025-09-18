"""
Demo script to reproduce ensembles for MAM4 and PartMC
and generate ensemble plots.

Based on:
Fierce et al., Quantifying structural errors in cloud condensation nuclei activity
from reduced representation of aerosol size distributions,
J. Aerosol Science 181 (2024) 106388
https://doi.org/10.1016/j.jaerosci.2024.106388
"""

import os
import logging
from math import log10

import numpy as np
import scipy.stats as stats

import ambrs

# -----------------------------------------------------------
# Setup logging
# -----------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------
# Run configuration
# -----------------------------------------------------------
ensemble_name = "1h_5k"
root_dir = os.path.join("/Users/fier887/Downloads/ambrs_runs", ensemble_name)

partmc_dir = os.path.join(root_dir, "partmc_runs")
mam4_dir = os.path.join(root_dir, "mam4_runs")

for d in [root_dir, partmc_dir, mam4_dir]:
    os.makedirs(d, exist_ok=True)

n = 5        # number of ensemble members
n_part = 5000  # PartMC particles per run

dt = 60      # timestep [s]
nstep = 60   # 60 min run

p0 = 101325  # reference pressure [Pa]
h0 = 500     # reference height [m]

# -----------------------------------------------------------
# Define species
# -----------------------------------------------------------
so4 = ambrs.AerosolSpecies("SO4", molar_mass=96., density=1800., hygroscopicity=0.65)
pom = ambrs.AerosolSpecies("OC", molar_mass=12.01, density=1000., hygroscopicity=0.001)
soa = ambrs.AerosolSpecies("MSA", molar_mass=40., density=2600., hygroscopicity=0.53)
bc  = ambrs.AerosolSpecies("BC",  molar_mass=12.01, density=1800., hygroscopicity=0.)
dst = ambrs.AerosolSpecies("OIN", molar_mass=135.065, density=2600., hygroscopicity=0.1)
na  = ambrs.AerosolSpecies("Na",  molar_mass=23., density=2200., hygroscopicity=0.53)
cl  = ambrs.AerosolSpecies("Cl",  molar_mass=35.5, density=2200., hygroscopicity=0.53)
ncl = na
h2o = ambrs.AerosolSpecies("H2O", molar_mass=18., density=1000., ions_in_soln=1)

so2   = ambrs.GasSpecies("SO2", molar_mass=64.07)
h2so4 = ambrs.GasSpecies("H2SO4", molar_mass=98.079)

# -----------------------------------------------------------
# Define aerosol processes
# -----------------------------------------------------------
processes = ambrs.AerosolProcesses(
    coagulation=True,
    condensation=True,
    # gas_phase_chemistry=True,
    # aqueous_chemistry=True,
)

# -----------------------------------------------------------
# Ensemble specification
# -----------------------------------------------------------
spec = ambrs.EnsembleSpecification(
    name=ensemble_name,
    aerosols=(so4, pom, soa, bc, dst, ncl, h2o),
    gases=(so2, h2so4),
    size=ambrs.AerosolModalSizeDistribution(modes=[
        ambrs.AerosolModeDistribution(
            name="accumulation",
            species=[so4, pom, soa, bc, dst, ncl],
            number=stats.uniform(1e7, 1e10),
            geom_mean_diam=stats.rv_discrete(values=([1.1e-7], [1.])),
            log10_geom_std_dev=log10(1.6),
            mass_fractions=[
                stats.rv_discrete(values=([1.], [1.])),  # SO4
                stats.rv_discrete(values=([0.], [1.])),  # POM
                stats.rv_discrete(values=([0.], [1.])),  # SOA
                stats.rv_discrete(values=([0.], [1.])),  # BC
                stats.rv_discrete(values=([0.], [1.])),  # DST
                stats.rv_discrete(values=([0.], [1.])),  # NCL
            ],
        ),
        ambrs.AerosolModeDistribution(
            name="aitken",
            species=[so4, soa, ncl],
            number=stats.uniform(1e7, 1e11),
            geom_mean_diam=stats.rv_discrete(values=([2.6e-8], [1.])),
            log10_geom_std_dev=log10(1.6),
            mass_fractions=[
                stats.rv_discrete(values=([1.], [1.])),  # SO4
                stats.rv_discrete(values=([0.], [1.])),  # SOA
                stats.rv_discrete(values=([0.], [1.])),  # NCL
            ],
        ),
        ambrs.AerosolModeDistribution(
            name="coarse",
            species=[dst, ncl, so4, bc, pom, soa],
            number=stats.uniform(1e6, 1e7),
            geom_mean_diam=stats.rv_discrete(values=([2e-6], [1.])),
            log10_geom_std_dev=log10(1.8),
            mass_fractions=[
                stats.rv_discrete(values=([0.], [1.])),
                stats.rv_discrete(values=([0.], [1.])),
                stats.rv_discrete(values=([1.], [1.])),  # SO4
                stats.rv_discrete(values=([0.], [1.])),
                stats.rv_discrete(values=([0.], [1.])),
                stats.rv_discrete(values=([0.], [1.])),
            ],
        ),
        ambrs.AerosolModeDistribution(
            name="primary carbon",
            species=[pom, bc],
            number=stats.rv_discrete(values=([0.], [1.])),
            geom_mean_diam=stats.loguniform(1e-8, 5e-8),
            log10_geom_std_dev=log10(1.8),
            mass_fractions=[
                stats.rv_discrete(values=([1.], [1.])),  # POM
                stats.rv_discrete(values=([0.], [1.])),  # BC
            ],
        ),
    ]),
    gas_concs=tuple([stats.uniform(1e-10, 1e-8) for _ in range(2)]),
    flux=stats.uniform(1e-11, 1e-8),
    relative_humidity=stats.uniform(0, 0.99),
    temperature=stats.uniform(240, 310),
    pressure=p0,
    height=h0,
)

# -----------------------------------------------------------
# Create ensemble
# -----------------------------------------------------------
ensemble = ambrs.lhs(specification=spec, n=n)

# fixme: make zfill(fill_value) dynamic based on n_scenarios

scenario_names = [str(ii).zfill(1) for ii in range(1, len(ensemble.flux))]

# -----------------------------------------------------------
# Run MAM4
# -----------------------------------------------------------
mam4 = ambrs.mam4.AerosolModel(processes=processes)
mam4_inputs = mam4.create_inputs(ensemble=ensemble, dt=dt, nstep=nstep)
mam4_runner = ambrs.PoolRunner(
    model=mam4,
    executable="mam4",
    root=mam4_dir,
    num_processes=1,
)
# Allow skipping model execution for quick visualization-only runs
VIS_ONLY = os.environ.get('AMBRS_VIS_ONLY', os.environ.get('AMBRS_VIS_ONLY', '0')).lower() in ('1', 'true', 'yes')
if not VIS_ONLY:
    mam4_runner.run(mam4_inputs)
else:
    logging.info('AMBRS_VIS_ONLY set: skipping MAM4 model execution')

# -----------------------------------------------------------
# Run PartMC
# -----------------------------------------------------------
partmc = ambrs.partmc.AerosolModel(
    processes=processes,
    run_type="particle",
    n_part=n_part,
    n_repeat=1,
)
partmc_inputs = partmc.create_inputs(ensemble=ensemble, dt=dt, nstep=nstep)
partmc_runner = ambrs.PoolRunner(
    model=partmc,
    executable="partmc",
    root=partmc_dir,
    num_processes=1,
)
if not VIS_ONLY:
    partmc_runner.run(partmc_inputs)
else:
    logging.info('AMBRS_VIS_ONLY set: skipping PartMC model execution')


# If visualization-only, build PartMC and MAM4 outputs using the repository
# retrieval functions and draw a PyParticle grid comparing scenarios.
if VIS_ONLY:
    from pathlib import Path
    import warnings

    # PyParticle helpers
    from PyParticle.viz.grids import make_grid_scenarios_models
    # optional, used for a small demo compute later
    from PyParticle.analysis import compute_variable

    # repository retrieval helpers
    from ambrs.partmc import retrieve_model_state as retrieve_partmc
    from ambrs.mam4 import retrieve_model_state as retrieve_mam4
    timestep_to_plot = 1 # nstep
    # assemble output lists using the existing retrieval functions
    partmc_outputs = []
    mam4_outputs = []
    for scenario_name in scenario_names:
        partmc_output = retrieve_partmc(
            scenario_name=scenario_name,
            scenario=ensemble.member(int(scenario_name)),
            timestep=timestep_to_plot,  # timestep used when creating populations; keep consistent with run
            repeat_num=1,
            species_modifications={},
            ensemble_output_dir=partmc_dir,
        )
        partmc_outputs.append(partmc_output)

        mam4_output = retrieve_mam4(
            scenario_name=scenario_name,
            scenario=ensemble.member(int(scenario_name)),
            timestep=timestep_to_plot,
            repeat_num=1,
            species_modifications={},
            ensemble_output_dir=mam4_dir,
        )
        mam4_outputs.append(mam4_output)

    # Build simple config dicts that carry the retrieved outputs to the
    # model_cfg_builders used by make_grid_scenarios_models. We do not
    # fabricate data; if retrieve_* raised, execution will already have
    # terminated.
    scenario_cfgs = []
    for sid, (p_out, m_out) in zip(scenario_names, zip(partmc_outputs, mam4_outputs)):
        scenario_cfgs.append({
            "scenario_name": sid,
            "partmc_output": p_out,
            "mam4_output": m_out,
        })

    # variable list and small defaults for var_cfg mapping
    variables = ["dNdlnD", "frac_ccn"]

    def _var_cfg_for(v):
        if v == "dNdlnD":
            return {"wetsize": True, "N_bins": 40, "D_min": 1e-8, "D_max": 2e-6}
        if v in ("frac_ccn", "Nccn"):
            return {"s_eval": np.logspace(-2, 1, 40)}
        return {}

    var_cfg_mapping = {v: _var_cfg_for(v) for v in variables}

    # Builders that return the already-retrieved PyParticle population objects
    def partmc_builder(cfg):
        pop = cfg["partmc_output"].particle_population
        # best-effort provenance marker
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

    # Suppress a known benign surface-tension warning from hygroscopic growth
    warnings.filterwarnings(
        "ignore",
        message="Surface tension not implemented",
        category=UserWarning,
        module="PyParticle",
    )

    # Create output directory for figure
    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "out_grid_partmc_mam4.png"

    # Call PyParticle grid helper. Any exceptions should surface so the
    # user can see missing-files / API mismatches (no fallbacks).
    fig, axes = make_grid_scenarios_models(
        scenario_cfgs,
        variables,
        model_cfg_builders=[partmc_builder, mam4_builder],
        var_cfg=var_cfg_mapping,
        figsize=(4 * len(variables), 3 * len(scenario_cfgs)),
    )

    fig.suptitle("PartMC vs MAM4 scenario comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Wrote: {out_path}")

    # -------------------------
    # Range-bar ensemble summary
    # -------------------------
    import pandas as pd
    from ambrs.visualization.input_ranges import plot_range_bars

    metrics = []
    s_target = 0.01
    for idx, p_out in enumerate(partmc_outputs):
        try:
            sd = compute_variable(population=p_out.particle_population, varname="dNdlnD", var_cfg={"N_bins": 40})
            dNdlnD = sd.get("dNdlnD") if isinstance(sd, dict) else sd
            total_N = float(dNdlnD.sum())
        except Exception as e:
            total_N = float('nan')
            print(f"Warning computing dNdlnD for scenario {scenario_names[idx]}: {e}")

        try:
            nccn = compute_variable(population=p_out.particle_population, varname="Nccn", var_cfg={"s_eval": [s_target]})
            Nccn_val = float(nccn.get("Nccn")[0]) if (isinstance(nccn, dict) and "Nccn" in nccn) else float('nan')
        except Exception as e:
            Nccn_val = float('nan')
            print(f"Warning computing Nccn for scenario {scenario_names[idx]}: {e}")

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
        print(f"Dropped {dropped} metric entries with NaN values before plotting range bars")

    if df_metrics.empty:
        print("No valid metric data available for range-bars; skipping.")
    else:
        present_samples = sorted(df_metrics['sample'].unique().tolist())
        highlight_colors = {s: f"C{ii % 10}" for ii, s in enumerate(present_samples)}
        fig_rb = plot_range_bars(df_metrics, ["total_N", "Nccn_1pct"], var_col="variable", value_col="value",
                                 highlight_idx=present_samples, highlight_colors=highlight_colors, figsize=(8, 4))
        out_rb = out_dir / "out_range_bars.png"
        fig_rb.savefig(out_rb, dpi=180)
        print(f"Wrote range-bars: {out_rb}")

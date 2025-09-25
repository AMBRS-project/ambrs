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
from pathlib import Path

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
ensemble_name = "1h_1k"
root_dir = os.path.join("/Users/fier887/Downloads/ambrs_runs", ensemble_name)

partmc_dir = os.path.join(root_dir, "partmc_runs")
mam4_dir = os.path.join(root_dir, "mam4_runs")

for d in [root_dir, partmc_dir, mam4_dir]:
    os.makedirs(d, exist_ok=True)

n = 5        # number of ensemble members
n_part = 1000  # PartMC particles per run

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


# -----------------------------------------------------------
# Visualization
# -----------------------------------------------------------
from PyParticle.viz.style import StyleManager, Theme
from PyParticle.viz.builder import build_plotter
import matplotlib.pyplot as plt
import numpy as np

mgr = StyleManager(Theme(), deterministic=True)

scenario_name = '1'
timestep_to_plot = 1
partmc_output = ambrs.partmc.retrieve_model_state(
    scenario_name=scenario_name,
    scenario=ensemble.member(int(scenario_name) - 1),
    timestep=timestep_to_plot,
    repeat_num=1,
    species_modifications={},
    ensemble_output_dir=partmc_dir,
)
mam4_output = ambrs.mam4.retrieve_model_state(
    scenario_name=scenario_name,
    scenario=ensemble.member(int(scenario_name) - 1),
    timestep=timestep_to_plot,
    repeat_num=1,
    species_modifications={},
    ensemble_output_dir=mam4_dir,
)

# Lines
series = [
    {"key": "partmc", "population": partmc_output.particle_population, "label": "PartMC"},
    {"key": "mam4", "population": mam4_output.particle_population, "label": "MAM4"},
]

# Build line styles after series is known
line_styles = mgr.plan("line", [s["key"] for s in series])

fig, ax = plt.subplots()
for s in series:
    cfg = {
        "varname": "dNdlnD",
        "var_cfg": {"D": np.logspace(-9, -5, 50)},
        "style": line_styles[s["key"]],
    }

    # cfg = { 
    #     "varname": "frac_ccn",
    #     "var_cfg": {"s_eval": np.logspace(-2, 2, 50)},  # simple case: single x
    #     "style": line_styles[s["key"]],
    # }

    # cfg = { 
    #     "varname": "b_scat",
    #     "var_cfg": {"wvl_grid": [550e-9], "rh_grid": np.linspace(0.,100.,21)},  # simple case: single x
    #     "style": line_styles[s["key"]],
    # }


    # cfg = { 
    #     "varname": "b_scat",
    #     "var_cfg": {"wvl_grid": np.linspace(350e-9,1050e-9,29), "rh_grid": [0.]},  # simple case: single x
    #     "style": line_styles[s["key"]],
    # }


    # cfg = {
    #     "varname": "b_scat",
        # explicit var_cfg: wavelength grid and RH grid so returned variables are arrays
        # "var_cfg": {"wvl_grid": np.linspace(350e-9, 1050e-9, 29), "rh_grid": [0.]},
    #     "style": line_styles[s["key"]],
    # }
    plotter = build_plotter("state_line", cfg)
    plotter.plot(s["population"], ax, label=s["label"])
ax.set_xscale('log'); 
ax.legend(); fig.tight_layout()
out_dir = Path(__file__).resolve().parent / "reports"
out_dir.mkdir(parents=True, exist_ok=True)
fname = out_dir / ("s" + scenario_name + '_t' + str(timestep_to_plot) + '_' + str(cfg.get('varname')) + '.png')
fig.savefig(fname)


# If visualization-only, build PartMC and MAM4 outputs using the repository
# retrieval functions and draw a PyParticle grid comparing scenarios.
# if VIS_ONLY:
#     # Delegate visualization to ambrs.viz so this logic can be reused.
#     #from ambrs.viz_new_new import visualize_ensemble
#     from ambrs.viz import visualize_ensemble
#     # Use the same defaults as the inlined code: timestep 1 and repository
#     # reports directory.
#     out = visualize_ensemble(
#         ensemble=ensemble,
#         partmc_dir=partmc_dir,
#         mam4_dir=mam4_dir,
#         scenario_names=scenario_names,
#         timestep_to_plot=1,
#         out_dir=(Path(__file__).resolve().parent / "reports"),
#     )
#     print(f"Wrote visualization outputs: {out}")

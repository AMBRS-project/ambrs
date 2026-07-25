# MAM4-JAX in AMBRS

[MAM4-JAX](https://github.com/reflective-org/MAM4-JAX) is a JAX re-implementation
of the 4-mode MAM4 (Modal Aerosol Model) box model: the same fixed-structure modal
scheme as the executable MAM4 AMBRS already supports — accumulation, Aitken,
coarse and primary-carbon modes with fixed geometric standard deviations — but as
a differentiable Python library.

Like TOMAS-JAX (and unlike the executable MAM4), it has no input files and no
command line, so AMBRS steps it **in process** rather than through
`ambrs.runners.PoolRunner`.

## Installation

MAM4-JAX is declared in `requirements.txt`, so the usual install covers it:

```sh
pip install -r requirements.txt
```

To install it on its own (it pulls in `jax`, `jaxlib` and `diffrax`):

```sh
pip install "mam4-jax @ git+https://github.com/reflective-org/MAM4-JAX.git@main"
```

The dependency is optional at import time: `import ambrs` works without it, and
`ambrs.mam4_jax._MAM4_JAX_AVAILABLE` reports whether it was importable.
Constructing `ambrs.mam4_jax.AerosolModel` without it raises `ImportError`. The
tests in `test/test_mam4_jax.py` skip themselves when it is absent.

## Usage

Inputs are created exactly as for the executable MAM4 — a 4-mode scenario, in
mode order accumulation, Aitken, coarse, primary carbon — and run in process:

```python
import ambrs

model = ambrs.mam4_jax.AerosolModel(
    ambrs.AerosolProcesses(coagulation=True, condensation=True, nucleation=True))

inputs = model.create_inputs(ensemble, dt=30.0, nstep=120)   # one hour
outputs = model.run_ensemble(inputs)                          # list[analysis.Output]
```

A runnable walkthrough with plots is in
[`demo_mam4_jax.ipynb`](demo_mam4_jax.ipynb).

`run_ensemble` reuses a single JIT-compiled step (`calcsize → wateruptake →
amicphys`) across the whole ensemble, so compilation is paid once. A single input
runs with `model.run(input, scenario_name=...)`.

Each `Output` is the same `ambrs.analysis.Output` every other model produces — a
`part2pop` binned-lognormal population built from the final per-mode number,
diameter and composition, plus a `GasMixture` (H2SO4, SO2, SOAG) and the
thermodynamic state — so `compute_variable`, `nmae` and `kl_divergence` work
against it unchanged.

### Provenance and `retrieve_model_state`

`write_input_files(input, dir, prefix)` records the state a run started from
(`<prefix>.npz` plus a readable `<prefix>.txt`) for provenance and
reproducibility. `retrieve_model_state(scenario_name, scenario, timestep, ...)`
mirrors the PartMC/MAM4 functions: it loads the recorded input from
`<ensemble_output_dir>/<scenario_name>/` and steps it forward to `timestep`
(`timestep=1` is the initial state), honouring the recorded process flags unless
an `AerosolProcesses` is passed. `invocation()` raises — there is no executable.

## Modelling assumptions

**Processes.** `AerosolProcesses` maps onto MAM4's `mdo_*` toggles the same way
`ambrs.mam4` writes its namelist: `condensation → mdo_gasaerexch`,
`nucleation → mdo_newnuc`, `coagulation → mdo_coag`, and `mdo_rename` always on.
`gas_phase_chemistry` is not a toggle; it enables MAM4-JAX's built-in H2SO4
production term (1e-16 mol/mol/s, injected inside gas-aerosol exchange, so it
has no effect unless condensation is on — a warning says so).

**Fixed mode structure.** MAM4's per-mode geometric standard deviations
(1.8, 1.6, 1.8, 1.6) are part of the model, not free parameters. A scenario mode
whose GSD differs warns, and the fixed value is used. The mode volume is built
with the fixed width so that `calcsize` reproduces the input number and mean
diameter as a fixed point on step 0.

**Humidity enters through the water-vapour tracer.** MAM4-JAX's `wateruptake`
derives relative humidity from the water-vapour tracer `q[0]`, not from the
state's `relhum` field (only nucleation reads that). The adapter sets
`q[0] = RH · qsat(T, p)`, which reproduces the reference driver's initial
condition; without it the aerosol would stay dry regardless of the scenario RH.

**Composition.** AMBRS species map onto MAM4's nine species types:

| AMBRS species | MAM4 type |
| --- | --- |
| `SO4` | sulfate |
| `OC` | primary organic |
| `MSA`, `ARO1/2`, `ALK1`, `OLE1`, `API1/2`, `LIM1/2` | secondary organic |
| `BC` | black carbon |
| `Na`, `Cl` | sea salt |
| `OIN`, `Ca`, `CO3` | dust |
| `NH4`, `NO3` | **no slot in this MAM4-MOM build** |

Not every mode carries every type (the Aitken mode has no black-carbon or dust
slot, primary carbon has no sulfate slot, and no mode has ammonium or nitrate).
Species with no native slot in a mode are lumped — into sulfate where it exists,
else primary organic — with a warning naming them; total mass is conserved but
the speciation is approximate. Aerosol water in a scenario is dropped, since the
model computes its own water uptake.

**Units.** Mode `number` is `[# m^-3]`, converted to MAM4's `[# per kg dry air]`
via the dry-air density. Gas concentrations in `Scenario.gas_concs` are MAM4
mass mixing ratios `[kg gas / kg dry air]`, matching `ambrs.mam4` (note this
differs from `ambrs.tomas_jax`, which takes `[molec cm^-3]`). SO2 is carried but
is inert in this build and is echoed to the output unchanged.

**Clear-sky only.** MAM4-JAX's cloudy microphysics path is unimplemented, so the
cloud fraction is fixed at zero. `zmid`/`pblh` (which PBL nucleation reads) are
constructor settings, since a `Scenario` does not carry them.

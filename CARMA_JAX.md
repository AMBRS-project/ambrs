# CARMA-JAX in AMBRS

[CARMA-JAX](https://github.com/reflective-org/carma-jax) is a JAX port of the
CARMA sectional (bin) aerosol microphysics model. It resolves the size
distribution on a geometric mass grid — by default 47 mass-doubling bins
spanning particle radii of about 0.2 nm to 8 µm — and, like the other JAX
models, it is a Python library with no input files and no command line, so
AMBRS steps it **in process** rather than through `ambrs.runners.PoolRunner`.

## What is wired (and what deliberately is not)

This adapter wires CARMA-JAX's **coagulation** — its production-quality path: a
single internally-mixed involatile particle group, stepped by a fully
JIT-compiled kernel with the physical Brownian coagulation kernel built from the
scenario's temperature and pressure exactly the way CARMA's own drivers build it
(`setup_atm → setup_vf → setup_ckern`).

CARMA-JAX's condensational growth and sulfate nucleation exist on its `dev`
branch, but the environment builder they need is not yet part of the installed
package (it lives in the repository's `scripts/`). Rather than vendor that code
or silently do nothing, the adapter **raises `ValueError`** when `condensation`,
`nucleation` or any other unwired process is requested. Wiring them becomes
straightforward once CARMA-JAX exports its environment builder.

The dependency is pinned to a `dev` commit: `dev` is the active branch, and the
curated `main` release is coagulation-only with older, pre-Fortran-parity bin
edges.

## Installation

CARMA-JAX is declared in `requirements.txt`, so the usual install covers it:

```sh
pip install -r requirements.txt
```

The dependency is optional at import time: `import ambrs` works without it, and
`ambrs.carma_jax._CARMA_JAX_AVAILABLE` reports whether it was importable.
Constructing `ambrs.carma_jax.AerosolModel` without it raises `ImportError`.
The tests in `test/test_carma_jax.py` skip themselves when it is absent.

## Usage

```python
import ambrs

model = ambrs.carma_jax.AerosolModel(
    ambrs.AerosolProcesses(coagulation=True),
    nbin=47,            # bins (default)
    rmin=2e-10,         # smallest bin radius [m] (default 0.2 nm)
    rmrat=2.0,          # mass ratio between bins (default: mass doubling)
    density=1770.0)     # particle density [kg m^-3]

inputs = model.create_inputs(ensemble, dt=60.0, nstep=1440)   # one day
outputs = model.run_ensemble(inputs)                           # list[analysis.Output]
```

A runnable walkthrough with plots is in
[`demo_carma_jax.ipynb`](demo_carma_jax.ipynb).

The bin grid is fixed at model construction, because CARMA's coagulation pair
tables are built from it (a ~1 s, once-per-model cost); the compiled step is
then reused across every scenario in an ensemble.

Each `Output` is the standard `ambrs.analysis.Output`: a `part2pop` particle
population with one particle per populated bin, plus the thermodynamic state,
so `compute_variable`, `nmae` and `kl_divergence` work unchanged. This CARMA
configuration carries no gas phase, so the `GasMixture` is empty.

### Provenance and `retrieve_model_state`

`write_input_files(input, dir, prefix)` records the state a run started from
(`<prefix>.npz` plus a readable `<prefix>.txt`).
`retrieve_model_state(scenario_name, scenario, timestep, ...)` mirrors the
PartMC/MAM4 functions: it loads the recorded input from
`<ensemble_output_dir>/<scenario_name>/` and steps it forward to `timestep`
(`timestep=1` is the initial state). Reading a recording requires a model built
with the same bin grid; a mismatch raises. `invocation()` raises — there is no
executable.

## Modelling assumptions

**Binning is exact.** Each log-normal mode is placed on the grid by the
analytic integral of the distribution between every bin's boundary radii, so
the total number is preserved by construction (a warning fires if more than 1%
of a mode falls outside the grid — enlarge `nbin` or move `rmin` if so).

**One internally-mixed element.** This CARMA configuration tracks a single
involatile element per bin, so a multi-mode scenario's compositions are blended
(volume-weighted) into one mixture, which is reported uniformly across the
output population. A warning fires when the modes' compositions actually
differ. Aerosol water is dropped. The particle `density` is a model setting,
not derived per-scenario.

**One moment.** CARMA's number element carries number only; within a bin the
particle mass is the bin's. Coagulation moves number between bins and conserves
total dry mass to machine precision — the adapter's tests pin this.

**Units.** Mode `number` is `[# m^-3]` (converted to CARMA's `[# cm^-3]`);
temperature `[K]` and pressure `[Pa]` are converted to CARMA's CGS internally.
The scenario's relative humidity is recorded in the output but does not affect
coagulation of an involatile group.

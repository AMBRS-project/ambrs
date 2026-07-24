# TOMAS-JAX in AMBRS

[TOMAS-JAX](https://github.com/reflective-org/tomas-jax) is a JAX re-implementation
of the TwO-Moment Aerosol Sectional (TOMAS) model: a sectional scheme that tracks
both number and per-species mass in each of 40 mass-doubling size bins, spanning
roughly 1.7 nm to 17.5 µm.

It differs from PartMC and MAM4 in one way that matters for the framework: it is a
Python/JAX library, not a compiled executable. There are no native input files to
write and no command to invoke, so it is stepped **in process** rather than through
`ambrs.runners.PoolRunner`.

## Installation

TOMAS-JAX is declared in `requirements.txt`, so the usual install covers it:

```sh
pip install -r requirements.txt
```

To install it on its own (it pulls in `jax`, `jaxlib` and `diffrax`):

```sh
pip install "tomas-jax @ git+https://github.com/reflective-org/tomas-jax.git@dev"
```

The dependency is optional at import time: `import ambrs` works without it, and
`ambrs.tomas_jax._TOMAS_AVAILABLE` reports whether it was importable. Constructing
`ambrs.tomas_jax.AerosolModel` without it raises `ImportError`. The tests in
`test/test_tomas_jax.py` skip themselves when it is absent.

## Usage

Inputs are created exactly as for the other models; only the running differs.

```python
import ambrs
import ambrs.aerosol as aerosol
import ambrs.gas as gas

so4 = aerosol.AerosolSpecies(name='SO4', molar_mass=97.071, density=1770,
                             hygroscopicity=0.507)
h2so4 = gas.GasSpecies(name='H2SO4', molar_mass=98.079)

mode = aerosol.AerosolModeState(
    name='aitken', species=(so4,),
    number=1e10,             # [# m^-3]
    geom_mean_diam=8e-8,     # [m]
    log10_geom_std_dev=0.204,
    mass_fractions=(1.0,))

scenario = ambrs.Scenario(
    aerosols=(so4,), gases=(h2so4,),
    size=aerosol.AerosolModalSizeState(modes=(mode,)),
    gas_concs=(1e7,),        # H2SO4 [molec cm^-3]
    flux=0.0, relative_humidity=0.5,
    temperature=298.0, pressure=101325.0, height=500.0)

model = ambrs.tomas_jax.AerosolModel(
    ambrs.AerosolProcesses(coagulation=True, condensation=True),
    h2so4_production=1e5)    # [molec cm^-3 s^-1]

inputs = model.create_inputs([scenario], dt=60.0, nstep=60)   # one hour
outputs = model.run_ensemble(inputs)

for out in outputs:
    pop = out.particle_population
    print(out.scenario_name, pop.get_Ntot(), pop.get_tot_dry_mass())
    print(out.compute_variable('dNdlnD'))
```

`run_ensemble` reuses a single JIT-compiled step across the whole ensemble, so the
compilation cost is paid once rather than per scenario. A single input can also be
run directly with `model.run(input, scenario_name=...)`.

Each `Output` is the same `ambrs.analysis.Output` the other models produce — a
`part2pop` particle population plus a gas mixture and thermodynamic state — so
`compute_variable`, `nmae` and `kl_divergence` all work against it unchanged.

### Provenance and `retrieve_model_state`

`write_input_files(input, dir, prefix)` records the state a run started from
(`<prefix>.npz` plus a readable `<prefix>.txt`); there is no native input format,
so this exists for provenance and reproducibility rather than for the model to read.

`retrieve_model_state(scenario_name, scenario, timestep, ...)` mirrors the PartMC and
MAM4 functions of the same name, so callers can treat every model alike. Because an
in-process model leaves no output on disk, it loads the recorded input from
`<ensemble_output_dir>/<scenario_name>/` and steps it forward to `timestep`
(`timestep=1` is the initial state). When the `Input` is still in hand,
`model.run(...)` is the cheaper path.

`invocation()` raises, explaining that there is no executable to run.

## Modelling assumptions

**Processes.** `AerosolProcesses.coagulation`, `.condensation` and `.nucleation` map
onto the corresponding TOMAS processes, applied in TOMAS's operator-split order.
Condensation acts on H2SO4 only. Nucleation uses the `ricco_dunne` scheme by default;
`nucl_scheme='zhao2024'` selects the other.

**Nucleation precursors.** A `Scenario` doesn't carry `org_conc`, `nh3_conc` or
`fion`, so they are constructor settings (zero by default). A `Scenario` NH3
concentration, if present, overrides `nh3_conc`. `h2so4_production` adds H2SO4 gas
each step, since TOMAS's step doesn't apply a source term itself.

**Composition.** TOMAS carries sulfate, a run of organics, ammonium and water.
AMBRS species map on as follows:

| AMBRS species | TOMAS slot |
| --- | --- |
| `SO4` | sulfate |
| `NH4` | ammonium |
| `H2O` | water |
| `OC`, `MSA`, `ARO1/2`, `ALK1`, `OLE1`, `API1/2`, `LIM1/2` | the organic slot |
| `BC`, `OIN`, `NO3`, `Cl`, `Na`, `Ca`, `CO3` | **no TOMAS slot** |

Species in the last row have their mass lumped into the organic slot so that total
mass is conserved, and a warning names them. The speciation is approximate; TOMAS
condenses onto sulfate regardless.

**Units.** A `Scenario`'s modal `number` is `[# m^-3]` and TOMAS bins number in
`[# cm^-3]`; the adapter converts in both directions, so populations come back in
`[# m^-3]`. Gas concentrations in `Scenario.gas_concs` are taken to be
`[molec cm^-3]`.

**Two moments.** Because TOMAS tracks number *and* mass per bin, a bin's mean
particle mass is `Mk/Nk` rather than the geometric bin mass. The output population
takes its diameters from `Mk/Nk`, which is what makes the population's mass match
the model's — it is conserved to machine precision under coagulation. Using the
geometric bin mass instead drifts by several percent.

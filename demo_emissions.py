"""Simple demo: create and run a small PartMC scenario with aerosol emissions and background entrainment."""

from pathlib import Path
import shutil
import subprocess
import textwrap

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.partmc as partmc
from ambrs.scenario import Scenario, AerosolEmissions


def build_scenario() -> Scenario:
    so4 = aerosol.AerosolSpecies(
        name='SO4',
        molar_mass=97.071,
        density=1770,
        hygroscopicity=0.507,
    )
    bc = aerosol.AerosolSpecies(
        name='BC',
        molar_mass=12.01,
        density=1800,
        hygroscopicity=0.0,
    )
    nh4 = aerosol.AerosolSpecies(
        name='NH4',
        molar_mass=18.038,
        density=1750,
        hygroscopicity=0.5,
    )
    h2o = aerosol.AerosolSpecies(
        name='H2O',
        molar_mass=18.015,
        density=1000,
        hygroscopicity=0.6,
    )
    mode_sulfate = aerosol.AerosolModeState(
        name = "mode1",
        species = [so4, bc, nh4, h2o],
        number = 5e8,
        geom_mean_diam = 1e-7,
        log10_geom_std_dev = 0.2,
        mass_fractions = (0.4, 0.25, 0.15, 0.2),
    )
    modal_state = aerosol.AerosolModalSizeState(modes=(mode_sulfate,))

    emission_event = AerosolEmissions(
        time = 0.0,
        rate = 1.0,
        size = modal_state,
    )
    background_event = AerosolEmissions(
        time = 0.0,
        rate = 0.5,
        size = modal_state,
    )

    gases = (
        gas.GasSpecies(name='SO2', molar_mass=64.07),
        gas.GasSpecies(name='O3', molar_mass=48.00),
    )

    return Scenario(
        aerosols = (so4, bc, nh4, h2o),
        gases = gases,
        size = modal_state,
        gas_concs = (1e4, 1750.0),
        flux = 0.0,
        relative_humidity = 0.6,
        temperature = 298.0,
        pressure = 101325.0,
        height = 500.0,
        aerosol_emissions = [emission_event],
        aerosol_background = [background_event],
    )


def main():
    run_dir = Path('runs/partmc_simple_emissions')
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    scenario = build_scenario()
    processes = aerosol.AerosolProcesses(
        coagulation = True,
        condensation = True,
        do_camp_chem = False,
    )
    model = partmc.AerosolModel(
        processes = processes,
        run_type = 'particle',
        n_part = 200,
        n_repeat = 1,
    )
    input = model.create_input(scenario, dt = 0.5, nstep = 2)
    model.write_input_files(input, run_dir, 'demo_emission')

    for fname in ['aero_emit.dat', 'aero_back.dat']:
        print(f"\n>>> {fname} contents:")
        print(textwrap.indent(Path(run_dir, fname).read_text().strip(), "    "))
    
    emit_dist = run_dir / 'aero_emit_dist_1_dist.dat'
    print(f"\n>>> {emit_dist} contents:")
    print(textwrap.indent(emit_dist.read_text().strip(), "    "))

    print("\n>>> Launching PartMC:")
    partmc_exe = shutil.which('partmc')
    if not partmc_exe:
        raise RuntimeError("partmc executable not found in PATH")
    result = subprocess.run(
        [partmc_exe, run_dir / 'demo_emission.spec'],
        cwd = run_dir,
        capture_output = True,
        text = True,
        check = True,
    )
    print("\n>>> PartMC stdout/stderr:")
    print(textwrap.indent(result.stdout.strip(), "    "))

    print("\n>>> Out directory contents:")
    for path in sorted(run_dir.glob('out/*')):
        print(f"    {path.name}")


if __name__ == '__main__':
    main()

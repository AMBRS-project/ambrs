# This script attempts to reproduce the ensembles for MAM4 and PartMC from
# Fierce et al, _Quantifying_structural_errors_in_cloud_condensation_nuclei_
# activity_from_reduced_representation_of_aerosol_size_distributions_,
# J. Aerosol Science 181 (2024) 106388 (https://doi.org/10.1016/j.jaerosci.2024.106388)
#
# The reproduction is approximate because this script does not sample size
# distribution parameters from E3SM.

import ambrs
import logging
from math import log10
import os
import scipy.stats as stats

# log to the terminal
logging.basicConfig(level = logging.INFO)

# aerosol processes under consideration
processes = ambrs.AerosolProcesses(
    coagulation = True,
    condensation = True,
)

# simulation parameters
dt    = 60   # time step size [s]
nstep = 1440 # number of steps [-]

# relevant aerosol and gas species
# FIXME: proper species and properties need to be filled in here!
so4 = ambrs.AerosolSpecies(
    name='SO4',
    molar_mass = 97.071, # NOTE: 1000x smaller than "molecular weight"!
    density = 1770,
    hygroscopicity = 0.507,
)
pom = ambrs.AerosolSpecies(
    name='OC',
    molar_mass = 12.01,
    density = 1000, # wrong
    hygroscopicity = 0.5,
)
soa = ambrs.AerosolSpecies(
    name='OC',
    molar_mass = 12.01,
    density = 1000, # wrong
    hygroscopicity = 0.5,
)
bc = ambrs.AerosolSpecies(
    name='BC',
    molar_mass = 12.01,
    density = 1000, # wrong
    hygroscopicity = 0.5,
)
dst = ambrs.AerosolSpecies(
    name='OIN',
    molar_mass = 135.065,
    density = 1000, # wrong
    hygroscopicity = 0.5,
)
na = ambrs.AerosolSpecies(
    name='Na',
    molar_mass = 22.99,
    density = 1000, # wrong
    ions_in_soln = 1,
)
cl = ambrs.AerosolSpecies(
    name='Cl',
    molar_mass = 35.45,
    density = 1000, # wrong
    ions_in_soln = 1,
)
ncl = na # FIXME: we use this as a proxy for now, assuming 1:1 stoich with Cl

so2 = ambrs.GasSpecies(
    name='SO2',
    molar_mass = 64.07,
)
h2so4 = ambrs.GasSpecies(
    name='H2SO4',
    molar_mass = 98.079,
)
# FIXME: I can't figure out how SOAG maps to PartMC/MOSAIC
#soag = ambrs.GasSpecies(
#    name='soag',
#    molar_mass = 12.01,
#)

# reference pressure and height
p0 = 101325 # [Pa]
h0 = 500    # [m]

# specify distributions sampled for the ensemble
spec = ambrs.EnsembleSpecification(
    name = 'demo',
    aerosols = (so4, pom, soa, bc, dst, ncl),
    gases = (so2, h2so4), # FIXME: re-add soag?
    size = ambrs.AerosolModalSizeDistribution(
        modes = [
            ambrs.AerosolModeDistribution(
                name = "accumulation",
                species = [so4, pom, soa, bc, dst, ncl],
                number = stats.loguniform(3e7, 2e12),
                geom_mean_diam = stats.loguniform(0.5e-7, 1.1e-7),
                log10_geom_std_dev = log10(1.6),
                mass_fractions = [
                    stats.uniform(0, 1), # so4
                    stats.uniform(0, 1), # pom
                    stats.uniform(0, 1), # soa
                    stats.uniform(0, 1), # bc
                    stats.uniform(0, 1), # dst
                    stats.uniform(0, 1), # ncl
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "aitken",
                species = [so4, soa, ncl],
                number = stats.loguniform(3e7, 2e12),
                geom_mean_diam = stats.loguniform(0.5e-8, 3e-8),
                log10_geom_std_dev = log10(1.6),
                mass_fractions = [
                    stats.uniform(0, 1), # so4
                    stats.uniform(0, 1), # soa
                    stats.uniform(0, 1), # ncl
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "coarse",
                species = [dst, ncl, so4, bc, pom, soa],
                number = stats.loguniform(3e7, 2e12),
                geom_mean_diam = stats.loguniform(1e-6, 2e-6),
                log10_geom_std_dev = log10(1.8),
                mass_fractions = [
                    stats.uniform(0, 1), # dst
                    stats.uniform(0, 1), # ncl
                    stats.uniform(0, 1), # so4
                    stats.uniform(0, 1), # bc
                    stats.uniform(0, 1), # pom
                    stats.uniform(0, 1), # soa
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "primary carbon",
                species = [pom, bc],
                number = stats.loguniform(3e7, 2e12),
                geom_mean_diam = stats.loguniform(1e-8, 6e-8),
                log10_geom_std_dev = log10(1.8),
                mass_fractions = [
                    stats.uniform(0, 1), # pom
                    stats.uniform(0, 1), # bc
                ],
            ),
        ]),
    gas_concs = tuple([stats.uniform(1e5, 1e6) for g in range(2)]), # FIXME: change to 3 when SOAG is re-added
    flux = stats.loguniform(1e-2*1e-9, 1e1*1e-9),
    relative_humidity = stats.uniform(0, 0.99),
    temperature = stats.uniform(240, 310),
    pressure = p0,
    height = h0,
)

cwd = os.getcwd()

# create an ensemble using latin hypercube sampling
n = 100
ensemble = ambrs.lhs(specification = spec, n = n)

# run a MAM4 ensemble
mam4 = ambrs.mam4.AerosolModel(
    processes = processes,
)
mam4_inputs = mam4.create_inputs(
    ensemble = ensemble,
    dt = dt,
    nstep = nstep
)
mam4_dir = os.path.join(cwd, 'mam4_runs')
if not os.path.exists(mam4_dir):
    os.mkdir(mam4_dir)
mam4_runner = ambrs.PoolRunner(
    model = mam4,
    executable = 'mam4',
    root = mam4_dir,
)
mam_outputs = mam4_runner.run(mam4_inputs)

# run a PartMC ensemble
partmc = ambrs.partmc.AerosolModel(
    processes = processes,
    run_type = 'particle',
    n_part = 1000,
    n_repeat = 1,
)
partmc_inputs = partmc.create_inputs(
    ensemble = ensemble,
    dt = dt,
    nstep = nstep
)
partmc_dir = os.path.join(cwd, 'partmc_runs')
if not os.path.exists(partmc_dir):
    os.mkdir(partmc_dir)
partmc_runner = ambrs.PoolRunner(
    model = partmc,
    executable = 'partmc',
    root = partmc_dir,
)
partmc_outputs = partmc_runner.run(partmc_inputs)


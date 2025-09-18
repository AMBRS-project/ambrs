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


import numpy as np

# log to the terminal
logging.basicConfig(level = logging.INFO)

# aerosol processes under consideration
processes = ambrs.AerosolProcesses(
    coagulation = True,
    condensation = True,
    # gas_phase_chemistry=True,
    # aqueous_chemistry=True
)

# ensemble_name = 'ensemble_01'
# ensemble_dir = '/Users/fier887/Downloads/' + ensemble_name + '/'
# if not os.path.exists(ensemble_dir):
#     os.mkdir(ensemble_dir)

# partmc_dir = ensemble_dir + 'partmc_runs'
# mam4_dir = ensemble_dir + 'mam4_runs'

# partmc_dir = 'partmc_runs'
# mam4_dir = 'mam4_runs'

ensemble_name = '1h_5k'

if not os.path.exists('/Users/fier887/Downloads/ambrs_runs/' + ensemble_name):
    os.mkdir('/Users/fier887/Downloads/ambrs_runs/' + ensemble_name)

    
partmc_dir = '/Users/fier887/Downloads/ambrs_runs/' + ensemble_name + '/partmc_runs'
mam4_dir = '/Users/fier887/Downloads/ambrs_runs/' + ensemble_name + '/mam4_runs'

if not os.path.exists(partmc_dir):
    os.mkdir(partmc_dir)

if not os.path.exists(mam4_dir):
    os.mkdir(mam4_dir)

n = 5
n_part = 5000

# simulation parameters
dt    = 60   # time step size [s]
nstep = 60 # number of steps [-]
# 60-minute runs

# relevant aerosol and gas species
# FIXME: proper species and properties need to be filled in here!
# FIXME: not sure how MAM4 uses this info. I don't think PartMC actually 
so4 = ambrs.AerosolSpecies(
    name='SO4',
    molar_mass = 96., # NOTE: 1000x smaller than "molecular weight"!
    density = 1800.,
    hygroscopicity = 0.65,
)
pom = ambrs.AerosolSpecies(
    name='OC',
    molar_mass = 12.01,
    density = 1000., # from PartMC
    hygroscopicity = 0.001,
)
soa = ambrs.AerosolSpecies(
    name='MSA', # FIXME: applying an unused species for "SOA" placeholder; will set it to zero
    molar_mass = 40.,
    density = 2600., # from PartMC
    hygroscopicity = 0.53
)
bc = ambrs.AerosolSpecies(
    name='BC',
    molar_mass = 12.01,
    density = 1800., # from PartMC
    hygroscopicity = 0.,
)
dst = ambrs.AerosolSpecies(
    name='OIN',
    molar_mass = 135.065,
    density = 2600., # from PartMC
    hygroscopicity = 0.1,
)
na = ambrs.AerosolSpecies(
    name='Na',
    molar_mass = 23,
    density = 2200, # from PartMC
    hygroscopicity = 0.53,
)
cl = ambrs.AerosolSpecies(
    name='Cl',
    molar_mass = 35.5,
    density = 2200, # wrong
    hygroscopicity = 0.53
)
ncl = na # FIXME: we use this as a proxy for now, assuming 1:1 stoich with Cl
h2o = ambrs.AerosolSpecies(
    name='H2O',
    molar_mass = 18.,
    density = 1000, # wrong
    ions_in_soln = 1,
)

# so4 = ambrs.AerosolSpecies(
#     name='SO4',
#     molar_mass = 97.071, # NOTE: 1000x smaller than "molecular weight"!
#     density = 1770,
#     hygroscopicity = 0.507,
# )
# pom = ambrs.AerosolSpecies(
#     name='OC',
#     molar_mass = 12.01,
#     density = 1000, # wrong
#     hygroscopicity = 0.5,
# )
# soa = ambrs.AerosolSpecies(
#     name='MSA', # FIXME: applying an unused species for "SOA" placeholder; will set it to zero
#     molar_mass = 12.01,
#     density = 1000, # wrong
#     hygroscopicity = 0.5,
# )
# bc = ambrs.AerosolSpecies(
#     name='BC',
#     molar_mass = 12.01,
#     density = 1000, # wrong
#     hygroscopicity = 0.5,
# )
# dst = ambrs.AerosolSpecies(
#     name='OIN',
#     molar_mass = 135.065,
#     density = 1000, # wrong
#     hygroscopicity = 0.5,
# )
# na = ambrs.AerosolSpecies(
#     name='Na',
#     molar_mass = 22.99,
#     density = 1000, # wrong
#     ions_in_soln = 1,
# )
# cl = ambrs.AerosolSpecies(
#     name='Cl',
#     molar_mass = 35.45,
#     density = 1000, # wrong
#     ions_in_soln = 1,
# )
# h2o = ambrs.AerosolSpecies(
#     name='H2O',
#     molar_mass = 18.,
#     density = 1000, # wrong
#     ions_in_soln = 1,
# )
# ncl = na # FIXME: we use this as a proxy for now, assuming 1:1 stoich with Cl

# fixme: need to zero out SO2 for consistency with MAM4 (no actual chemistry? just H2SO4)
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
    name = ensemble_name,
    aerosols = (so4, pom, soa, bc, dst, ncl, h2o),
    gases = (so2, h2so4), # FIXME: re-add soag?
    size = ambrs.AerosolModalSizeDistribution(
        modes = [
            ambrs.AerosolModeDistribution(
                name = "accumulation",
               species = [so4, pom, soa, bc, dst, ncl],
                # species = [so4, pom, bc, dst],
                number = stats.uniform(1e7,1e10),
                # number = stats.loguniform(1e7, 3),
                # geom_mean_diam = stats.loguniform(0.5e-7, 1.1e-7),
                # geom_mean_diam = stats.uniform(0.5e-7, 0.6e-7),
                geom_mean_diam = stats.rv_discrete(values=([1.1e-7], [1.])), #soa
                log10_geom_std_dev = log10(1.6),
                mass_fractions = [
                    stats.rv_discrete(values=([1.], [1.])), # so4
                    stats.rv_discrete(values=([0.], [1.])), # pom
                    # stats.uniform(0, 1), # so4
                    # stats.uniform(0, 1), # pom
                    stats.rv_discrete(values=([0.], [1.])), #soa
                    stats.rv_discrete(values=([0.], [1.])), # bc
                    stats.rv_discrete(values=([0.], [1.])), # dst
                    stats.rv_discrete(values=([0.], [1.])), # ncl
                    # stats.uniform(0, 0), # soa
                    # stats.uniform(0, 0), # bc
                    # stats.uniform(0, 0), # dst
                    # stats.uniform(0, 0), # ncl
                    # stats.uniform(0, 0), # h2o
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "aitken",
                species = [so4, soa, ncl],#, h2o],
                # species = [so4],
                # number = stats.loguniform(1e7, 4),
                number = stats.uniform(1e7,1e11),
                # geom_mean_diam = stats.loguniform(0.5e-8, 3e-8),
                # number = stats.loguniform(1e7, 4),
                # geom_mean_diam = stats.uniform(0.5e-8, 1.5e-8),
                geom_mean_diam = stats.rv_discrete(values=([2.6e-8], [1.])), #soa
                log10_geom_std_dev = log10(1.6),
                mass_fractions = [
                    stats.rv_discrete(values=([1.], [1.])), # pom
                    stats.rv_discrete(values=([0.], [1.])),
                    stats.rv_discrete(values=([0.], [1.])),
                    # stats.uniform(0, 0), # soa # FIXME: zeroing out problematic species
                    # stats.uniform(0, 0), # ncl # FIXME: zeroing out problematic species
                    # stats.uniform(0, 0), # ncl # FIXME: zeroing out problematic species
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "coarse",
                # species = [dst, so4, bc, pom],
                species = [dst, ncl, so4, bc, pom, soa],#, h2o],
                # number = stats.loguniform(1e6, 1),
                number = stats.uniform(1e6,1e7),
                # geom_mean_diam = stats.loguniform(1e-6, 2e-6),
                # number = stats.loguniform(1e6, 1),
                # geom_mean_diam = stats.uniform(1e-6, 1e-6),
                geom_mean_diam = stats.rv_discrete(values=([2e-6], [1.])), #soa
                log10_geom_std_dev = log10(1.8),
                mass_fractions = [
                    stats.rv_discrete(values=([0.], [1.])),
                    stats.rv_discrete(values=([0.], [1.])),
                    # stats.uniform(0, 0), # dst
                    # stats.uniform(0, 0), # ncl  # FIXME: zeroing out problematic species
                    # stats.uniform(0, 1), # so4
                    stats.rv_discrete(values=([1.], [1.])),
                    stats.rv_discrete(values=([0.], [1.])),
                    # stats.uniform(0, 0), # bc
                    # stats.uniform(0, 1), # pom
                    stats.rv_discrete(values=([0.], [1.])),
                    stats.rv_discrete(values=([0.], [1.])),
                    # stats.uniform(0, 0), # soa  # FIXME: zeroing out problematic species
                    # stats.uniform(0, 0), # h2o # FIXME: zeroing out problematic species
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "primary carbon",
                species = [pom, bc],#, h2o],
                number = stats.rv_discrete(values=([0.],[1.])),#stats.loguniform(3e7, 2e12),
                # geom_mean_diam = stats.loguniform(1e-8, 6e-8),
                geom_mean_diam = stats.loguniform(1e-8, 5e-8),
                log10_geom_std_dev = log10(1.8),
                mass_fractions = [
                    # dirac_delta_gen(1.), # pom
                    # stats.uniform(0, 0), # bc
                    # stats.uniform(0.9, 1), # pom
                    stats.rv_discrete(values=([1.], [1.])), # pom
                    stats.rv_discrete(values=([0.], [1.])),
                    # stats.uniform(0, 0), # bc
                    # stats.uniform(0, 0), # h2o # FIXME: zeroing out problematic species
                ],
            ),
        ]),
    # FIXME: we need to document units a bit better; specifying in mol/mol-air
    gas_concs = tuple([stats.uniform(1e-10,1e-8) for g in range(2)]), # FIXME: change to 3 when SOAG is re-added
    # flux = stats.loguniform(1e-2*1e-9, np.log10(1e1*1e-9)),
    flux = stats.uniform(1e-2*1e-9, 1e1*1e-9),
    relative_humidity = stats.uniform(0, 0.99),
    # temperature = stats.uniform(240, 310),
    temperature = stats.uniform(240, 70),
    pressure = p0,
    height = h0,
)

cwd = os.getcwd()

# create an ensemble using latin hypercube sampling

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
if not os.path.exists(mam4_dir):
    os.mkdir(mam4_dir)
mam4_runner = ambrs.PoolRunner(
    model = mam4,
    executable = 'mam4',
    root = mam4_dir,
    num_processes = 1
)
mam4_runner.run(mam4_inputs)

# fixme: output processing is now a separate step
# mam4_outputs = mam4_runner.run(mam4_inputs)
# mam_outputs = mam4_runner.run(mam4_inputs)

# run a PartMC ensemble
partmc = ambrs.partmc.AerosolModel(
    processes = processes,
    run_type = 'particle',
    n_part = n_part,
    n_repeat = 1,
)
partmc_inputs = partmc.create_inputs(
    ensemble = ensemble,
    dt = dt,
    nstep = nstep
)

# partmc_dir = os.path.join(cwd, 'partmc_runs')
if not os.path.exists(partmc_dir):
    os.mkdir(partmc_dir)

partmc_runner = ambrs.PoolRunner(
    model = partmc,
    executable = 'partmc',
    root = partmc_dir,
    num_processes = 1
)
partmc_runner.run(partmc_inputs)


#partmc_outputs = partmc_runner.run(partmc_inputs) 

# import math
# from PyParticle import build_population
# num_inputs = len(partmc_inputs)
# max_num_digits = math.floor(math.log10(num_inputs)) + 1

# mam4_outputs = []
# partmc_outputs = []
# for ii,(partmc_input,mam4_input) in enumerate(zip(partmc_inputs,mam4_inputs)):
    
#     num_digits = math.floor(math.log10(ii+1)) + 1
#     formatted_index = '0' * (max_num_digits - num_digits) + f'{ii+1}'
#     scenario_name = formatted_index
    
#     # fixme: assuming 1 repeat right now -- "0001" in next line
#     # partmc_output_filename = partmc_dir + '/' + scenario_name + '/out/' + scenario_name + '_0001_' + str(nstep).zfill(8) + '.nc'
#     partmc_dir_onescenario = partmc_dir + '/' + scenario_name + '/'
#     partmc_population_cfg = {
#         'type':'partmc',
#         'partmc_dir': partmc_dir_onescenario,
#         'timestep': nstep,
#         'repeat': 1
#         }
#     partmc_population = build_population(partmc_population_cfg)
#     # partmc_output = ambrs.Output(
#     #     input=partmc_input, model='partmc', 
#     #     config=partmc_population_cfg, 
#     #     population=partmc_population)
    
#     mam4_output_filename = mam4_dir + '/' + scenario_name + '/mam_output.nc'
#     mam4_population_cfg = {
#         'type':'mam4',
#         'output_filename': mam4_output_filename,
#         'timestep':nstep-1,
#         'GSD':[1.6,1.6,1.8,1.3], #fixme: put in the correct GSD values!
#         'D_min':1e-9, #fixme: option for +/- sigmas
#         'D_max':1e-2,
#         'N_bins':int(100),
#         'T':mam4_input.temp,
#         'p':mam4_input.press} 
#     mam4_population = build_population(mam4_population_cfg)
    
    
    
    # mam4_output = ambrs.Output(
    #     input=mam4_input, model='mam4', 
    #     config=mam4_population_cfg, 
    #     population=mam4_population)
    
    # partmc_outputs.append(partmc_output)
    # mam4_outputs.append(mam4_output)
    
from ambrs.visualization.output_plot import plot_ensemble_state
# scenario names -- automatically enumerate based on inputs length
np.ceil(np.log10(len(ensemble)))
scenario_names = [str(i+1).zfill(3) for i in range(len(ensemble))]
try:
    plot_ensemble_state(partmc_dir, mam4_dir, scenario_names, ensemble, timestep=-1)
except Exception as e:
    logging.info(f"plot_ensemble_state failed (probably missing outputs or model bins): {e}")



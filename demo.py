import ambrs
import scipy.stats

# aerosol processes under consideration (all are off by default)
processes = ambrs.AerosolProcesses(
    coagulation = True,
)

# relevant aerosol and gas species
so4 = ambrs.AerosolSpecies(name='so4')
pom = ambrs.AerosolSpecies(name='pom')
soa = ambrs.AerosolSpecies(name='soa')
bc  = ambrs.AerosolSpecies(name='bc')
dst = ambrs.AerosolSpecies(name='dst')
ncl = ambrs.AerosolSpecies(name='ncl')

so2   = ambrs.GasSpecies(name='so2')
h2so4 = ambrs.GasSpecies(name='h2so4')
soag  = ambrs.GasSpecies(name='soag')

# specify all the distributions to sample in the creation of an ensemble
# (taken roughly from get_lhs_ensemble_settings in ensemble_maker.py)
spec = ambrs.EnsembleSpecification(
    name = 'demo',
    size = ambrs.AerosolModalSizeDistribution(
        modes = [
            ambrs.AerosolModeDistribution(
                name = "accumulation",
                species = [so4, pom, soa, bc, dst, ncl],
                number = scipy.stats.loguniform(3e7, 2e12),
                geom_mean_diam = scipy.stats.loguniform(0.5e-7, 1.1e-7),
                mass_fractions = [
                    scipy.stats.uniform(0, 1), # so4
                    scipy.stats.uniform(0, 1), # pom
                    scipy.stats.uniform(0, 1), # soa
                    scipy.stats.uniform(0, 1), # bc
                    scipy.stats.uniform(0, 1), # dst
                    scipy.stats.uniform(0, 1), # ncl
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "aitken",
                species = [so4, soa, ncl],
                number = scipy.stats.loguniform(3e7, 2e12),
                geom_mean_diam = scipy.stats.loguniform(0.5e-8, 3e-8),
                mass_fractions = [
                    scipy.stats.uniform(0, 1), # so4
                    scipy.stats.uniform(0, 1), # soa
                    scipy.stats.uniform(0, 1), # ncl
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "coarse",
                species = [dst, ncl, so4, bc, pom, soa],
                number = scipy.stats.loguniform(3e7, 2e12),
                geom_mean_diam = scipy.stats.loguniform(1e-6, 2e-6),
                mass_fractions = [
                    scipy.stats.uniform(0, 1), # dst
                    scipy.stats.uniform(0, 1), # ncl
                    scipy.stats.uniform(0, 1), # so4
                    scipy.stats.uniform(0, 1), # bc
                    scipy.stats.uniform(0, 1), # pom
                    scipy.stats.uniform(0, 1), # soa
                ],
            ),
            ambrs.AerosolModeDistribution(
                name = "primary carbon",
                species = [pom, bc],
                number = scipy.stats.loguniform(3e7, 2e12),
                geom_mean_diam = scipy.stats.loguniform(1e-8, 6e-8),
                mass_fractions = [
                    scipy.stats.uniform(0, 1), # pom
                    scipy.stats.uniform(0, 1), # bc
                ],
            ),
        ]),
    flux = scipy.stats.loguniform(1e-2*1e-9, 1e1*1e-9),
    relative_humidity = scipy.stats.loguniform(0, 0.99),
    temperature = scipy.stats.uniform(240, 310),
)

# create an ensemble using latin hypercube sampling
n = 100
ensemble = ambrs.lhs(specification = spec, n = n)

# create MAM4 inputs for each ensemble member
mam4_inputs = ambrs.create_mam4_inputs(processes, ensemble)

# create partmc inputs for each ensemble member
n_particles = 10000
partmc_inputs = ambrs.create_partmc_inputs(processes, ensemble, n_particles,
                                           ...)

# run simulations
# ...
for i, input in enumerate(mam4_inputs):
    f = open(f'mam4_{i}.nl', 'w')
    f.write(repr(input))
    f.close()

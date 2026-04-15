from .ppe import EnsembleSpecification
from .aerosol import AerosolSpecies, AerosolModalSizePopulation
from .gas import GasSpecies
import os
import numpy as np
from scipy.optimize import fsolve
from scipy.constants import gas_constant

from math import floor, log10


# FIXME: IN PROGRESS!

####################################################################################################
#> CAMP configuration
####################################################################################################
from .ppe import Ensemble

import json
import math

def activity_coefficient(N_star, T, target=0.65):
    """CAMP acitivity coefficient formulation to compute Nstar by root finding.
       Defaults to MAM4 value of 0.65."""
    del_H = 4.184e3 * (-10.0*(N_star - 1.0) + 7.53*(N_star**(2.0/3.0) - 1.0) - 1.0)
    del_S = 4.184e0 * (-32.0*(N_star - 1.0) + 9.21*(N_star**(2.0/3.0) - 1.0) - 1.3)
    del_G = del_H - T * del_S
    alpha = np.exp(-del_G / (gas_constant * T))
    alpha = alpha / (1.0 + alpha)
    return alpha - target

class CAMP:
    """CAMP: configures CAMP from ambrs PPEs"""
    def __init__(
            self,
            ppe: Ensemble,
            aero_rep_type: str,
            reactions: list[dict],
            species: list[GasSpecies | AerosolSpecies]=None,
            absolute_integration_tolerance: float=1e-6,
            diffusion_coeff: dict=None,
            phase_names: list[str]=None,
            phase_species: list[list[str]]=None,
            layers: list[dict]=None,
            maximum_computational_particles: int=None,
            modes: list[str]=None,
            mechanism_name: str='Mechanism',
    ):
        self.ppe = ppe
        self.species = None
        self.aerosol_phases = None
        self.aero_rep_type = aero_rep_type
        self.aerosol_representation = None
        self.mechanism = None
        self.ambient_conditions = [
            {
                'relative_humidity': member.relative_humidity,
                'temperature': member.temperature,
                'pressure': member.pressure,
            }
            for member in self.ppe
        ]
        self.config = None

        self.configure_species(
            species=species,
            absolute_integration_tolerance=absolute_integration_tolerance,
            diffusion_coeff=diffusion_coeff
        )
        self.configure_aerosol_phases(
            phase_names=phase_names,
            phase_species=phase_species
        )
        self.configure_aerosol_representation(
            aero_rep_type,
            layers=layers,
            maximum_computational_particles=maximum_computational_particles,
            modes=modes
        )
        self.configure_mechanism(
            name=mechanism_name,
            reactions=reactions
        )

    def configure_species(
            self,
            species: list[GasSpecies | AerosolSpecies]=None,
            absolute_integration_tolerance=1e-6,
            diffusion_coeff: dict=None,
    ):
        if species:
            gases_in = [s for s in species if not hasattr(s,'density')]
            aerosols_in = [s for s in species if hasattr(s,'density')]
        else:
            gases_in = self.ppe.gases
            aerosols_in = self.ppe.aerosols

        gases = [
            {
                'name': gas.name,
                'type': 'CHEM_SPEC',
                'absolute integration tolerance': absolute_integration_tolerance,
                'molecular weight [kg mol-1]': 1e-3 * gas.molar_mass,
            }
            for gas in gases_in
        ]
        if diffusion_coeff:
            for gas in gases:
                if gas['name'] in diffusion_coeff.keys():
                    gas['diffusion coeff [m2 s-1]'] = diffusion_coeff[gas['name']]
        is_there_water = False
        for gas in gases:
            if gas['name']=='H2O' or gas['name']=='h2o':
                is_there_water = True
        if not is_there_water:
            gases.append(
                {
                    'name': 'H2O',
                    'type': 'CHEM_SPEC',
                    'absolute integration tolerance': absolute_integration_tolerance,
                    'molecular weight [kg mol-1]': 0.018,
                    'is gas-phase water': True
                }
            )

        aerosols = [
            {
                'name': aerosol.name,
                'type': 'CHEM_SPEC',
                'phase': 'AEROSOL',
                'absolute integration tolerance': absolute_integration_tolerance,
                'molecular weight [kg mol-1]': 1e-3 * aerosol.molar_mass,
                'density [kg m-3]': aerosol.density,
                'num_ions': aerosol.ions_in_soln,
                'kappa': aerosol.hygroscopicity,
            }
            for aerosol in aerosols_in
        ]
        self.species = gases + aerosols
        return self

    def configure_aerosol_phases(
            self,
            phase_names: list[str]=None,
            phase_species: dict[list[str]]=None,
    ):
        if not self.species:
            raise NotImplementedError(
                'Please configure CAMP species before aerosol phases.'
            )
        if phase_names and phase_species:
            phases = [
                {
                    'name': name,
                    'type': 'AERO_PHASE',
                    'species': [s for s in phase_species[name]]
                }
                for name in phase_names
            ]
        else:
            phases = [
                {
                    'name': 'mixed',
                    'type': 'AERO_PHASE',
                    'species': [s['name'] for s in self.species if 'phase' in s.keys()]
                }
            ]
        self.aerosol_phases = phases
        return self
    
    def configure_aerosol_representation(
            self,
            type: str,
            layers: list[dict]=None,
            maximum_computational_particles: int=None,
            modes: list[AerosolModalSizePopulation]=None,
    ):
        if not self.aerosol_phases:
            raise NotImplementedError(
                'Please configure CAMP aerosol phases before aerosol representation.'
            )
        if type not in ['AERO_REP_SINGLE_PARTICLE','AERO_REP_MODAL_BINNED_MASS']:
            raise ValueError(
                'type must be AERO_REP_SINGLE_PARTICLE or AERO_REP_MODAL_BINNED_MASS.'
            )
        if type=='AERO_REP_SINGLE_PARTICLE' and not maximum_computational_particles:
            raise NotImplementedError(
                'For AERO_REP_SINGLE_PARTICLE representation, need to specify maximum_computational_particles.'
            )
        
        self.aero_rep_type = type

        if not modes:
            modes = self.ppe.size.modes

        if type=='AERO_REP_SINGLE_PARTICLE':
            if layers:
                aero_rep = {
                    'name': 'PartMC single particle',
                    'type': type,
                    'layers': layers,
                    'maximum computational particles': maximum_computational_particles,
                }
            else:
                aero_rep = {
                    'name': 'PartMC single particle',
                    'type': type,
                    'layers': [
                        {
                            'name': 'core',
                            'covers': 'none',
                            'phases': [phase['name'] for phase in self.aerosol_phases]
                        }
                    ],
                    'maximum computational particles': maximum_computational_particles,
                }
        elif type=='AERO_REP_MODAL_BINNED_MASS':
            aero_rep = [{
                'name': 'Modal/binned',
                'type': type,
                'modes/bins':
                {
                    mode.name: {
                        'type': 'MODAL',
                        'phases': [phase['name'] for phase in self.aerosol_phases if phase['name']==mode.name],
                        'shape': 'LOG_NORMAL',
                        'geometric mean diameter [m]': mode.geom_mean_diam.item(),
                        'geometric standard deviation': 10**mode.log10_geom_std_dev.item(),
                    }
                    for mode in scenario.size.modes
                }
            } for scenario in self.ppe]
        self.aerosol_representation = aero_rep
        return self
    
    def configure_mechanism(
            self,
            name: str='Mechanism',
            reactions: list[dict]=None
    ):
        if not reactions:
            raise ValueError(
                'Need to specify mechanism reactions'
            )
        mech = {
            'name': name,
            'type': 'MECHANISM',
            'reactions': reactions
        }
        self.mechanism = mech
        return self

    def configure(
            self,
            root: str
        ):

        # prep scenarios to run
        num_inputs = len(self.ppe)
        max_num_digits = math.floor(math.log10(num_inputs)) + 1

        for i, member in enumerate(self.ppe):

            self.config = f'{root}/camp.json'

            for species in self.species:
                for key,value in species.items():
                    if callable(value):
                        species[key] = value(**self.ambient_conditions[i])
            for reaction in self.mechanism['reactions']:
                for key,value in reaction.items():
                    if isinstance(value,list):
                        for j,subvalue in enumerate(value):
                            if callable(subvalue):
                                reaction[key][j] = subvalue(**self.ambient_conditions[i])
                    elif callable(value):
                        reaction[key] = value(**self.ambient_conditions[i])

            if self.aero_rep_type=='AERO_REP_MODAL_BINNED_MASS':
                camp = {
                    'camp-data': [
                        self.aerosol_representation[i],
                        *self.aerosol_phases,
                        *self.species,
                        self.mechanism,
                        {
                            'type': 'RELATIVE_TOLERANCE',
                            'value': 1e-6
                        }
                    ],
                    'camp-files': [
                        self.config
                    ]
                }
            else:
                camp = {
                    'camp-data': [
                        self.aerosol_representation,
                        *self.aerosol_phases,
                        *self.species,
                        self.mechanism,
                        {
                            'type': 'RELATIVE_TOLERANCE',
                            'value': 1e-6
                        }
                    ],
                    'camp-files': [
                        self.config
                    ]
                }

            with open(self.config, 'w') as f:
                json.dump(camp, f, indent=4)
            f.close()

####################################################################################################
#> End
####################################################################################################

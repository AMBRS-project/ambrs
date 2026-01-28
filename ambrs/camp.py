from .ppe import Ensemble

import os
import json
import math

# from scipy.optimize import fsolve
from scipy.constants import gas_constant

def activity_coefficient(N_star, T, target=0.65):
    """CAMP acitivity coefficient formulation to compute Nstar by root finding.
       Defaults to MAM4 value of 0.65."""
    del_H = 4.184e3 * (-10.0*(N_star - 1.0) + 7.53*(N_star**(2.0/3.0) - 1.0) - 1.0)
    del_S = 4.184e0 * (-32.0*(N_star - 1.0) + 9.21*(N_star**(2.0/3.0) - 1.0) - 1.3)
    del_G = del_H - T * del_S
    alpha = math.exp(-del_G / (gas_constant * T))
    alpha = alpha / (1.0 + alpha)
    return alpha - target

class CAMP:
    """CAMP: configures CAMP from ambrs PPEs"""
    def __init__(
            self,
            ppe: Ensemble
    ):
        self.ppe = ppe
        self.species = None
        self.aerosol_phases = None
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

    def configure_species(
            self,
            absolute_integration_tolerance=1e-6,
            diffusion_coeff:dict=None,
    ):
        gases = [
            {
                'name': gas.name,
                'type': 'CHEM_SPEC',
                'absolute integration tolerance': absolute_integration_tolerance,
                'molecular weight [kg mol-1]': 1e-3 * gas.molar_mass,
            }
            for gas in self.ppe.gases
        ]
        if diffusion_coeff:
            for gas in gases:
                if gas['name'] in diffusion_coeff.keys():
                    gas['diffusion coeff [m2 s-1]'] = diffusion_coeff[gas['name']]
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
            for aerosol in self.ppe.aerosols
        ]
        self.species = gases + aerosols
        return self

    def configure_aerosol_phases(
            self,
            phase_names:list[str]=None,
            phase_species:list[list[str]]=None,
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
                    'species': [s['name'] for s in species if 'phase' in s.keys()]
                }
                for name, species in zip(phase_names,phase_species)
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
            type:str,
            layers:list[dict]=None,
            maximum_computational_particles:int=None,
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
            aero_rep = {
                'name': 'Modal/binned',
                'type': type,
                'modes/bins':
                {
                    mode.name: {
                        'type': 'MODAL',
                        'phases': self.aerosol_phases,
                        'shape': 'LOG_NORMAL',
                        'geometric mean diameter [m]': mode.geom_mean_diam,
                        'geometric standard deviation': 10**mode.log10_geom_std_dev,
                    }
                    for mode in self.ppe.size.modes
                }
            }
        self.aerosol_representation = aero_rep
        return self
    
    def configure_mechanism(
            self,
            name:str='Mechanism',
            reactions:list[dict]=None
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
            dir:str,
    ):
        for i, member in enumerate(self.ppe):

            for species in self.species:
                for key,value in species.items():
                    if callable(value):
                        species[key] = value(**self.ambient_conditions[i])
            for reaction in self.mechanism['reactions']:
                for key,value in reaction.items():
                    if isinstance(value,list):
                        for i,subvalue in enumerate(value):
                            if callable(subvalue):
                                reaction[key][i] = subvalue(**self.ambient_conditions[i])
                    elif callable(value):
                        reaction[key] = value(**self.ambient_conditions[i])

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
                    f'{dir}/camp.json'
                ]
            }
            with open(f'{dir}/camp.json', 'w') as f:
                json.dump(camp, f, indent=4)
            f.close()

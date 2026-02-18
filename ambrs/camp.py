from .ppe import EnsembleSpecification
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
    alpha = math.exp(-del_G / (gas_constant * T))
    alpha = alpha / (1.0 + alpha)
    return alpha - target

class CAMP:
    """CAMP: configures CAMP from ambrs PPEs"""
    def __init__(
            self,
            ppe:Ensemble,
            aero_rep_type:str,
            reactions:list[dict],
            absolute_integration_tolerance:float=1e-6,
            diffusion_coeff:dict=None,
            phase_names:list[str]=None,
            phase_species:list[list[str]]=None,
            layers:list[dict]=None,
            maximum_computational_particles:int=None,
            mechanism_name:str='Mechanism',
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
            maximum_computational_particles=maximum_computational_particles
        )
        self.configure_mechanism(
            name=mechanism_name,
            reactions=reactions
        )

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
        
        self.aero_rep_type = type

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
                        'phases': [phase['name'] for phase in self.aerosol_phases],
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
            root:str
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

# orig: 
# from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, os, sys
from typing import Iterable, Mapping, Optional

def _p(pathlike) -> str:
    return str(Path(pathlike).resolve())

@dataclass
class CampConfig:
    """
    General CAMP helper:
      - Builds species/phases JSON and one or more mechanism JSONs
      - Writes a "camp_files.json" file list the host model points to
      - Provides a runtime_env() that makes libcamp discoverable
    """
    mechanism_name: str = "ambrs_mechanism"
    explicit_lib_dirs: Iterable[Path] = field(default_factory=list)

    # ---- General SIMPOL builder ----
    def simpol_phase_transfer(self,
                              gas_species: str,
                              aero_species: str,
                              B: Mapping[str, float],
                              N_star: float,
                              phase_name: str = "mixed") -> dict:
        """
        Create a SIMPOL_PHASE_TRANSFER reaction from gas to aerosol species.
        """
        return {
            "type": "SIMPOL_PHASE_TRANSFER",
            "gas_species": gas_species,
            "aerosol_species": aero_species,
            "aerosol_phase": phase_name,
            "B": {
                "B0": B.get("B0", 0.0),
                "B1": B.get("B1", 0.0),
                "B2": B.get("B2", 0.0),
                "B3": B.get("B3", 0.0),
            },
            "N*": N_star
        }

    # ---- Files ----
    def _species_block(self) -> dict:
        # Keep this general. Include the species we know we’ll need; caller can extend later.
        # Molecular weights in kg/mol (match what CAMP requires).
        return {
            "species": [
                {"name": "H2SO4", "phase": "gas",     "molecular_weight": 0.098079},
                {"name": "SOAG",  "phase": "gas",     "molecular_weight": 0.150000},
                # Aerosol species that host condensation products:
                {"name": "SO4",   "phase": "aerosol", "molecular_weight": 0.11510734, "aerosol_phase": "mixed"},
                {"name": "SOA",   "phase": "aerosol", "molecular_weight": 0.150000,   "aerosol_phase": "mixed"},
            ]
        }

    def _phases_block(self) -> dict:
        return {
            "gas_phases": [{"name": "gas"}],
            "aerosol_phases": [{"name": "mixed"}],
        }

    def _mechanism_block(self) -> dict:
        # H2SO4 -> SO4 : essentially nonvolatile (big N*), “irreversible” condensation surrogate
        h2so4 = self.simpol_phase_transfer(
            gas_species="H2SO4",
            aero_species="SO4",
            B={"B0": 0.0, "B1": 0.0, "B2": 0.0, "B3": 0.0},
            N_star=1e20,
        )
        # SOAG -> SOA : semi-volatile-ish; tune later as needed
        soag = self.simpol_phase_transfer(
            gas_species="SOAG",
            aero_species="SOA",
            B={"B0": -7.0, "B1": 0.0, "B2": 0.0, "B3": 0.0},
            N_star=1e10,
        )
        return {"mechanism": {"name": self.mechanism_name, "reactions": [h2so4, soag]}}

    # ---- Write set for a given model/scenario dir ----
    def write_for_model(self, run_dir: Path | str, model_name: str, include: Optional[dict] = None) -> Path:
        """
        Writes: <run_dir>/camp/<model_name>/{species.json,phases.json,mechanism.json,camp_files.json}
        Returns absolute path to camp_files.json
        """
        base = Path(run_dir).resolve() / "camp" / model_name
        base.mkdir(parents=True, exist_ok=True)

        species = self._species_block()
        phases  = self._phases_block()
        mech    = self._mechanism_block()

        # Allow caller to extend with additional mechanisms/species/phases
        if include:
            if "species" in include: species["species"].extend(include["species"])
            if "gas_phases" in include or "aerosol_phases" in include:
                phases["gas_phases"].extend(include.get("gas_phases", []))
                phases["aerosol_phases"].extend(include.get("aerosol_phases", []))
            if "reactions" in include:
                mech["mechanism"]["reactions"].extend(include["reactions"])

        sp = base / "species.json"; sp.write_text(json.dumps(species, indent=2))
        ph = base / "phases.json";  ph.write_text(json.dumps(phases,  indent=2))
        me = base / "mechanism.json"; me.write_text(json.dumps(mech,   indent=2))

        file_list = {
            "species":   [_p(sp)],
            "phases":    [_p(ph)],
            "mechanism": [_p(me)]
        }
        fl = base / "camp_files.json"
        fl.write_text(json.dumps(file_list, indent=2))
        return fl

    # ---- Env for dynamic loader on macOS/Linux ----
    def runtime_env(self) -> dict:
        env = {}
        # Prefer explicit lib dirs if supplied (e.g., $(CONDA_PREFIX)/lib)
        candidates = [Path(d) for d in self.explicit_lib_dirs] if self.explicit_lib_dirs else []
        if not candidates:
            cp = os.environ.get("CONDA_PREFIX", "")
            if cp:
                candidates.append(Path(cp)/"lib")

        # macOS: DYLD_FALLBACK_LIBRARY_PATH; Linux: LD_LIBRARY_PATH
        lib_str = ":".join(str(p) for p in candidates if p.exists())
        if lib_str:
            if sys.platform == "darwin":
                env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{lib_str}:{os.environ.get('DYLD_FALLBACK_LIBRARY_PATH','')}".rstrip(":")
            elif sys.platform.startswith("linux"):
                env["LD_LIBRARY_PATH"] = f"{lib_str}:{os.environ.get('LD_LIBRARY_PATH','')}".rstrip(":")
        return env



def accoef(N_star, T):
    del_H = 4.184e3 * (-10.0*(N_star - 1.0) + 7.53*(N_star**(2.0/3.0) - 1.0) - 1.0)
    del_S = 4.184e0 * (-32.0*(N_star - 1.0) + 9.21*(N_star**(2.0/3.0) - 1.0) - 1.3)
    del_G = del_H - T * del_S
    alpha = np.exp(-del_G / (gas_constant * T))
    alpha = alpha / (1.0 + alpha)
    return alpha - 0.65

from .ppe import EnsembleSpecification
import os
import numpy as np
from scipy.optimize import fsolve
from scipy.constants import gas_constant


# FIXME: IN PROGRESS!

####################################################################################################
#> CAMP configuration
####################################################################################################
class CAMP:
    """ambrs.CAMP -- a helper to write CAMP configuration files for a single scenrio"""
    def __init__(self, spec: EnsembleSpecification, scenario:int):
        self.temps = spec.temperature
        self.pres = spec.pressure
        self.n = self.temps.size
        self.scenario = scenario

    def write_config_json(self, n_part_max:int=1100):
        for member in range(self.n):
            path = f'/Users/duncancq/Research/AMBRS/aero_unit_tests/alpha-pinene/camp_config/scenarios/{self.scenario}/{member+1:0>3}'
            json = \
'''{
    "camp-data" : [
        {
            "name" : "mixed",
            "type" : "AERO_PHASE",
            "species" : [
                "SO4",
                "POM",
                "SOA",
                "BC",
                "DST",
                "NCL",
                "MOM"
            ]
        }
    ]
}
'''
            if not os.path.exists(path):
                os.mkdir(path)
            with open(f'{path}/aero_phases.json', 'w') as f:
                f.write(json)
            f.close()

            json = \
f'''{{
  "camp-data" : [
    {{
      "name" : "PartMC single particle",
      "type" : "AERO_REP_SINGLE_PARTICLE",
      "layers": [
	      {{
		      "name": "core",
		      "covers": "none",
		      "phases": [
			      "mixed"
		      ]
	      }}
      ],
      "maximum computational particles" : {n_part_max}
    }}
  ]
}}
'''
            if not os.path.exists(path):
                os.mkdir(path)
            with open(f'{path}/aero_rep.json', 'w') as f:
                f.write(json)
            f.close()

            json = \
f'''{{
    "camp-files" : [
        "{path}/aero_phases.json",
        "{path}/aero_rep.json",
        "{path}/species.json",
        "{path}/mech.json"
    ]
}}
'''
            if not os.path.exists(path):
                os.mkdir(path)
            with open(f'{path}/config.json', 'w') as f:
                f.write(json)
            f.close()
        return self

    def write_mech_json(self):
        delta__H_v = 156e3
        log10e = np.log10(np.e)
        B1 = -delta__H_v * log10e / gas_constant
        B2 = -10 + delta__H_v * log10e / (298*gas_constant)
        for member in range(self.n):
            N_star = fsolve(accoef, 1.1, args=(self.temps[member],))
            json = \
f'''{{
    "camp-data" : [
        {{
            "type" : "RELATIVE_TOLERANCE",
            "value" : 1.0e-10
        }},
        {{
            "name" : "MAM4_SOA_partitioning",
            "type" : "MECHANISM",
            "reactions" : [
                {{
                    "type" : "SIMPOL_PHASE_TRANSFER",
                    "gas-phase species" : "SOAG",
                    "aerosol phase" : "mixed",
                    "aerosol-phase species" : "SOA",
                    "B" : [ {B1}, {B2}, 0.0, 0.0 ],
                    "N star" : {N_star}
                }}
            ]
        }}
    ]
}}
'''
            if not os.path.exists(path):
                os.mkdir(path)
            with open(f'{path}/mech.json', 'w') as f:
                f.write(json)
            f.close()
        return self
    
    # fixme: hard-coded for MAM4 species for now, add flexibility later
    def write_species_json(self):
        diffus = 0.557e-4 * (self.temps**1.75) / self.pres
        mam_spec = [
            {
                "name" : "SO2",
                "molecular weight [kg mol-1]" : 0.0640
            },
            {
                "name" : "SO4",
                "phase" : "AEROSOL",
                "molecular weight [kg mol-1]" : 0.115107340,
                "density [kg m-3]" : 1770.0000,
                "num_ions" : 2,
                "charge" : -2,
                "kappa" : 0.0
            },
            {
                "name" : "H2O",
                "molecular weight [kg mol-1]" : 0.018,
                "is gas-phase water" : True
            },
            {
                "name" : "POM",
                "phase" : "AEROSOL",
                "molecular weight [kg mol-1]" : 0.012011,
                "density [kg m-3]" : 1000.0000,
                "num_ions" : 0,
                "charge" : 0,
                "kappa" : 0.010
            },
            {
                "name" : "SOA",
                "phase" : "AEROSOL",
                "molecular weight [kg mol-1]" : 0.012011,
                "density [kg m-3]" : 1000.0000,
                "num_ions" : 0,
                "charge" : 0,
                "kappa" : 0.140
            },
            {
                "name" : "BC",
                "phase" : "AEROSOL",
                "molecular weight [kg mol-1]" : 0.012011,
                "density [kg m-3]" : 1700.0000,
                "num_ions" : 0,
                "charge" : 0,
                "kappa" : 1.0e-10
            },
            {
                "name" : "DST",
                "phase" : "AEROSOL",
                "molecular weight [kg mol-1]" : 0.135064039,
                "density [kg m-3]" : 2600.0000,
                "num_ions" : 0,
                "charge" : 0,
                "kappa" : 0.068
            },
            {
                "name" : "NCL",
                "phase" : "AEROSOL",
                "molecular weight [kg mol-1]" : 0.058442468,
                "density [kg m-3]" : 1900.0000,
                "num_ions" : 0,
                "charge" : 0,
                "kappa" : 1.160
            },
            {
                "name" : "MOM",
                "phase" : "AEROSOL",
                "molecular weight [kg mol-1]" : 250.092672000,
                "density [kg m-3]" : 1601.0000,
                "num_ions" : 0,
                "charge" : 0,
                "kappa" : 0.100
            }
        ]
        for member in range(self.n):
            json = \
'''{
    "camp-data": ['''
            tdep_mam_spec = [
                {
                    'name': 'SOAG',
                    'molecular weight [kg mol-1]': 0.012011,
                    'diffusion coeff [m2 s-1]': 0.81 * diffus[member]
                },
                {
                    'name': 'H2SO4',
                    'molecular weight [kg mol-1]': 0.098,
                    'diffusion coeff [m2 s-1]': diffus[member],
                }
            ]
            for species in [*mam_spec,*tdep_mam_spec]:
                json += \
'''
        {
                "type": "CHEM_SPEC",
                "absolute integration tolerance" : 1e-06,'''
                for key, value in species.items():
                    if type(value)==str:
                        json += \
f'''
                "{key}": "{value}",'''
                    elif type(value)==bool and value:
                        json += \
f'''
                "{key}": true,'''

                    else:
                        json += \
f'''
                "{key}": {value},'''
                json = json[:-1] + \
'''
        },'''
            json = json[:-1] + \
'''
    ]
}
'''
            path = f'/Users/duncancq/Research/AMBRS/aero_unit_tests/alpha-pinene/camp_config/scenarios/{self.scenario}/{member+1:0>3}'
            if not os.path.exists(path):
                os.mkdir(path)
            with open(f'{path}/species.json', 'w') as f:
                f.write(json)
            f.close()
        return self

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

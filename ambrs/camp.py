"""
ambrs.camp -- generalized CAMP config builder (mechanisms & files)

Goals:
- Common files (species, aerosol phases) shared by models.
- Model-specific files (aerosol representation + mechanism).
- Produce per-scenario file list: camp/camp_files.json.
- Provide runtime env (CAMP_FILE_LIST + DYLD/LD paths) automatically.

Example mechanism here:
  * H2SO4 (gas) -> SO4 (aerosol) via first-order sink (rate k [s^-1])
  * SOAG partitioning to SOA via SIMPOL parameters

This is an example; the public API is generic enough to plug other species/reactions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union
import json
import os
import sys


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class MechanismTemplate:
    """Generic holder for a small set of reactions; today we expose a focused
    H2SO4 + SOAG example but keep the shape extensible."""
    # First-order H2SO4 -> SO4 rate [s^-1]; None disables
    h2so4_first_order_k: Optional[float] = None

    # SIMPOL (SOAG -> SOA) parameters
    simpol_B: Optional[list[float]] = None
    simpol_Nstar: Optional[float] = None

    # phases/species names used inside CAMP
    gas_h2so4: str = "H2SO4"
    aer_so4: str = "SO4"
    gas_soag: str = "SOAG"
    aer_soa: str = "SOA"
    aerosol_phase_name: str = "mixed"

    def is_empty(self) -> bool:
        return (self.h2so4_first_order_k is None) and (self.simpol_B is None or self.simpol_Nstar is None)


@dataclass
class CampConfig:
    """Builder for CAMP configs."""
    # Optional explicit path to lib dir (otherwise we auto-detect).
    lib_dir: Optional[Union[str, Path]] = None

    # Mechanism sources. Choose ONE of:
    #  (a) template: MechanismTemplate
    #  (b) files: explicit list of config files (absolute or relative)
    #  (c) directory: a directory containing JSON files to list
    template: Optional[MechanismTemplate] = None
    files: Optional[Iterable[Union[str, Path]]] = None
    directory: Optional[Union[str, Path]] = None

    # Internal cache: set per scenario when we write
    _last_camp_files_json: Optional[Path] = field(default=None, init=False, repr=False)

    # -------------------
    # Public entry points
    # -------------------

    def write_for_model(self, run_dir: Union[str, Path], model_name: str) -> Path:
        """Create <run_dir>/camp/* and return the file-list JSON path."""
        run_dir = Path(run_dir)
        camp_dir = _ensure_dir(run_dir / "camp")

        # Case: files/dir provided verbatim
        if self.files or self.directory:
            files = []
            if self.files:
                files.extend([str(Path(f)) for f in self.files])
            if self.directory:
                d = Path(self.directory)
                files.extend([str(p.resolve()) for p in sorted(d.glob("*.json"))])
            out = {"camp-files": files}
            camp_files_json = camp_dir / "camp_files.json"
            camp_files_json.write_text(json.dumps(out, indent=2))
            self._last_camp_files_json = camp_files_json
            return camp_files_json

        # Case: build from a template (today: H2SO4 + SOAG example)
        tmpl = self.template or MechanismTemplate()
        # 1) species.json  2) aerosol_phases.json  3) aero_rep.json  4) mechanism.json
        species = self._species_block(tmpl)
        (camp_dir / "species.json").write_text(json.dumps({"camp-data": species}, indent=2))

        phases = self._aerosol_phases_block(tmpl)
        (camp_dir / "aerosol_phases.json").write_text(json.dumps({"camp-data": phases}, indent=2))

        aero_rep = self._aero_rep_block(model_name, tmpl)
        rep_name = "mam4_aerosol_representation.json" if model_name.lower().startswith("mam") else "partmc_aerosol_representation.json"
        (camp_dir / rep_name).write_text(json.dumps({"camp-data": [aero_rep]}, indent=2))

        mech = self._mechanism_block(model_name, tmpl)
        mech_name = "mam_mech.json" if model_name.lower().startswith("mam") else "pmc_mech.json"
        (camp_dir / mech_name).write_text(json.dumps({"camp-data": mech}, indent=2))

        # File list
        file_list = [
            str((camp_dir / "aerosol_phases.json").resolve()),
            str((camp_dir / rep_name).resolve()),
            str((camp_dir / "species.json").resolve()),
            str((camp_dir / mech_name).resolve()),
        ]
        camp_files_json = camp_dir / "camp_files.json"
        camp_files_json.write_text(json.dumps({"camp-files": file_list}, indent=2))
        self._last_camp_files_json = camp_files_json
        return camp_files_json

    def runtime_env(self, run_dir: Optional[Union[str, Path]] = None) -> dict:
        """Environment variables for subprocess:
           - CAMP_FILE_LIST points to camp_files.json (if we wrote one).
           - DYLD_FALLBACK_LIBRARY_PATH / LD_LIBRARY_PATH points to libcamp.
        """
        env = {}

        # CAMP file list
        camp_file = None
        if self._last_camp_files_json and self._last_camp_files_json.exists():
            camp_file = self._last_camp_files_json
        elif run_dir:
            candidate = Path(run_dir) / "camp" / "camp_files.json"
            if candidate.exists():
                camp_file = candidate
        if camp_file:
            env["CAMP_FILE_LIST"] = str(camp_file.resolve())

        # Shared library path
        lib_dir = self._detect_lib_dir()
        if lib_dir:
            lib_dir = str(Path(lib_dir).resolve())
            if sys.platform == "darwin":
                key = "DYLD_FALLBACK_LIBRARY_PATH"
            elif sys.platform.startswith("linux"):
                key = "LD_LIBRARY_PATH"
            else:
                key = None
            if key:
                current = os.environ.get(key, "")
                env[key] = f"{lib_dir}:{current}" if current else lib_dir

        return env

    # -------------------
    # Builders
    # -------------------

    def _detect_lib_dir(self) -> Optional[Path]:
        if self.lib_dir:
            return Path(self.lib_dir)
        # Best-effort: CONDA_PREFIX/lib
        conda = os.environ.get("CONDA_PREFIX")
        if conda:
            cand = Path(conda) / "lib"
            if cand.exists():
                return cand
        # Nothing else reliable without probing binaries
        return None

    def _species_block(self, tmpl: MechanismTemplate) -> list[dict]:
        """Minimal but sufficient set for H2SO4 + SOAG example, using CAMP schema
        keys as in the attached configs: 'molecular weight [kg mol-1]', etc."""
        out = []

        # Gas species
        out.append({
            "name": tmpl.gas_soag,
            "type": "CHEM_SPEC",
            "absolute integration tolerance": 1.0e-6,
            "molecular weight [kg mol-1]": 0.012011,      # placeholder (matches example config)
            "diffusion coeff [m2 s-1]": 9.517e-6
        })
        out.append({
            "name": tmpl.gas_h2so4,
            "type": "CHEM_SPEC",
            "absolute integration tolerance": 1.0e-6,
            "molecular weight [kg mol-1]": 0.098
        })

        # Aerosol species (SOA & SO4)
        out.append({
            "name": tmpl.aer_soa,
            "type": "CHEM_SPEC",
            "phase": "AEROSOL",
            "absolute integration tolerance": 1.0e-6,
            "molecular weight [kg mol-1]": 0.012011,
            "density [kg m-3]": 1000.0,
            "num_ions": 0,
            "charge": 0,
            "kappa": 0.14
        })
        out.append({
            "name": tmpl.aer_so4,
            "type": "CHEM_SPEC",
            "phase": "AEROSOL",
            "absolute integration tolerance": 1.0e-6,
            "molecular weight [kg mol-1]": 0.11510734,   # consistent with example
            "density [kg m-3]": 1770.0,
            "num_ions": 2,
            "charge": -2,
            "kappa": 0.0
        })
        return out

    def _aerosol_phases_block(self, tmpl: MechanismTemplate) -> list[dict]:
        return [{
            "name": tmpl.aerosol_phase_name,
            "type": "AERO_PHASE",
            "species": [tmpl.aer_so4, tmpl.aer_soa]
        }]

    def _aero_rep_block(self, model_name: str, tmpl: MechanismTemplate) -> dict:
        if model_name.lower().startswith("mam"):
            return {
                "name": "MAM4",
                "type": "AERO_REP_MODAL_BINNED_MASS",
                "modes/bins": {
                    "accumulation    ": {
                        "type": "MODAL",
                        "phases": [tmpl.aerosol_phase_name],
                        "shape": "LOG_NORMAL",
                        "geometric mean diameter [m]": 1.1e-7,
                        "geometric standard deviation": 1.8
                    },
                    "aitken          ": {
                        "type": "MODAL",
                        "phases": [tmpl.aerosol_phase_name],
                        "shape": "LOG_NORMAL",
                        "geometric mean diameter [m]": 2.6e-8,
                        "geometric standard deviation": 1.6
                    },
                    "coarse          ": {
                        "type": "MODAL",
                        "phases": [tmpl.aerosol_phase_name],
                        "shape": "LOG_NORMAL",
                        "geometric mean diameter [m]": 2e-6,
                        "geometric standard deviation": 1.8
                    },
                    "primary_carbon  ": {
                        "type": "MODAL",
                        "phases": [tmpl.aerosol_phase_name],
                        "shape": "LOG_NORMAL",
                        "geometric mean diameter [m]": 5e-8,
                        "geometric standard deviation": 1.6
                    }
                }
            }
        else:
            # PartMC
            return {
                "name": "PartMC single particle",
                "type": "AERO_REP_SINGLE_PARTICLE",
                "layers": [{
                    "name": "core", "covers": "none", "phases": [tmpl.aerosol_phase_name]
                }],
                "maximum computational particles": 2000  # safe default; PartMC itself sets actual n_part
            }

    def _mechanism_block(self, model_name: str, tmpl: MechanismTemplate) -> list[dict]:
        mech: list[dict] = [{"type": "RELATIVE_TOLERANCE", "value": 1.0e-10}]

        # SOAG partitioning via SIMPOL (works in both MAM & PartMC)
        if tmpl.simpol_B is not None and tmpl.simpol_Nstar is not None:
            mech.append({
                "name": "SOAG_partitioning",
                "type": "MECHANISM",
                "reactions": [{
                    "type": "SIMPOL_PHASE_TRANSFER",
                    "gas-phase species": tmpl.gas_soag,
                    "aerosol phase": tmpl.aerosol_phase_name,
                    "aerosol-phase species": tmpl.aer_soa,
                    "B": tmpl.simpol_B,          # [B0, B1, B2, B3] like example configs
                    "N star": tmpl.simpol_Nstar
                }]
            })

        # H2SO4 gas -> SO4 aerosol as first-order transfer (simple sink)
        if tmpl.h2so4_first_order_k is not None:
            mech.append({
                "name": "H2SO4_condensation",
                "type": "MECHANISM",
                "reactions": [{
                    # This reaction type is the simple, explicit first-order pathway:
                    # gas H2SO4 is removed with rate k and mass is added to aerosol SO4.
                    "type": "FIRST_ORDER_PHASE_TRANSFER",
                    "gas-phase species": tmpl.gas_h2so4,
                    "aerosol phase": tmpl.aerosol_phase_name,
                    "aerosol-phase species": tmpl.aer_so4,
                    "k [s-1]": tmpl.h2so4_first_order_k
                }]
            })

        return mech

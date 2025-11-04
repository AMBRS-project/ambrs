# ambrs/camp.py
from __future__ import annotations
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

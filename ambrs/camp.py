# ambrs/camp.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Iterable, Optional

@dataclass
class CampConfig:
    """
    Builds CAMP config files for a scenario and returns the path to the top-level
    file-list JSON that CAMP expects.

    Design:
      - COMMON files (shared by MAM4 & PartMC): species, mechanism stub, solver.
      - MODEL-specific override (optional): empty by default, but gives each model
        its own hook without "crazy" config sprawl.
      - A file-list JSON (`camp_files.json`) that lists all of the above in order.

    You can pass an explicit list of gas species; otherwise we’ll generate a
    minimal superset that works for the demo (SO2, H2SO4, and a placeholder SOAG).
    """
    mechanism_name: str = "ambrs_mech"
    rel_tol: float = 1.0e-6
    abs_tol_ppm: float = 0.0      # 0 lets CAMP choose defaults
    gas_species: Optional[Iterable[str]] = None

    # internal cache for last produced file-list (per directory/model)
    _last_files: dict[tuple[Path, str], Path] = field(default_factory=dict, init=False, repr=False)

    def _species_list(self, user_gases: Optional[Iterable[str]]) -> list[str]:
        if user_gases:
            return sorted({g for g in user_gases})
        # conservative default superset that matches the demo & keeps CAMP happy
        return ["SO2", "H2SO4", "SOAG"]

    def write_common_files(self, camp_dir: Path, gases: Optional[Iterable[str]] = None) -> dict[str, Path]:
        """
        Writes common CAMP files (species, mechanism stub with no reactions,
        and solver tolerances). Returns their paths.
        """
        camp_dir.mkdir(parents=True, exist_ok=True)

        species = self._species_list(gases)
        # 1) species.json
        species_json = {
            "camp-data": [
                {"type": "CHEM_SPEC", "name": s, "phase": "gas"} for s in species
            ]
        }

        # 2) mechanism.json (stub: zero reactions -> identity chemistry; safe default)
        mech_json = {
            "camp-data": [
                {
                    "type": "MECHANISM",
                    "name": self.mechanism_name,
                    "reactions": []  # You can add reactions later without changing ambrs.
                }
            ]
        }

        # 3) solver.json (tolerances, OK if host model ignores some fields)
        solver_json = {
            "camp-data": [
                {"type": "SOLVER", "name": "default",
                 "rel_tol": self.rel_tol,
                 # CAMP uses ppm for gas-state by default; abs_tol array is optional
                 "abs_tol": self.abs_tol_ppm}
            ]
        }

        files = {
            "species": camp_dir / "species.json",
            "mechanism": camp_dir / "mechanism.json",
            "solver": camp_dir / "solver.json",
        }
        with files["species"].open("w") as f:
            json.dump(species_json, f, indent=2)
        with files["mechanism"].open("w") as f:
            json.dump(mech_json, f, indent=2)
        with files["solver"].open("w") as f:
            json.dump(solver_json, f, indent=2)

        return files

    def write_for_model(self, root_dir: Path, model_name: str, gases: Optional[Iterable[str]] = None) -> Path:
        """
        Creates (if needed) the full CAMP config for `model_name` under
        <root_dir>/camp/, returning the path to 'camp_files.json' (file-list).

        This file-list is what both PartMC and MAM4 pass to CAMP.
        """
        key = (root_dir.resolve(), model_name)
        if key in self._last_files:
            return self._last_files[key]

        camp_dir = (root_dir / "camp")
        common = self.write_common_files(camp_dir, gases)

        # Optional model-specific override hook (empty; safe to include).
        override = camp_dir / f"{model_name}_override.json"
        if not override.exists():
            with override.open("w") as f:
                json.dump({"camp-data": []}, f, indent=2)

        # File-list JSON that CAMP expects: { "camp-files": [ ... ] }
        file_list = camp_dir / "camp_files.json"
        file_list_json = {
            "camp-files": [
                str(common["species"].name),
                str(common["mechanism"].name),
                str(common["solver"].name),
                str(override.name),
            ]
        }
        with file_list.open("w") as f:
            json.dump(file_list_json, f, indent=2)

        # cache and return a *relative* path usable in input files
        self._last_files[key] = file_list
        return file_list

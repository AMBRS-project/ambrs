# ambrs/camp.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import os

@dataclass
class CampConfig:
    """
    Common CAMP configuration made once and reused by MAM4 and PartMC writers.

    Parameters
    ----------
    lib_dir : Optional[str]
        Directory that contains libcamp.* (for macOS DYLD issues and Linux LD).
        If None, we won't set any library env vars.
    enable_h2so4_condensation : bool
        If True, include a simple first-order H2SO4 gas sink in the CAMP mechanism.
        This is a *placeholder* condensation scheme (gas -> sink).
        Leave False if your model (MAM gas-aerosol or PartMC-MOSAIC) will handle condensation.
    h2so4_cond_rate_s_inv : float
        The first-order rate constant [s^-1] used when enable_h2so4_condensation is True.
        Pick a timescale ~ 30 min => 5.6e-4 s^-1, 1 hour => ~2.8e-4 s^-1, etc.
    """
    lib_dir: Optional[str] = None
    enable_h2so4_condensation: bool = True
    h2so4_cond_rate_s_inv: float = 2.8e-4  # ~1 h timescale by default

    # --- internal, filled by write_common_files ---
    _file_list_path: Optional[Path] = None

    def write_common_files(self, scenario_dir: str) -> Path:
        """Create <scenario_dir>/camp/* CAMP files and return absolute path to camp_file_list.json."""
        camp_dir = Path(scenario_dir).resolve() / "camp"
        camp_dir.mkdir(parents=True, exist_ok=True)

        # 1) species.json – keep it minimal for now
        species = {
            "camp-data": [
                # GAS species used by the simple mechanism (only H2SO4 required here)
                {"type": "CHEM_SPEC", "name": "H2SO4", "phase": "GAS", "molecular weight": 0.098079},
                # If we later move to true phase-transfer, we’ll deposit into an aerosol species:
                {"type": "CHEM_SPEC", "name": "H2SO4_sink", "phase": "AEROSOL",
                 "molecular weight": 0.098079, "density": 1830.0},
            ]
        }
        (camp_dir / "species.json").write_text(json.dumps(species, indent=2))

        # 2) aero_phase.json – trivial; present to be future-proof if we upgrade to HL/SIMPOL
        aero_phase = {
            "camp-data": [
                {"type": "AERO_PHASE", "name": "default_phase", "species": ["H2SO4_sink"]}
            ]
        }
        (camp_dir / "aero_phase.json").write_text(json.dumps(aero_phase, indent=2))

        # 3) mechanism.json – simple H2SO4 first-order loss (gas -> sink)
        # CAMP Arrhenius input: reactants/products as mappings; A is s^-1 for unimolecular
        mech_reactions = []
        if self.enable_h2so4_condensation and self.h2so4_cond_rate_s_inv > 0.0:
            mech_reactions.append({
                "type": "ARRHENIUS",
                "A": float(self.h2so4_cond_rate_s_inv),  # s^-1
                "reactants": {"H2SO4": {}},
                # put the mass into an inert sink species to make the budget obvious
                "products": {"H2SO4_sink": {"yield": 1.0}}
            })
        mechanism = {"camp-data": [{"type": "MECHANISM", "name": "ambrs_mechanism",
                                    "reactions": mech_reactions}]}
        (camp_dir / "mechanism.json").write_text(json.dumps(mechanism, indent=2))

        # 4) file list – ABSOLUTE paths (fixes 'Cannot file file: species.json')
        file_list = {
            "camp-files": [
                str((camp_dir / "species.json").resolve()),
                str((camp_dir / "aero_phase.json").resolve()),
                str((camp_dir / "mechanism.json").resolve()),
            ]
        }
        file_list_path = (camp_dir / "camp_file_list.json")
        file_list_path.write_text(json.dumps(file_list, indent=2))

        self._file_list_path = file_list_path.resolve()
        return self._file_list_path

    def runtime_env(self) -> dict:
        """
        Environment vars to make the model find CAMP config and libraries.

        Returns env dict you can pass to subprocess.run(env=...).
        Always include CAMP_FILE_LIST when available; add DYLD/LD paths if lib_dir provided.
        """
        env = os.environ.copy()
        if self._file_list_path is not None:
            env["CAMP_FILE_LIST"] = str(self._file_list_path)
        if self.lib_dir:
            lib_path = str(Path(self.lib_dir).resolve())
            # macOS
            env["DYLD_LIBRARY_PATH"] = f"{lib_path}:{env.get('DYLD_LIBRARY_PATH','')}".rstrip(":")
            # Linux
            env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH','')}".rstrip(":")
        return env


# # ambrs/camp.py
# from __future__ import annotations
# from dataclasses import dataclass, field
# from pathlib import Path
# import json
# from typing import Iterable, Optional

# @dataclass
# class CampConfig:
#     """
#     Builds CAMP config files for a scenario and returns the path to the top-level
#     file-list JSON that CAMP expects.

#     Design:
#       - COMMON files (shared by MAM4 & PartMC): species, mechanism stub, solver.
#       - MODEL-specific override (optional): empty by default, but gives each model
#         its own hook without "crazy" config sprawl.
#       - A file-list JSON (`camp_files.json`) that lists all of the above in order.

#     You can pass an explicit list of gas species; otherwise we’ll generate a
#     minimal superset that works for the demo (SO2, H2SO4, and a placeholder SOAG).
#     """
#     mechanism_name: str = "ambrs_mech"
#     rel_tol: float = 1.0e-6
#     abs_tol_ppm: float = 0.0      # 0 lets CAMP choose defaults
#     gas_species: Optional[Iterable[str]] = None

#     # internal cache for last produced file-list (per directory/model)
#     _last_files: dict[tuple[Path, str], Path] = field(default_factory=dict, init=False, repr=False)

#     def _species_list(self, user_gases: Optional[Iterable[str]]) -> list[str]:
#         if user_gases:
#             return sorted({g for g in user_gases})
#         # conservative default superset that matches the demo & keeps CAMP happy
#         return ["SO2", "H2SO4", "SOAG"]

#     def write_common_files(self, camp_dir: Path, gases: Optional[Iterable[str]] = None) -> dict[str, Path]:
#         """
#         Writes common CAMP files (species, mechanism stub with no reactions,
#         and solver tolerances). Returns their paths.
#         """
#         camp_dir.mkdir(parents=True, exist_ok=True)

#         species = self._species_list(gases)
#         # 1) species.json
#         species_json = {
#             "camp-data": [
#                 {"type": "CHEM_SPEC", "name": s, "phase": "gas"} for s in species
#             ]
#         }

#         # 2) mechanism.json (stub: zero reactions -> identity chemistry; safe default)
#         mech_json = {
#             "camp-data": [
#                 {
#                     "type": "MECHANISM",
#                     "name": self.mechanism_name,
#                     "reactions": []  # You can add reactions later without changing ambrs.
#                 }
#             ]
#         }

#         # 3) solver.json (tolerances, OK if host model ignores some fields)
#         solver_json = {
#             "camp-data": [
#                 {"type": "SOLVER", "name": "default",
#                  "rel_tol": self.rel_tol,
#                  # CAMP uses ppm for gas-state by default; abs_tol array is optional
#                  "abs_tol": self.abs_tol_ppm}
#             ]
#         }

#         files = {
#             "species": camp_dir / "species.json",
#             "mechanism": camp_dir / "mechanism.json",
#             "solver": camp_dir / "solver.json",
#         }
#         with files["species"].open("w") as f:
#             json.dump(species_json, f, indent=2)
#         with files["mechanism"].open("w") as f:
#             json.dump(mech_json, f, indent=2)
#         with files["solver"].open("w") as f:
#             json.dump(solver_json, f, indent=2)

#         return files

#     def write_for_model(self, root_dir: Path, model_name: str, gases: Optional[Iterable[str]] = None) -> Path:
#         """
#         Creates (if needed) the full CAMP config for `model_name` under
#         <root_dir>/camp/, returning the path to 'camp_files.json' (file-list).

#         This file-list is what both PartMC and MAM4 pass to CAMP.
#         """
#         key = (root_dir.resolve(), model_name)
#         if key in self._last_files:
#             return self._last_files[key]

#         camp_dir = (root_dir / "camp")
#         common = self.write_common_files(camp_dir, gases)

#         # Optional model-specific override hook (empty; safe to include).
#         override = camp_dir / f"{model_name}_override.json"
#         if not override.exists():
#             with override.open("w") as f:
#                 json.dump({"camp-data": []}, f, indent=2)

#         # File-list JSON that CAMP expects: { "camp-files": [ ... ] }
#         file_list = camp_dir / "camp_files.json"
#         # file_list_json = {
#         #     "camp-files": [
#         #         str(common["species"].name),
#         #         str(common["mechanism"].name),
#         #         str(common["solver"].name),
#         #         str(override.name),
#         #     ]
#         # }
#         file_list_json = {
#             "camp-files": [
#                 str(common["species"].resolve()),
#                 str(common["mechanism"].resolve()),
#                 str(common["solver"].resolve()),
#                 str(override.resolve()),
#             ]
#         }
#         with file_list.open("w") as f:
#             json.dump(file_list_json, f, indent=2)

#         # cache and return a *relative* path usable in input files
#         self._last_files[key] = file_list
#         return file_list

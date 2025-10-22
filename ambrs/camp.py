# ambrs/camp.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, os, sys, glob
from typing import Dict, List, Optional, Tuple, Any

def _as_abs(p: Path) -> str:
    return str(p.resolve())

@dataclass
class MechanismComponent:
    """A general container for CAMP 'camp-data' items (species, phases, reactions)."""
    camp_data: List[dict] = field(default_factory=list)

    def extend(self, items: List[dict]) -> None:
        self.camp_data.extend(items)

    def to_json_obj(self) -> dict:
        return {"camp-data": self.camp_data}

@dataclass
class CampConfig:
    """
    General, mechanism-driven CAMP config builder.

    - Writes absolute-path 'camp_files.json'
    - Provides a generic SIMPOL phase-transfer helper (works for SOAG; can be tuned for H2SO4)
    - Adds first-order-loss (when desired) for gas-only sinks
    - Ensures all condensed species declare a molecular weight (fixes 'Missing molecular weight' error)
    - Supplies a robust runtime_env() with DYLD/LD paths
    """
    # mechanism name appears in MECHANISM objects
    mechanism_name: str = "ambrs_mech"

    # Library discovery hints (optional)
    explicit_lib_dirs: Tuple[Path, ...] = field(default_factory=tuple)  # e.g., (Path(os.environ["CONDA_PREFIX"])/"lib",)
    explicit_lib_files: Tuple[Path, ...] = field(default_factory=tuple) # e.g., (Path("/opt/somewhere/libcamp.1.1.0.dylib"),)

    # Species registry (so we can avoid duplicates as we add mechanisms)
    _species_gas: Dict[str, dict] = field(default_factory=dict, init=False)
    _species_aer: Dict[str, dict] = field(default_factory=dict, init=False)

    # Phase registry (keep one aerosol phase by default; host model maps bins/modes)
    aerosol_phase_name: str = "default_phase"

    def _discover_lib_dirs(self) -> List[Path]:
        candidates: List[Path] = []
        # 1) Hints from constructor
        candidates.extend(list(self.explicit_lib_dirs))
        # 2) Environment
        for key in ("CAMP_LIB_DIR", "LIBCAMP_DIR", "AMBRS_LIBCAMP_DIR"):
            if key in os.environ:
                candidates.append(Path(os.environ[key]))
        # 3) Conda env
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            candidates.append(Path(conda_prefix) / "lib")
        # 4) Common extras
        for p in (Path("/usr/local/lib"), Path("/usr/lib")):
            candidates.append(p)
        # De-dup & existing only
        uniq: List[Path] = []
        for c in candidates:
            if c and c not in uniq and c.exists():
                uniq.append(c)
        return uniq

    def _find_libcamp_dir(self) -> Optional[Path]:
        # Accept any libcamp*.so/.dylib in a candidate directory
        for d in self._discover_lib_dirs():
            hits = list(d.glob("libcamp*.dylib")) + list(d.glob("libcamp*.so"))
            if hits:
                # Prefer the most versioned-looking name
                hits.sort(key=lambda p: len(p.name), reverse=True)
                return d
        return None

    # -------- species / phases / reactions builders -------- #

    def add_gas_species(self, name: str, *, molecular_weight_kgmol: float,
                        diffusion_coeff_m2s: Optional[float] = None,
                        extra: Optional[dict] = None) -> None:
        if name in self._species_gas:  # already present
            return
        entry = {
            "name": name,
            "type": "CHEM_SPEC",
            "parameters": {
                "phase": "gas",
                "molecular weight [kg mol-1]": molecular_weight_kgmol,
            }
        }
        if diffusion_coeff_m2s is not None:
            entry["parameters"]["diffusion coeff [m2 s-1]"] = diffusion_coeff_m2s
        if extra:
            entry["parameters"].update(extra)
        self._species_gas[name] = entry

    def add_aerosol_species(self, name: str, *, molecular_weight_kgmol: float,
                            extra: Optional[dict] = None) -> None:
        if name in self._species_aer:
            return
        entry = {
            "name": name,
            "type": "CHEM_SPEC",
            "parameters": {
                "phase": "aerosol",
                "aerosol phase": self.aerosol_phase_name,
                "molecular weight [kg mol-1]": molecular_weight_kgmol,
            }
        }
        if extra:
            entry["parameters"].update(extra)
        self._species_aer[name] = entry

    def make_default_phases(self) -> List[dict]:
        # One gas phase is implicit via CHEM_SPEC entries; define our aerosol phase explicitly
        return [{
            "name": self.aerosol_phase_name,
            "type": "AEROSOL_PHASE",
            "parameters": {}  # host provides representation; we just name a phase
        }]

    def simpol_phase_transfer(self,
                              gas_species: str,
                              aerosol_species: str,
                              *,
                              B: Tuple[float, float, float, float],
                              N_star: Optional[float] = None,
                              activity_coeff_species: Optional[str] = None) -> dict:
        """A general SIMPOL.1 phase transfer (gas <-> aerosol) reaction."""
        rxn = {
            "name": f"{gas_species}_to_{aerosol_species}_simpol",
            "type": "SIMPOL_PHASE_TRANSFER",
            "reactants": {gas_species: {}},
            "products": {aerosol_species: {}},
            "parameters": {
                "gas-phase species": gas_species,
                "aerosol phase": self.aerosol_phase_name,
                "aerosol-phase species": aerosol_species,
                "B": list(B),
            }
        }
        if N_star is not None:
            rxn["parameters"]["N star"] = N_star
        if activity_coeff_species:
            rxn["parameters"]["aerosol-phase activity coefficient"] = activity_coeff_species
        return rxn

    def first_order_loss(self, gas_species: str, *, label: Optional[str] = None,
                         k_s_inv: float = 0.0) -> dict:
        """Gas-phase first-order loss (host can update rate at run time; constant default)."""
        return {
            "name": label or f"{gas_species}_first_order_loss",
            "type": "FIRST_ORDER_LOSS",
            "reactants": {gas_species: {}},
            "products": {},
            "parameters": {
                "rate constant [s-1]": float(k_s_inv)
            }
        }

    # -------- writers -------- #

    def write_for_model(self, run_dir: Path, model_name: str,
                        gases_present: Optional[List[str]] = None) -> Path:
        """
        Writes a small set of CAMP jsons under <run_dir>/camp/<model_name>/ and returns
        the path to camp_files.json (absolute path).
        """
        run_dir = Path(run_dir)
        out_dir = run_dir / "camp" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Species block (gas + aerosol)
        species_block = MechanismComponent()
        # H2SO4 + SOAG defaults (always safe to include; unused species are ignored by host)
        # Gas species (include diffusivity for SIMPOL)
        self.add_gas_species("H2SO4", molecular_weight_kgmol=98.079e-3,
                             diffusion_coeff_m2s=8.0e-6)
        self.add_gas_species("SOAG",  molecular_weight_kgmol=150.0e-3,
                             diffusion_coeff_m2s=8.0e-6)
        # Aerosol counterparts (must have molecular weight) – note we condense H2SO4 as H2SO4
        self.add_aerosol_species("H2SO4", molecular_weight_kgmol=98.079e-3)
        self.add_aerosol_species("SOAG",  molecular_weight_kgmol=150.0e-3)

        # Allow user-provided gas list to *prune* if desired (but keep required ones)
        # Here we just output all we’ve registered; CAMP ignores unused.
        species_block.extend(list(self._species_gas.values()))
        species_block.extend(list(self._species_aer.values()))

        # 2) Phases block
        phases_block = MechanismComponent(self.make_default_phases())

        # 3) Mechanism & reactions
        mech_block = MechanismComponent()
        mech_block.extend([{
            "name": self.mechanism_name,
            "type": "MECHANISM",
            "reactions": []
        }])

        reactions: List[dict] = []

        # ---- H2SO4 condensation (SIMPOL) ----
        # Use conservative SIMPOL B (zeros) + a modest N* to emulate low volatility.
        # This pushes flux from gas -> particle and is close to MAM’s simple H2SO4 sink.
        reactions.append(self.simpol_phase_transfer(
            gas_species="H2SO4",
            aerosol_species="H2SO4",
            B=(0.0, 0.0, 0.0, 0.0),
            N_star=1.25
        ))

        # ---- SOAG condensation (SIMPOL) ----
        reactions.append(self.simpol_phase_transfer(
            gas_species="SOAG",
            aerosol_species="SOAG",
            # canonical “semi-volatile” example (you can override in PPE)
            B=(0.0, -8.9, 0.0, 0.0),
            N_star=1.233
        ))

        # (Optional) You can also add a gas-only first-order loss by uncommenting:
        # reactions.append(self.first_order_loss("H2SO4", k_s_inv=0.0))

        # install reactions into the mechanism
        mech_block.camp_data[0]["reactions"] = reactions

        # 4) Write files (ABSOLUTE paths in camp_files.json)
        species_json = out_dir / "species.json"
        phases_json  = out_dir / "phases.json"
        mech_json    = out_dir / "mechanism.json"
        for pth, obj in ((species_json, species_block),
                         (phases_json,  phases_block),
                         (mech_json,    mech_block)):
            with open(pth, "w") as f:
                json.dump(obj.to_json_obj(), f, indent=2)

        camp_files = out_dir / "camp_files.json"
        with open(camp_files, "w") as f:
            json.dump({"camp-files": [_as_abs(species_json),
                                      _as_abs(phases_json),
                                      _as_abs(mech_json)]}, f, indent=2)

        return camp_files.resolve()

    # -------- runtime environment for the runner -------- #

    def runtime_env(self) -> Dict[str, str]:
        """
        Return env vars to pass to subprocess.run so CAMP’s dynamic lib is found
        and so the host always loads the right files.
        """
        env = dict(os.environ)  # copy current
        lib_dir = self._find_libcamp_dir()
        if lib_dir:
            lib_dir_s = str(lib_dir.resolve())
            if sys.platform == "darwin":
                # Be explicit on macOS
                env["DYLD_LIBRARY_PATH"] = f"{lib_dir_s}:{env.get('DYLD_LIBRARY_PATH','')}".rstrip(":")
                env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{lib_dir_s}:{env.get('DYLD_FALLBACK_LIBRARY_PATH','')}".rstrip(":")
            elif sys.platform.startswith("linux"):
                env["LD_LIBRARY_PATH"] = f"{lib_dir_s}:{env.get('LD_LIBRARY_PATH','')}".rstrip(":")
        return env

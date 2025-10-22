# test/test_integration_h2so4.py

# FIXME: move to separate test dir that is run only periodically?

# Integration tests for AMBRS × CAMP with a simple H2SO4 mechanism.
# These tests:
#   1) Run MAM4 baseline (native condensation) vs CAMP (simple H2SO4 sink).
#   2) Run PartMC baseline (MOSAIC) vs CAMP (simple H2SO4 sink).
#   3) Check CAMP ~ native (per model), and within-model conservation (gas + SO4).
#   4) Compare final aerosol SO4 between MAM4 and PartMC (rough parity).

import os
import math
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

import ambrs
from ambrs import mam4, partmc
from ambrs.aerosol import AerosolProcesses
from ambrs.ppe import Ensemble
from ambrs.runners import PoolRunner
from ambrs.camp import CampConfig  # General helper you integrated

# --------------------------
# Config & helpful defaults
# --------------------------

# Use PATH or override with env vars
MAM4_EXE = os.getenv("AMBRS_MAM4_EXE", "mam4")
PARTMC_EXE = os.getenv("AMBRS_PARTMC_EXE", "partmc")

# CAMP shared library directory (if needed)
LIBCAMP_DIR = os.getenv("AMBRS_CAMP_LIBDIR", None)

# Timescale for the simple H2SO4 sink (seconds)
H2SO4_TAU_S = float(os.getenv("AMBRS_H2SO4_TAU_S", "3600"))  # 1 hour
H2SO4_K = 1.0 / H2SO4_TAU_S

# Ensemble size (keep small for CI)
N_SCENARIOS = int(os.getenv("AMBRS_TEST_N_SCENARIOS", "1"))
DT = float(os.getenv("AMBRS_TEST_DT", "300.0"))     # 5 minutes
NSTEPS = int(os.getenv("AMBRS_TEST_NSTEPS", "8"))   # short run

# Tolerances (tune as needed)
RELERR_MAX = float(os.getenv("AMBRS_TEST_RELERR_MAX", "0.20"))           # 20% mean relative error
TOTAL_S_RELSPAN_MAX = float(os.getenv("AMBRS_TEST_TOTALS_RELSPAN_MAX", "0.10"))  # 10% variation within a run
XMODEL_SO4_RELERR_MAX = float(os.getenv("AMBRS_TEST_XMODEL_SO4_RELERR_MAX", "0.25"))  # 25% final SO4 disparity

pytestmark = pytest.mark.integration

# Skips if the model executables are not present/usable
def _which(exe: str) -> bool:
    from shutil import which
    return which(exe) is not None

skip_mam4 = pytest.mark.skipif(not _which(MAM4_EXE), reason=f"{MAM4_EXE} not found on PATH")
skip_partmc = pytest.mark.skipif(not _which(PARTMC_EXE), reason=f"{PARTMC_EXE} not found on PATH")

# --------------------------
# Output readers (best-effort)
# --------------------------

def _read_mam4_timeseries(run_root: Path, scenario_name: str):
    """Return {time, h2so4_gas, so4_aer} for a MAM4 run (best-effort variable names)."""
    nc_path = run_root / scenario_name / "mam_output.nc"
    if not nc_path.exists():
        return {"time": None, "h2so4_gas": None, "so4_aer": None}
    nc = Dataset(str(nc_path))
    out = {}
    out["time"] = np.array(nc.variables["time"][:]) if "time" in nc.variables else None

    # Gas H2SO4
    if "h2so4_gas" in nc.variables:
        out["h2so4_gas"] = np.array(nc.variables["h2so4_gas"][:])
    else:
        cand = [v for v in nc.variables if "h2so4" in v.lower() and "gas" in v.lower()]
        out["h2so4_gas"] = np.array(nc.variables[cand[0]][:]) if cand else None

    # Aerosol sulfate: sum any SO4* aerosol vars
    so4 = None
    for v in nc.variables:
        vl = v.lower()
        if "so4" in vl and ("aer" in vl or "mode" in vl or "pm" in vl) and "gas" not in vl:
            arr = np.array(nc.variables[v][:])
            so4 = arr if so4 is None else so4 + arr
    out["so4_aer"] = so4
    return out

def _read_partmc_timeseries(run_root: Path, scenario_name: str):
    """Return {time, h2so4_gas, so4_aer} for a PartMC run.
       Gas H2SO4 is read from 'gas_mixing_ratio' with names in 'gas_species'.
       Aerosol SO4 is optional (depends on outputs)."""
    outdir = run_root / scenario_name / "out"
    if not outdir.exists():
        return {"time": None, "h2so4_gas": None, "so4_aer": None}
    files = sorted([f for f in os.listdir(outdir) if f.endswith(".nc")])
    if not files:
        return {"time": None, "h2so4_gas": None, "so4_aer": None}
    times, h2so4_series = [], []
    so4_series = None
    for f in files:
        nc = Dataset(str(outdir / f))
        # time from filename suffix or variable
        try:
            tstep = int(f.split("_")[-1].split(".")[0])
        except Exception:
            tstep = len(times)
        times.append(tstep)

        # Gas H2SO4 (usually units of ppb)
        if "gas_species" in nc.variables and "gas_mixing_ratio" in nc.variables:
            names = nc.variables["gas_species"].names.split(",")
            data = np.array(nc.variables["gas_mixing_ratio"][:])  # [nspecies]
            if "H2SO4" in names:
                idx = names.index("H2SO4")
                h2so4_series.append(data[idx])
            else:
                h2so4_series.append(np.nan)

        # Aerosol SO4 proxy (if present)
        if so4_series is None:
            so4_series = []
        found_so4 = False
        for v in nc.variables:
            vl = v.lower()
            if "so4" in vl and ("aer" in vl or "mass" in vl or "pm" in vl) and "gas" not in vl:
                arr = np.array(nc.variables[v][:])
                # reduce to a single value per file if array
                val = float(np.atleast_1d(arr)[-1]) if arr.size else np.nan
                so4_series.append(val)
                found_so4 = True
                break
        if not found_so4:
            so4_series.append(np.nan)

    return {
        "time": np.array(times),
        "h2so4_gas": np.array(h2so4_series) if h2so4_series else None,
        "so4_aer": np.array(so4_series) if so4_series is not None else None,
    }

def _rel_err(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return np.abs(b - a) / np.maximum(1e-30, np.abs(a))

# --------------------------
# CAMP mechanism source (general)
# --------------------------

def _camp_h2so4_template():
    """General CAMP mechanism source using the built-in H2SO4 sink template."""
    return CampConfig(
        lib_dir=LIBCAMP_DIR,
        source={"template": {"name": "h2so4_sink", "rate_s_inv": H2SO4_K}},
    )

# --------------------------
# Core runners used by tests
# --------------------------

def _build_small_ensemble(n=N_SCENARIOS, seed=42):
    # Replace with your PPE if desired: Ensemble.from_config(...)
    return Ensemble.default(n=n, seed=seed)

@skip_mam4
def _run_mam4(tmp: Path, ensemble, use_camp: bool):
    root = tmp / ("mam4_camp" if use_camp else "mam4_base")
    root.mkdir(parents=True, exist_ok=True)

    processes = AerosolProcesses(
        gas_phase_chemistry=True,
        condensation=(not use_camp),  # disable native condensation if using CAMP
        coagulation=True,
        nucleation=False,
        optics=False,
    )

    camp_cfg = _camp_h2so4_template() if use_camp else None
    model = mam4.AerosolModel(processes=processes, camp=camp_cfg)

    inputs = [model.create_input(s, DT, NSTEPS) for s in ensemble.scenarios]

    # Make sure per-scenario camp/* exists before runner (nice-to-have)
    if use_camp:
        width = int(math.log10(len(ensemble.scenarios))+1) if len(ensemble.scenarios) > 1 else 1
        for i, s in enumerate(ensemble.scenarios):
            scen_dir = root / f"{i+1:0{width}d}"
            scen_dir.mkdir(parents=True, exist_ok=True)
            model.camp.write_common_files(str(scen_dir), scenario=s)

    PoolRunner(model, MAM4_EXE, str(root), num_processes=1).run(inputs)
    return root

@skip_partmc
def _run_partmc(tmp: Path, ensemble, use_camp: bool):
    root = tmp / ("partmc_camp" if use_camp else "partmc_base")
    root.mkdir(parents=True, exist_ok=True)

    processes = AerosolProcesses(
        gas_phase_chemistry=False,
        condensation=(not use_camp),  # PartMC: 'condensation' flag means MOSAIC pathway here
        coagulation=True,
        nucleation=False,
        optics=False,
    )

    camp_cfg = _camp_h2so4_template() if use_camp else None
    model = partmc.AerosolModel(
        processes=processes,
        camp=camp_cfg,
        run_type="particle",
        n_part=20000,
        n_repeat=1,
    )

    inputs = [model.create_input(s, DT, NSTEPS) for s in ensemble.scenarios]

    if use_camp:
        width = int(math.log10(len(ensemble.scenarios))+1) if len(ensemble.scenarios) > 1 else 1
        for i, s in enumerate(ensemble.scenarios):
            scen_dir = root / f"{i+1:0{width}d}"
            scen_dir.mkdir(parents=True, exist_ok=True)
            model.camp.write_common_files(str(scen_dir), scenario=s)

    PoolRunner(model, PARTMC_EXE, str(root), num_processes=1).run(inputs)
    return root

# --------------------------
# Tests
# --------------------------

@pytest.mark.order(1)
@skip_mam4
def test_mam4_camp_matches_native(tmp_path):
    """MAM4 baseline vs CAMP (H2SO4 sink): H2SO4 and SO4 close; total S roughly conserved."""
    ensemble = _build_small_ensemble()
    base_root = _run_mam4(tmp_path, ensemble, use_camp=False)
    camp_root = _run_mam4(tmp_path, ensemble, use_camp=True)

    width = int(math.log10(len(ensemble.scenarios))+1) if len(ensemble.scenarios) > 1 else 1
    for i in range(len(ensemble.scenarios)):
        name = f"{i+1:0{width}d}"
        b = _read_mam4_timeseries(base_root, name)
        c = _read_mam4_timeseries(camp_root, name)

        # H2SO4 similarity
        assert b["h2so4_gas"] is not None and c["h2so4_gas"] is not None, "Missing H2SO4 gas in outputs"
        n = min(len(b["h2so4_gas"]), len(c["h2so4_gas"]))
        mean_rel = float(np.nanmean(_rel_err(b["h2so4_gas"][:n], c["h2so4_gas"][:n])))
        assert mean_rel <= RELERR_MAX, f"MAM4 H2SO4 rel err {mean_rel:.3f} > {RELERR_MAX}"

        # SO4 similarity if available
        if b["so4_aer"] is not None and c["so4_aer"] is not None:
            n2 = min(len(b["so4_aer"]), len(c["so4_aer"]))
            mean_rel_so4 = float(np.nanmean(_rel_err(b["so4_aer"][:n2], c["so4_aer"][:n2])))
            assert mean_rel_so4 <= max(RELERR_MAX, 0.30), f"MAM4 SO4 rel err {mean_rel_so4:.3f} too large"

        # Total sulfur conservation proxy (within each run)
        if b["so4_aer"] is not None:
            nb = min(n, len(b["so4_aer"]))
            tot_b = np.asarray(b["h2so4_gas"][:nb]) + np.asarray(b["so4_aer"][:nb])
            span_b = float(np.nanmax(tot_b) - np.nanmin(tot_b))
            denom_b = max(1e-30, float(abs(tot_b[0])))
            assert span_b/denom_b <= TOTAL_S_RELSPAN_MAX, f"MAM4 baseline total S span too large: {span_b/denom_b:.3f}"
        if c["so4_aer"] is not None:
            nc_ = min(n, len(c["so4_aer"]))
            tot_c = np.asarray(c["h2so4_gas"][:nc_]) + np.asarray(c["so4_aer"][:nc_])
            span_c = float(np.nanmax(tot_c) - np.nanmin(tot_c))
            denom_c = max(1e-30, float(abs(tot_c[0])))
            assert span_c/denom_c <= TOTAL_S_RELSPAN_MAX, f"MAM4 CAMP total S span too large: {span_c/denom_c:.3f}"

@pytest.mark.order(2)
@skip_partmc
def test_partmc_camp_matches_native(tmp_path):
    """PartMC baseline vs CAMP (H2SO4 sink): H2SO4 and SO4 close; total S roughly conserved (if comparable)."""
    ensemble = _build_small_ensemble()
    base_root = _run_partmc(tmp_path, ensemble, use_camp=False)
    camp_root = _run_partmc(tmp_path, ensemble, use_camp=True)

    width = int(math.log10(len(ensemble.scenarios))+1) if len(ensemble.scenarios) > 1 else 1
    for i in range(len(ensemble.scenarios)):
        name = f"{i+1:0{width}d}"
        b = _read_partmc_timeseries(base_root, name)
        c = _read_partmc_timeseries(camp_root, name)

        # H2SO4 similarity (units within-model are consistent; cross-model not required)
        assert b["h2so4_gas"] is not None and c["h2so4_gas"] is not None, "Missing H2SO4 gas in PartMC outputs"
        n = min(len(b["h2so4_gas"]), len(c["h2so4_gas"]))
        mean_rel = float(np.nanmean(_rel_err(b["h2so4_gas"][:n], c["h2so4_gas"][:n])))
        assert mean_rel <= RELERR_MAX, f"PartMC H2SO4 rel err {mean_rel:.3f} > {RELERR_MAX}"

        # SO4 similarity if available
        if b["so4_aer"] is not None and c["so4_aer"] is not None:
            n2 = min(len(b["so4_aer"]), len(c["so4_aer"]))
            mean_rel_so4 = float(np.nanmean(_rel_err(b["so4_aer"][:n2], c["so4_aer"][:n2])))
            assert mean_rel_so4 <= max(RELERR_MAX, 0.35), f"PartMC SO4 rel err {mean_rel_so4:.3f} too large"

        # Total sulfur conservation proxy (within each run) — only if units are comparable within model
        if b["so4_aer"] is not None and np.isfinite(b["so4_aer"]).any():
            nb = min(n, len(b["so4_aer"]))
            tot_b = np.asarray(b["h2so4_gas"][:nb]) + np.asarray(b["so4_aer"][:nb])
            span_b = float(np.nanmax(tot_b) - np.nanmin(tot_b))
            denom_b = max(1e-30, float(abs(tot_b[0])))
            assert span_b/denom_b <= max(TOTAL_S_RELSPAN_MAX, 0.20), f"PartMC baseline total S span too large: {span_b/denom_b:.3f}"
        if c["so4_aer"] is not None and np.isfinite(c["so4_aer"]).any():
            nc_ = min(n, len(c["so4_aer"]))
            tot_c = np.asarray(c["h2so4_gas"][:nc_]) + np.asarray(c["so4_aer"][:nc_])
            span_c = float(np.nanmax(tot_c) - np.nanmin(tot_c))
            denom_c = max(1e-30, float(abs(tot_c[0])))
            assert span_c/denom_c <= max(TOTAL_S_RELSPAN_MAX, 0.20), f"PartMC CAMP total S span too large: {span_c/denom_c:.3f}"

@pytest.mark.order(3)
@skip_mam4
@skip_partmc
def test_cross_model_so4_similarity(tmp_path):
    """Compare final aerosol SO4 between MAM4 and PartMC on the CAMP runs (rough parity)."""
    ensemble = _build_small_ensemble()
    # Run only CAMP cases; we don't need baselines here
    mam4_root = _run_mam4(tmp_path, ensemble, use_camp=True)
    pm_root = _run_partmc(tmp_path, ensemble, use_camp=True)

    width = int(math.log10(len(ensemble.scenarios))+1) if len(ensemble.scenarios) > 1 else 1
    for i in range(len(ensemble.scenarios)):
        name = f"{i+1:0{width}d}"
        m = _read_mam4_timeseries(mam4_root, name)
        p = _read_partmc_timeseries(pm_root, name)

        if m["so4_aer"] is None or p["so4_aer"] is None:
            pytest.skip("SO4 aerosol not present in one of the outputs; skipping cross-model SO4 check")

        m_final = float(np.asarray(m["so4_aer"])[-1])
        p_final = float(np.asarray(p["so4_aer"])[-1])

        denom = max(1e-30, abs(m_final))
        rel = abs(p_final - m_final) / denom
        assert rel <= XMODEL_SO4_RELERR_MAX, f"Cross-model final SO4 rel diff {rel:.3f} > {XMODEL_SO4_RELERR_MAX}"

#!/usr/bin/env bash
set -euo pipefail

# where to create the demo (run this from the ROOT of your ambrs repo)
DEMO_DIR="ambrs/demos/camp_h2so4_cond"
mkdir -p "$DEMO_DIR"

# -------------------------------
# README
# -------------------------------
cat > "$DEMO_DIR/README.md" << 'EOF'
# CAMP H2SO4 Condensation – Side-by-side MAM4 & PartMC (AMBRS demo)

This demo creates **two runs** using **CAMP** chemistry:
- `mam4` (MAM4 box model)
- `partmc` (PartMC box model)

It sets consistent initial conditions and writes the **CAMP `namelist`** file
that CAMP expects at runtime in each run directory.

> Works with AMBRS branch: `analysis_clean`

---

## 0) Environment (macOS / Apple Silicon shown)

```bash
# executables
export PATH="$CONDA_PREFIX/bin:$PATH"
# shared libs (macOS)
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"
# (Linux) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

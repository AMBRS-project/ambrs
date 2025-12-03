"""
AMBRS Ensemble Inputs: Figure Table with Ranges and Strip Plots (Linear Scale)
- Table columns: variable name | scale (log/lin) | min | max | strip plot (no text/labels)
- All strip plots on linear scale for visual comparability; no axis, ticks, or text.
- Clean, publication-ready layout.

Usage: Run after ensemble creation in demo.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- Collect sampled data ---
records = []
scale_info = {}

for i, sample in enumerate(ensemble):
    for mode_idx, mode in enumerate(sample.size.modes):
        mode_base = mode.name.replace(" ", "_")
        records.append({"sample": i, "variable": f"{mode_base}_number", "value": mode.number})
        scale_info[f"{mode_base}_number"] = "log"
        records.append({"sample": i, "variable": f"{mode_base}_geom_mean_diam", "value": mode.geom_mean_diam})
        scale_info[f"{mode_base}_geom_mean_diam"] = "log"
        records.append({"sample": i, "variable": f"{mode_base}_log10_geom_std_dev", "value": mode.log10_geom_std_dev})
        scale_info[f"{mode_base}_log10_geom_std_dev"] = "lin"
        for sp_idx, mf in enumerate(mode.mass_fractions):
            records.append({"sample": i, "variable": f"{mode_base}_mass_fraction_sp{sp_idx}", "value": mf})
            scale_info[f"{mode_base}_mass_fraction_sp{sp_idx}"] = "lin"
    for g_idx, gc in enumerate(sample.gas_concs):
        records.append({"sample": i, "variable": f"gas_conc_{g_idx}", "value": gc})
        scale_info[f"gas_conc_{g_idx}"] = "lin"
    records.append({"sample": i, "variable": "flux", "value": sample.flux})
    scale_info["flux"] = "log"
    records.append({"sample": i, "variable": "relative_humidity", "value": sample.relative_humidity})
    scale_info["relative_humidity"] = "lin"
    records.append({"sample": i, "variable": "temperature", "value": sample.temperature})
    scale_info["temperature"] = "lin"

df = pd.DataFrame(records)

# --- Select variables for table ---
variables_to_plot = [
    "accumulation_number",
    "accumulation_geom_mean_diam",
    "aitken_number",
    "aitken_geom_mean_diam",
    "coarse_number",
    "coarse_geom_mean_diam",
    "primary_carbon_number",
    "primary_carbon_geom_mean_diam",
    "gas_conc_0",
    "gas_conc_1",
    "flux",
    "relative_humidity",
    "temperature",
]

df_plot = df[df["variable"].isin(variables_to_plot)]

# --- Figure table layout ---
n_vars = len(variables_to_plot)
fig_height = 0.85 * n_vars + 1.2
fig = plt.figure(figsize=(11, fig_height))
gs = GridSpec(n_vars+1, 5, width_ratios=[2.7, 1.1, 1.7, 1.7, 6.5], height_ratios=[0.6]+[1]*n_vars)
fig.patch.set_facecolor('white')

# Table header
ax_title = fig.add_subplot(gs[0,:])
ax_title.axis('off')
ax_title.text(0.03, 0.5, "Variable", ha="left", va="center", fontsize=14, fontweight="bold")
ax_title.text(0.24, 0.5, "Scale", ha="left", va="center", fontsize=13, fontweight="bold")
ax_title.text(0.37, 0.5, "Min", ha="left", va="center", fontsize=13, fontweight="bold")
ax_title.text(0.50, 0.5, "Max", ha="left", va="center", fontsize=13, fontweight="bold")
ax_title.text(0.78, 0.5, "Ensemble Samples", ha="left", va="center", fontsize=14, fontweight="bold")

for i, var in enumerate(variables_to_plot):
    vals = df_plot[df_plot["variable"] == var]["value"].values

    # --- Variable name cell
    ax0 = fig.add_subplot(gs[i+1, 0])
    ax0.axis('off')
    ax0.text(1.0, 0.5, var.replace("_", " ").title(), ha="right", va="center", fontsize=13, fontweight="bold")

    # --- Scale cell
    ax1 = fig.add_subplot(gs[i+1, 1])
    ax1.axis('off')
    ax1.text(0.5, 0.5, scale_info[var], ha="center", va="center", fontsize=12, color="#444")

    # --- Min cell
    ax2 = fig.add_subplot(gs[i+1, 2])
    ax2.axis('off')
    ax2.text(0.5, 0.5, f"{np.min(vals):.2g}", ha="center", va="center", fontsize=12)

    # --- Max cell
    ax3 = fig.add_subplot(gs[i+1, 3])
    ax3.axis('off')
    ax3.text(0.5, 0.5, f"{np.max(vals):.2g}", ha="center", va="center", fontsize=12)

    # --- Strip plot cell (no text, no axes)
    ax4 = fig.add_subplot(gs[i+1, 4])
    ax4.scatter(vals, np.ones_like(vals), s=60, facecolor="white", edgecolor="black", alpha=0.85)
    ax4.set_yticks([])
    ax4.set_xticks([])
    for spine in ['top','right','left','bottom']:
        ax4.spines[spine].set_visible(False)
    ax4.set_ylabel("")
    ax4.set_xlabel("")
    ax4.set_xlim(np.min(vals), np.max(vals))

fig.suptitle("AMBRS Ensemble Inputs: Table of Ranges and Samples", fontsize=16, fontweight="bold", x=0.55, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
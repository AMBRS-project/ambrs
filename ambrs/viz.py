import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from typing import Sequence, Tuple, Dict, Any

from part2pop.viz.style import StyleManager, Theme
from part2pop.viz.builder import build_plotter
from .ppe import Ensemble

from ambrs import partmc, mam4 # import modules

# -----------------------------------------------------------
# Styling helpers (row color, solid vs dashed)
# -----------------------------------------------------------
def _scenario_colors(n_rows: int, palette=None):
    if palette is not None:
        return palette[:n_rows]
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n_rows)]

def _base_styles():
    # Start from your theme so font/rcParams stay consistent
    mgr = StyleManager(Theme(), deterministic=False)
    return mgr.plan("line", ["partmc", "mam4"])

def _row_styles(base_styles: Dict[str, Dict[str, Any]], color) -> Dict[str, Dict[str, Any]]:
    # PartMC: solid; MAM4: dashed + thicker; both share row color
    return {
        "partmc": {**base_styles["partmc"], "color": color, "linestyle": "-",  "linewidth": 2.0},
        "mam4":   {**base_styles["mam4"],   "color": color, "linestyle": "--", "linewidth": 3.0},
    }
    
def _format_panel(ax, *, xscale=None, yscale=None, minimal_spines=True):
    if xscale:
        ax.set_xscale(xscale)
    
    if yscale:
        ax.set_yscale(yscale)
    
    if minimal_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

def _add_row_label(ax, label: str, color: str = "black"):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.text(
        xlims[1] - 0.2*(xlims[1]-xlims[0]),
        0.95*(ylims[0]+ylims[1]),
        label,
        # rotation=90,
        ha='left',
        va='center',
        fontsize=12,
        transform=ax.transData,
        color=color
    )

# -----------------------------------------------------------
# Helper functions for variable cfgs
# -----------------------------------------------------------

# fixme: add other optional argments?
def make_dNdlnD_cfg(D_range=(1e-9, 1e-6), N_bins=50, normalize=True, method="kde"):
    D_min, D_max = D_range
    D_grid = np.logspace(np.log10(D_min), np.log10(D_max), int(N_bins))
    return {"D": D_grid, "normalize": normalize, "method": method}

# fixme: add other optional argments?
def make_frac_ccn_cfg(s_grid=np.logspace(-2, 1.0, 50)):
    return {"s_grid": s_grid}

# fixme: add other optional argments?
def make_bscat_cfg(wvl_grid=np.linspace(0.35e-6, 0.8e-6, 30), rh_grid=[0.0]):
    wvl_grid = np.asarray(wvl_grid)
    rh_grid = np.asarray(rh_grid)
    return ({"wvl_grid": wvl_grid, "rh_grid": rh_grid})

# -----------------------------------------------------------
# Functions for rendering grids of PartMC vs MAM4 comparisons
# -----------------------------------------------------------
def render_partmc_and_mam4_variable_grid(
    gs, # GridSpec
    *,
    varname: str,                     # e.g., "dNdlnD", "frac_ccn", "b_scat"
    var_cfg: Dict,                    # e.g., make_dNdlnD_cfg(...)
    ensemble: Ensemble,
    timesteps: Sequence[int],
    partmc_dir: str,
    mam4_dir: str,
    scenario_names: Sequence[str] | None = None,
    legend_loc: str | None = None,
    row_colors: Sequence[str] | None = None,
    xscale: str = 'linear',
    yscale: str = 'linear',
    species_modifications: Dict = {}, # modify aerosol species during post-processing (e.g., assume BrC rather than non-absorbing OC)
    sharex: bool = True,
    sharey: bool = False,
    color: str | Sequence[str] | None = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Render a grid where rows = scenarios and columns = user-defined 'columns'
    (e.g., timesteps). Draws PartMC vs MAM4 overlays per cell.

    Returns
    -------
    fig, axes : (matplotlib.figure.Figure, np.ndarray[(n_rows, n_cols)])
    """
    
    if scenario_names == None:
        # FIXME: move this to utils module to avoid duplication
        num_scenarios = ensemble.__len__()
        max_num_digits = math.floor(math.log10(num_scenarios)) + 1
        # zero-pad the 1-based scenario index
        scenario_names = []
        for i in range(num_scenarios):
            num_digits = math.floor(math.log10(i+1)) + 1
            formatted_index = '0' * (max_num_digits - num_digits) + f'{i+1}'
            scenario_name = str(i + 1).format(index = formatted_index)
            scenario_names.append(scenario_name)
    
    fig = gs.figure
    n_rows, n_cols = len(scenario_names), len(timesteps)
    axes = np.empty((n_rows, n_cols), dtype=object)
    
    base = _base_styles()
    if color == None:
        colors = _scenario_colors(n_rows) if row_colors is None else list(row_colors)
    elif isinstance(color, str):
        colors = [color]*n_rows
    else:
        colors = color
    
    
    # else:
    #     scenario_names = scenario_names
    
    for i_row, scenario_name in enumerate(scenario_names):
        row_style = _row_styles(base, colors[i_row])

        for i_col, timestep in enumerate(timesteps):
            ax = fig.add_subplot(
                gs[i_row, i_col], 
                sharex=axes[i_row-1,i_col] if sharex and i_row > 0 else None,
                sharey=axes[i_row-1,i_col] if sharey and i_row > 0 else None)
            axes[i_row, i_col] = ax
            
            partmc_output = partmc.retrieve_model_state(
                scenario_name=scenario_name,
                scenario=ensemble.member(int(scenario_name)-1),
                timestep=timestep,
                repeat_num=1, # fixme: hardcoded for single repeat
                species_modifications=species_modifications,
                ensemble_output_dir=partmc_dir,
            )
            mam4_output = mam4.retrieve_model_state(
                scenario_name=scenario_name,
                scenario=ensemble.member(int(scenario_name)-1),
                timestep=timestep,
                species_modifications=species_modifications,
                ensemble_output_dir=mam4_dir,
            )

            series = (
                ("partmc", partmc_output.particle_population, "PartMC"),
                ("mam4",   mam4_output.particle_population,   "MAM4"),
            )

            # Plot both series
            for key, population, label in series:
                cfg = {"varname": varname, "var_cfg": var_cfg, "style": row_style[key]}
                build_plotter("state_line", cfg).plot(population, ax, add_xlabel=False, add_ylabel=False, label=label)
            
            # panel cosmetics
            _format_panel(ax, xscale=xscale, yscale=yscale)

            if varname == 'b_scat':
                ax.set_ylim([0., ax.get_ylim()[1]])
            # # column titles (top row only)
            # if i == 0 and col_title:
            #     ax.set_title(col_title)

            # legend (only once)
            if legend_loc == 'upper right' and i_row == 0 and i_col == n_cols - 1:
                ax.legend(frameon=False, loc=legend_loc)
            elif legend_loc == 'upper left' and i_row == 0 and i_col == 0:
                ax.legend(frameon=False, loc=legend_loc)
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if varname == 'dNdlnD':
                xticks = ax.get_xticks()
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{x*1e6:.2f}" for x in xticks])  # convert to µm
            elif varname == 'b_scat':
                xticks = ax.get_xticks()
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{x*1e9:.1f}" for x in xticks])  # convert to nm

                yticks = ax.get_yticks()
                ax.set_yticks(yticks)
                ax.set_yticklabels([f"{y*1e6:.1f}" for y in yticks]) # convert to $M$m^-1
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    
    if varname == 'dNdlnD':
        yvarlab = 'normalized number density'
        xvarlab = r'diameter [$\mu$m]'
    elif varname == 'frac_ccn':
        yvarlab = 'CCN activation fraction'
        xvarlab = 'supersaturation [%]'
    elif varname == 'b_scat':
        yvarlab = 'scattering coefficient (Mm$^{-1}$)'
        xvarlab = 'wavelength [nm]'
    else:
        yvarlab = varname.replace("_", " ")
    axes[np.floor(n_rows/2).astype(int), 0].set_ylabel(yvarlab)
    
    for i_col in range(n_cols):
        axes[-1, i_col].set_xlabel(xvarlab)

    return fig, axes

def render_dNdlnD_grid(
    gs, *, ensemble, scenario_names, timesteps, partmc_dir, mam4_dir,
    D_range=(1e-9, 1e-6), N_bins=50, normalize=True, method="kde",
    legend_loc=None, row_colors=None,
    xscale='log', yscale='linear',
    spec_modifications={},
    sharex=True, sharey=False,
    color=None):
    """
    Render grid of bscat vs wavelength at specified RH values.
    """
    return render_partmc_and_mam4_variable_grid(
        gs, # GridSpec
        varname = 'dNdlnD',
        var_cfg = make_dNdlnD_cfg(
            D_range=D_range, N_bins=N_bins, normalize=normalize, method=method),
        ensemble = ensemble,
        scenario_names = scenario_names,
        timesteps = timesteps,
        partmc_dir = partmc_dir,
        mam4_dir = mam4_dir,
        legend_loc = legend_loc,
        row_colors = row_colors,
        xscale = xscale,
        yscale = yscale,
        species_modifications = spec_modifications, # modify aerosol species during post-processing (e.g., assume BrC rather than non-absorbing OC)
        sharex = sharex,
        sharey = sharey,
        color = color,
        )

def render_frac_ccn_grid(
    gs, *, ensemble, scenario_names, timesteps, partmc_dir, mam4_dir,
    s_grid=np.logspace(-2, 1.0, 50),
    legend_loc=None, row_colors=None,
    xscale='log', yscale='linear',
    spec_modifications={}):
    """
    Render grid of bscat vs wavelength at specified RH values.
    """
    return render_partmc_and_mam4_variable_grid(
        gs, # GridSpec
        varname = 'frac_ccn',
        var_cfg = make_frac_ccn_cfg(s_grid=s_grid),
        ensemble = ensemble,
        scenario_names = scenario_names,
        timesteps = timesteps,
        partmc_dir = partmc_dir,
        mam4_dir = mam4_dir,
        legend_loc = legend_loc,
        row_colors = row_colors,
        xscale = xscale,
        yscale = yscale,
        species_modifications = spec_modifications, # modify aerosol species during post-processing (e.g., assume BrC rather than non-absorbing OC)
        )

def render_bscat_grid(
    gs, *, ensemble, scenario_names, timesteps, partmc_dir, mam4_dir,
    wvl_grid=np.linspace(0.35e-6, 0.8e-6, 30), rh_grid=[0.0],
    legend_loc='upper right', row_colors=None,
    xscale='linear', yscale='linear',
    spec_modifications={}):
    """
    Render grid of bscat vs wavelength at specified RH values.
    """
    return render_partmc_and_mam4_variable_grid(
        gs, # GridSpec
        varname = 'b_scat',
        var_cfg = make_bscat_cfg(wvl_grid=wvl_grid, rh_grid=rh_grid),
        ensemble = ensemble,
        scenario_names = scenario_names,
        timesteps = timesteps,
        partmc_dir = partmc_dir,
        mam4_dir = mam4_dir,
        legend_loc = legend_loc,
        row_colors = row_colors,
        xscale = xscale,
        yscale = yscale,
        species_modifications = spec_modifications, # modify aerosol species during post-processing (e.g., assume BrC rather than non-absorbing OC)
        )


# ---------------------------------------------------------------------------
# PLOT INPUT RANGES
# ---------------------------------------------------------------------------
def plot_range_bars(
    df, variables, var_col='variable', value_col='value',
    scale_info=None, highlight_idx=None, highlight_colors=None,
    figsize=(8, None)
):
    """
    Plot variable ranges as horizontal bars with strip plot overlay.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [var_col, value_col, 'sample'].
    variables : list
        Variables to plot in order.
    var_col : str
        Column name for variable identifiers.
    value_col : str
        Column name for numeric values.
    scale_info : dict, optional
        Mapping {var: "log"/"lin"} for axis scaling.
    highlight_idx : list of int, optional
        Indices in df['sample'] to highlight.
    highlight_colors : list or dict, optional
        If list: colors for each highlighted index in order.
        If dict: mapping {sample_index: color}.
    figsize : tuple
        Width, height of figure. Height auto-scales if None.
    """
    n = len(variables)
    fig_h = max(3, 0.45*n + 0.8)
    if figsize[1] is None:
        figsize = (figsize[0], fig_h)
    fig, axs = plt.subplots(
        n, 1, figsize=figsize, 
        gridspec_kw={'height_ratios':[1]*n}
    )
    if n == 1:
        axs = [axs]
    
    for ax, var in zip(axs, variables):
        vals = df[df[var_col]==var][value_col].values
        samples = df[df[var_col]==var]['sample'].values
        if len(vals) == 0:
            ax.text(0.5, 0.5, 'No samples',
                    ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off'); continue
        
        vmin, vmax = vals.min(), vals.max()
        pad = 0.02*(vmax-vmin) if vmin!=vmax else 0.1

        # --- range bar
        ax.hlines(0.5, vmin, vmax, linewidth=3, color='0.4')

        # --- strip of all samples
        # default background sample dots are grey so highlighted scenarios
        # (drawn later) stand out visually
        ax.scatter(vals, np.full_like(vals, 0.5),
                   s=40, facecolor="0.6", edgecolor="white", lw=0.7, alpha=0.6, zorder=2)

        # --- highlight selected scenarios
        if highlight_idx is not None:
            if isinstance(highlight_colors, dict):
                for idx in highlight_idx:
                    color = highlight_colors.get(idx, "red")
                    mask = samples == idx
                    ax.scatter(vals[mask], np.full(mask.sum(), 0.5),
                               s=100, color=color, edgecolor='k', zorder=3)
            else:
                # assume list aligned with highlight_idx
                colors = highlight_colors or ["red"]*len(highlight_idx)
                for idx, color in zip(highlight_idx, colors):
                    mask = samples == idx
                    ax.scatter(vals[mask], np.full(mask.sum(), 0.5),
                               s=100, color=color, edgecolor='k', zorder=3)
        
        # --- axis scaling
        if scale_info and scale_info.get(var, "lin") == "log":
            ax.set_xscale("log")
            pad = 0  # no pad for log, avoid negative limits
            ax.set_xlim(vmin*(0.9 if vmin>0 else 1), vmax*1.1)
        else:
            ax.set_xlim(vmin - pad, vmax + pad)

        # --- clean formatting
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(vmin, -0.18, f"{vmin:.2g}", transform=ax.get_xaxis_transform(),
                ha='left', va='top', fontsize=9)
        ax.text(vmax, -0.18, f"{vmax:.2g}", transform=ax.get_xaxis_transform(),
                ha='right', va='top', fontsize=9)
        ax.set_ylabel(var.replace("_", " ").title(),
                      rotation=0, ha="right", va="center", fontsize=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig



def build_input_ranges_dataframe(
    ensemble,
    *,
    variables=None,
    gas_names=None,
    sample_ids=None,
):
    """
    Create a long-form DataFrame with columns [variable, value, sample]
    directly from an Ensemble produced by ambrs.lhs(...).

    Parameters
    ----------
    ensemble : object
        Must expose per-sample arrays like ensemble.temperature, ensemble.relative_humidity, etc.
    variables : dict[str, str] or None
        Mapping {label -> attribute_name} to extract from `ensemble`.
        Example: {"Temperature (K)": "temperature", "RH": "relative_humidity"}
        If None, tries sensible defaults when present.
    gas_names : list[str] or None
        Names for gas species columns to extract from `ensemble.gas_concs`.
        Accepts either shape (n, n_gas) or iterable of length n_gas with each entry shape (n,).
    sample_ids : array-like or None
        Per-sample identifiers. If None, uses 1..n (int).
    """
    # infer n from the first array-like attribute we can find
    candidates = [
        getattr(ensemble, "relative_humidity", None),
        getattr(ensemble, "temperature", None),
        getattr(ensemble, "flux", None),
        getattr(ensemble, "pressure", None),
    ]
    n = None
    for c in candidates:
        if c is not None:
            try:
                n = len(c)
                break
            except Exception:
                # Ignore if attribute is missing or not array-like; try next candidate
                pass
    if n is None:
        raise ValueError("Could not infer ensemble size `n` from standard attributes.")

    if sample_ids is None:
        sample_ids = np.arange(1, n+1, dtype=int)

    # default variables if not provided
    if variables is None:
        variables = {}
        for label, attr in [
            ("Temperature (K)", "temperature"),
            ("Relative humidity", "relative_humidity"),
            ("Pressure (Pa)", "pressure"),
            ("Flux", "flux"),
        ]:
            if hasattr(ensemble, attr):
                variables[label] = attr

    rows = []

    # top-level scalar/array attributes
    for label, attr in variables.items():
        arr = _as_array(getattr(ensemble, attr), n)
        if arr is not None:
            for s, v in zip(sample_ids, arr):
                rows.append({"variable": label, "value": float(v), "sample": int(s)})

    # gas concentrations (if requested and present)
    if gas_names and hasattr(ensemble, "gas_concs"):
        gas = getattr(ensemble, "gas_concs")
        # try (n, n_gas)
        try:
            g = np.asarray(gas)
            if g.ndim == 2 and g.shape[0] == n and g.shape[1] == len(gas_names):
                for j, name in enumerate(gas_names):
                    for s, v in zip(sample_ids, g[:, j]):
                        rows.append({"variable": f"{name} (mixing ratio)", "value": float(v), "sample": int(s)})
            else:
                raise Exception
        except Exception:
            # try iterable of length n_gas with each entry shape (n,)
            try:
                series = [np.asarray(x) for x in gas]
                if len(series) == len(gas_names) and all(len(x) == n for x in series):
                    for name, vec in zip(gas_names, series):
                        for s, v in zip(sample_ids, vec):
                            rows.append({"variable": f"{name} (mixing ratio)", "value": float(v), "sample": int(s)})
            except Exception:
                pass  # silently ignore if structure is unknown
    
    return pd.DataFrame(rows, columns=["variable", "value", "sample"])



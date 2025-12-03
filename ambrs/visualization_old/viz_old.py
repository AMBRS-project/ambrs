import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# PLOT OUTPUT STATE
# ---------------------------------------------------------------------------
# ===========================================================
# Flexible grid renderer (rows=scenarios × cols=user-defined)
# - Accepts a matplotlib.gridspec.GridSpec
# - Works for dNdlnD, frac_ccn, b_scat, etc. via var_cfg builders
# - PartMC solid / MAM4 dashed+thicker; color by scenario (row)
# ===========================================================
from typing import Callable, Iterable, Sequence, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from .. import partmc, mam4
from pyparticle.viz.style import StyleManager, Theme
from pyparticle.viz.builder import build_plotter

# -----------------------------------------------------------
# Retrieve partmc and mam4 model states for given scenario/timestep
# -----------------------------------------------------------
def make_state_retriever(ensemble, partmc_dir, mam4_dir, *, repeat_num=1):
    cache = {}

    def retrieve(scenario_name: str, timestep: int, species_modifications=None):
        key = (scenario_name, int(timestep))
        if key in cache:
            return cache[key]

        idx = int(scenario_name) - 1
        scen = ensemble.member(idx)

        # Assumes `ambrs` is available in your environment
        partmc_output = partmc.retrieve_model_state(
            scenario_name=scenario_name,
            scenario=scen,
            timestep=timestep,
            repeat_num=repeat_num,
            species_modifications=species_modifications or {},
            ensemble_output_dir=partmc_dir,
        )
        mam4_output = mam4.retrieve_model_state(
            scenario_name=scenario_name,
            scenario=scen,
            timestep=timestep,
            repeat_num=repeat_num,
            species_modifications=species_modifications or {},
            ensemble_output_dir=mam4_dir,
        )
        cache[key] = (partmc_output, mam4_output)
        return cache[key]

    return retrieve


# -----------------------------------------------------------
# Styling helpers (row color, solid vs dashed, thicker MAM4)
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

def _format_panel(ax, *, xscale=None):
    if xscale:
        ax.set_xscale(xscale)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -----------------------------------------------------------
# Column builder pattern
# - You pass a function that, given a column key (e.g., timestep),
#   returns (var_cfg: dict, title: str)
# -----------------------------------------------------------
VarCfgBuilder = Callable[[Any], Tuple[Dict[str, Any], str]]

def make_dNdlnD_cfg_builder(*, D_range=(1e-9, 1e-6), N_bins=50, normalize=True, method=""):
    D_min, D_max = D_range
    D_grid = np.logspace(np.log10(D_min), np.log10(D_max), int(N_bins))

    def builder(col_value) -> Tuple[Dict[str, Any], str]:
        # col_value is commonly the timestep; title can reflect that
        title = f"t={col_value-1} min" if isinstance(col_value, (int, np.integer)) else str(col_value)
        return ({"D": D_grid, "normalize": normalize, "method": method}, title)
    return builder

def make_frac_ccn_cfg_builder(*, s_grid=np.logspace(-2, 1.0, 50)):
    s = np.asarray(s_grid)
    def builder(col_value) -> Tuple[Dict[str, Any], str]:
        title = f"t={col_value-1} min" if isinstance(col_value, (int, np.integer)) else str(col_value)
        return ({"s_grid": s}, title)
    return builder

def make_bscat_cfg_builder(*, wavelengths_m=np.linspace(0.35e-6, 0.8e-6, 30), RH=0.0):
    wl = np.asarray(wavelengths_m).ravel()
    RH_scalar = float(np.asarray(RH).ravel()[0])
    def builder(col_value) -> Tuple[Dict[str, Any], str]:
        title = f"t={col_value-1} min" if isinstance(col_value, (int, np.integer)) else str(col_value)
        # Prefer 'wvl_grid' + 'rh_grid' to match your newer API; callers can change if needed
        return ({"wvl_grid": wl, "rh_grid": np.array([RH_scalar])}, title)
    return builder


# -----------------------------------------------------------
# Core engine: render_variable_grid(...)
# - Uses retriever + build_plotter("state_line", cfg)
# - Fills the provided GridSpec (must be at least rows×cols big)
# -----------------------------------------------------------
DEFAULT_XSCALE = {
    "dNdlnD"  : "log",
    "frac_ccn": "log",
    "b_scat"  : None,
}

def render_variable_grid(
    gs,                               # matplotlib.gridspec.GridSpec
    *,
    ensemble,
    scenarios: Sequence[str],
    columns: Sequence[Any],           # e.g., timesteps [1, 3, 6] or ["t0","t1",...]
    partmc_dir: str,
    mam4_dir: str,
    varname: str,                     # e.g., "dNdlnD", "frac_ccn", "b_scat"
    build_var_cfg: VarCfgBuilder,     # maps column value -> (var_cfg, title)
    legend_loc: str = "upper left",
    row_label_fmt: str = "scenario {name}",
    row_colors: Sequence[Any] | None = None,
    xscale: str | None = None,        # None -> uses DEFAULT_XSCALE[varname] if present
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Render a grid where rows = scenarios and columns = user-defined 'columns'
    (e.g., timesteps). Draws PartMC vs MAM4 overlays per cell.

    Returns
    -------
    fig, axes : (matplotlib.figure.Figure, np.ndarray[(n_rows, n_cols)])
    """
    fig = gs.figure
    n_rows, n_cols = len(scenarios), len(columns)

    # sanity: GridSpec must be big enough
    if getattr(gs, "nrows", None) is not None and getattr(gs, "ncols", None) is not None:
        if gs.nrows < n_rows or gs.ncols < n_cols:
            raise ValueError(f"GridSpec too small: needs at least {n_rows}×{n_cols}, "
                             f"got {gs.nrows}×{gs.ncols}")

    axes = np.empty((n_rows, n_cols), dtype=object)
    retr = make_state_retriever(ensemble, partmc_dir, mam4_dir)
    base = _base_styles()
    colors = _scenario_colors(n_rows) if row_colors is None else list(row_colors)
    xsc = DEFAULT_XSCALE.get(varname, None) if xscale is None else xscale

    for i, scenario_name in enumerate(scenarios):
        row_style = _row_styles(base, colors[i])

        for j, col_value in enumerate(columns):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax

            # Pull states once per cell
            partmc_output, mam4_output = retr(scenario_name, col_value)
            series = (
                ("partmc", partmc_output.particle_population, "PartMC"),
                ("mam4",   mam4_output.particle_population,   "MAM4"),
            )

            # Build var_cfg + column title
            var_cfg, col_title = build_var_cfg(col_value)

            # Plot both series
            for key, population, label in series:
                cfg = {"varname": varname, "var_cfg": var_cfg, "style": row_style[key]}
                build_plotter("state_line", cfg).plot(population, ax, label=label)

            # panel cosmetics
            _format_panel(ax, xscale=xsc)

            # column titles (top row only)
            if i == 0 and col_title:
                ax.set_title(col_title)

            # legend (only once)
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc=legend_loc)
            
    fig.canvas.draw_idle()
    return fig, axes


# -----------------------------------------------------------
# Convenience wrappers for common cases
# -----------------------------------------------------------
def render_dNdlnD_vs_time(
    gs, *, ensemble, scenarios, timesteps, partmc_dir, mam4_dir,
    D_range=(1e-9, 1e-6), N_bins=50, normalize=True, method="",
    legend_loc="upper left",
):
    return render_variable_grid(
        gs,
        ensemble=ensemble,
        scenarios=scenarios,
        columns=list(timesteps if isinstance(timesteps, Iterable) else [timesteps]),
        partmc_dir=partmc_dir,
        mam4_dir=mam4_dir,
        varname="dNdlnD",
        build_var_cfg=make_dNdlnD_cfg_builder(
            D_range=D_range, N_bins=N_bins, normalize=normalize, method=method
        ),
        legend_loc=legend_loc,
        row_label_fmt="scenario {name}",
    )

def render_frac_ccn_vs_time(
    gs, *, ensemble, scenarios, timesteps, partmc_dir, mam4_dir,
    s_grid=np.logspace(-2, 1.0, 50), legend_loc="upper left",
):
    return render_variable_grid(
        gs,
        ensemble=ensemble,
        scenarios=scenarios,
        columns=list(timesteps if isinstance(timesteps, Iterable) else [timesteps]),
        partmc_dir=partmc_dir,
        mam4_dir=mam4_dir,
        varname="frac_ccn",
        build_var_cfg=make_frac_ccn_cfg_builder(s_grid=s_grid),
        legend_loc=legend_loc,
        row_label_fmt="scenario {name}",
    )

def render_bscat_vs_time(
    gs, *, ensemble, scenarios, timesteps, partmc_dir, mam4_dir,
    wavelengths_m=np.linspace(0.35e-6, 0.8e-6, 30), RH=0.0, legend_loc="upper left",
):
    return render_variable_grid(
        gs,
        ensemble=ensemble,
        scenarios=scenarios,
        columns=list(timesteps if isinstance(timesteps, Iterable) else [timesteps]),
        partmc_dir=partmc_dir,
        mam4_dir=mam4_dir,
        varname="b_scat",
        build_var_cfg=make_bscat_cfg_builder(wavelengths_m=wavelengths_m, RH=RH),
        legend_loc=legend_loc,
        row_label_fmt="scenario {name}",
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



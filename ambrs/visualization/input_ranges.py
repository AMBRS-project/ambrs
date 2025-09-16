import matplotlib.pyplot as plt
import numpy as np

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
        ax.scatter(vals, np.full_like(vals, 0.5),
                   s=40, facecolor="black", edgecolor="white", lw=0.7, alpha=0.2, zorder=2)

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

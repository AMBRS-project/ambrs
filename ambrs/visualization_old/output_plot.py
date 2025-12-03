import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re

sns.set_theme(style="dark", rc={"axes.facecolor": (0, 0, 0, 0)})

# -------------------------------
# Utilities (unchanged behavior)
# -------------------------------

def get_row_colors(row_labs, palette=None):
    """Map particle sizes to colors."""
    if isinstance(palette, str) or palette is None:
        colors = sns.color_palette(palette=palette, n_colors=len(row_labs))
    elif isinstance(palette, list):
        colors = palette
        if len(colors) < len(row_labs):
            raise ValueError("Not enough colors in list for the number of sizes")
    else:
        raise ValueError("palette must be None, a string, or a list of colors")
    return dict(zip(row_labs, colors))

# todo: remove this?
def get_row_color(row_lab, row_to_color, fallback="#d3d3d3"):
    m = re.search(r'(\d+)', str(row_lab))
    if m:
        # size = int(m.group(1))
        return row_to_color.get(row_lab, fallback)
    return fallback

def blend_colors(foreground_rgb, background_rgb, alpha):
    return tuple(alpha*f + (1-alpha)*b for f,b in zip(foreground_rgb, background_rgb))

# Helper: build log-midpoint edges so shapes don’t stretch to the right
def _log_mid_edges(x):
    """Given strictly-positive, sorted centers x, return N+1 edges on log10 scale."""
    lx = np.log10(x)
    left  = lx[0] - (lx[1]-lx[0])/2.0 if len(lx) > 1 else lx[0] - 0.05
    right = lx[-1] + (lx[-1]-lx[-2])/2.0 if len(lx) > 1 else lx[0] + 0.05
    edges_log = np.concatenate(([left], (lx[:-1] + lx[1:]) / 2.0, [right]))
    return 10**edges_log

def _step_xy_from_bins(edges, y):
    """Make stepwise x,y arrays (constant over each bin) for fill/outline."""
    # edges: length N+1; y: length N
    x_step = np.repeat(edges, 2)[1:-1]
    y_step = np.repeat(y, 2)
    return x_step, y_step

def build_comparison_dfs(varname, var_config, partmc_outputs, mam4_outputs, scenario_names=None):
    """
    Build comparison DataFrames for CCN and optical properties using a var_config dictionary.
    var_config is passed to Output.compute_variable().

    Parameters
    ----------
    partmc_outputs, mam4_outputs : list[Output]
        Lists of Output objects for each model.
    var_config : dict
        Configuration for variable computation, e.g., {'varname':'Nccn', 's_eval':[0.05,0.1,...]}
    scenario_names : list[str], optional
        Scenario labels; inferred from Output.scenario_name if None.
    
    Returns
    -------
    df
    """
    
    model_lists = [('partmc', partmc_outputs), ('mam4', mam4_outputs)]
    df = pd.DataFrame({'x': [], 'w': [], 'g': []})
    for model_label, outputs in model_lists:
        for idx, out in enumerate(outputs):
            scen_label = getattr(out, 'scenario_name', None) or (scenario_names[idx] if scenario_names else f'{model_label}:{idx}')
            df['x'] = scen_label
            df['w'] = out.compute_variable(varname, var_config)[varname]
            df['g'] = model_label
            df['varname'] = varname #var_config['name'] 
    
    return df

def build_sizedist_df(
        outputs, group_type='scenario', # could also be timesteps
        wetsize = True, normalize=True, method = 'hist', 
        N_bins = 30, D_min = 1e-9, D_max = 1e-4):
    
    df = pd.DataFrame({'x': [], 'w': [], 'g': []})
    rows = []  # collect rows here
    for output in outputs:
        size_dict = output.compute_dNdlnD(
            wetsize = wetsize, normalize=normalize, 
            method = method, N_bins = N_bins, 
            D_min = D_min, D_max = D_max)
        if group_type == 'scenario':
            group_lab = output.scenario_name
        elif group_type == 'timestep':
            group_lab = output.timestep
        elif group_type == 'scenario&timestep':
            group_lab = output.scenario_name + ', ' + str(output.timestep)
        
        for D, dNdlnD in zip(size_dict['D'], size_dict['dNdlnD']):
            rows.append({"x": D, "w": dNdlnD, "g": group_lab})
    
        # build the DataFrame once
        df = pd.DataFrame(rows, columns=["x", "w", "g"])
    return df

# # -------------------------------
# # Ridge plot from PDFs, with second model line
# # -------------------------------
# def plot_ridge(
#     dfA, # data frame containing size distribution details
#     dfB=None, # optional second line
#     user_colors=None, 
#     log_x=True, show_row_label=True, vline_x=0.5, 
#     save_name=None,
#     blank_rows=[], # where to add blanks
#     N_blanks = 0, # N blank rows per blank
#     xlim = None,
#     ylim = None,
#     axis_fontsize=12, tick_fontsize=10, row_label_fontsize=10,
#     axis_font="Arial", tick_font="Arial", row_label_font="Arial",
#     height=0.5, aspect=10, outline_color='w',
#     critical_diams=None,               # array-like, one cutoff per plotted row (same order as columns)
#     normalize_mode="sum_dx",           # 'sum_dx' (normalize by sum(w)*Δln x), 'sum' (simple sum), or None
#     x_max_allowed=40.0,
#     line_kwargs=None,                  # for 2nd model: dict(lw=2, ls='--', alpha=1.0)
#     lighten_alpha=0.3,                 # how much to blend fill toward white
# ):
#     """
#     Draw a ridge of dN/dlnD-style PDFs directly (no sampling) with:
#       - Model A: filled area via fill_between (with outline + cutoff recolor)
#       - Model B: line overlay (optional)

#     CSV format: first column is x (diameter), remaining columns are rows (scenarios/sizes)
#     """
#     # ---------------- Load data ----------------
    
#     # edges = _log_mid_edges(x)
#     # dlnx = np.diff(np.log(x))          # used only if you want an approximate local Δlnx
#     # dx_eff = np.log(edges[1:]) - np.log(edges[:-1])  # exact Δlnx per bin

#     # # ---------------- Colors & outlines ----------------
#     # size_to_color = get_size_colors(particle_sizes, palette=user_colors)
#     # palette = {g: get_size_color(g, size_to_color) for g in groups}
    
#     # blank_rows = ['blank_1','blank_2','blank_3','blank_1b','blank_2b','blank_3b']
#     # for b in blank_rows:
#     #     palette[b] = 'white'  # neutral color for space rows

#     # Row order with blanks (exactly your ordering)
    
#     # for blank_row in blank_rows:
        
#     # groups = df['g']
#     # groups_with_blanks = ()
#     # groups_with_blanks = (
#     #     groups[:3] + ['blank_1'] + ['blank_1b'] +
#     #     groups[6:9] + ['blank_2'] + ['blank_2b'] +
#     #     groups[3:6] + ['blank_3'] + ['blank_3b'] +
#     #     groups[9:]
#     # )
    

#     # # Outline per row
#     # if isinstance(outline_color, list):
#     #     n_non_blank = len([g for g in groups_with_blanks if g not in blank_rows])
#     #     repeats = (n_non_blank // len(outline_color)) + 1
#     #     extended = (outline_color * repeats)[:n_non_blank]
#     #     outline_full, idx = [], 0
#     #     for g in groups_with_blanks:
#     #         if g in blank_rows:
#     #             outline_full.append('white')
#     #         else:
#     #             outline_full.append(extended[idx]); idx += 1
#     #     row_to_outline = dict(zip(groups_with_blanks, outline_full))
#     # else:
#     #     row_to_outline = {g: (outline_color if g not in blank_rows else 'white') for g in groups_with_blanks}
    
#     # # Critical diameters (per visible row)
#     # if critical_diams is None:
#     #     # Same default vector length as your example (repeats if needed)
#     #     critical_diams = np.array([0.4,1.2,5,0.4,1.2,5,0.4,1.2,5,0.4,1.2,5])
#     # cutoff_map = {gname: cutoff for gname, cutoff in zip(groups, critical_diams)}

#     # # Line style for model B
#     # if line_kwargs is None:
#     #     line_kwargs = dict(lw=2, ls='--', alpha=1.0)

#     # # ---------------- Melt just to drive FacetGrid rows ----------------
#     # # We only use this long frame to manage rows/spacing/labels like your version.
#     # long = dfA.melt(id_vars=["x"], var_name="g", value_name="w").replace([np.inf, -np.inf], np.nan)
#     # long = long.dropna(subset=["x", "w"])
#     # long = long[long["w"] > 0].copy()
#     # long["x_orig"] = long["x"]
#     # blanks = [pd.DataFrame({'x':[np.nan], 'g':[b], 'w':[np.nan], 'x_orig':[np.nan]}) for b in blank_rows]
#     # plot_data = pd.concat([long]+blanks, ignore_index=True)
#     # plot_data['g'] = pd.Categorical(plot_data['g'], categories=groups_with_blanks, ordered=True)

#     # ---------------- Ridge plot ----------------
#     if isinstance(outline_color, list):
#         row_to_outline = {g: one_outline_color for (g,one_outline_color) in zip(dfA['g'], outline_color)}
#     else:
#         row_to_outline = {g: outline_color for g in dfA['g']}
#     x = dfA['x']
    
#     row_to_color = get_row_colors(dfA['g'], palette=user_colors)
#     palette = {g: get_row_color(g, row_to_color) for g in dfA['g']}
    
#     g = sns.FacetGrid(dfA, row="g", hue="g", aspect=aspect, height=height, palette=palette)
    
#     def plot_layers(data, color, **kwargs):
#         row_name = data['g'].iloc[0]
#         if row_name.startswith('blank'):
#             return  # spacing only
#         ax = plt.gca()
        
#         outline = row_to_outline.get(row_name, 'w')
        
#         # Extract row series (already sorted)
#         wA = dfA[row_name].to_numpy() if row_name in dfA.columns else None
#         if wA is None:
#             return
        
#         # Hard x cutoff
#         mask = x <= x_max_allowed
#         x_cut = x[mask]
#         wA = wA[mask]
#         edges_cut = _log_mid_edges(x_cut)

#         # Normalize options (apply per-row)
#         if normalize_mode == "sum_dx":
#             norm = np.sum(wA * (np.log(edges_cut[1:]) - np.log(edges_cut[:-1])))
#             if norm > 0: wA = wA / norm
#         elif normalize_mode == "sum":
#             s = np.nansum(wA)
#             if s > 0: wA = wA / s
#         # else: leave as-is

#         # Build step arrays and draw filled area (Model A)
#         Xs, Ys = _step_xy_from_bins(edges_cut, wA)
        
#         # light_color = blend_colors(color, [1., 1., 1.], lighten_alpha)

#         # Outline first for crisp edge
#         ax.plot(Xs, Ys, color=outline, linewidth=2., zorder=2, solid_capstyle='round', clip_on=False)
#         # Main fill
#         ax.fill_between(Xs, 0, Ys, color=color, zorder=1, step=None, linewidth=0, clip_on=False)
        
#         # # Recolor portion above cutoff
#         # cutoff = cutoff_map.get(row_name, 0.0)
#         # if cutoff > 0:
#         #     # Select bins whose centers are >= cutoff
#         #     above = x_cut >= cutoff
#         #     if np.any(above):
#         #         Xs2, Ys2 = _step_xy_from_bins(edges_cut, np.where(above, wA, 0.0))
#         #         ax.fill_between(Xs2, 0, Ys2, color=color, zorder=3, linewidth=0, clip_on=False)

#         # Optional Model B: line overlay (no fills)
#         if dfB is not None:
#             wB = dfB[row_name].to_numpy()[mask]
#             if normalize_mode == "sum_dx":
#                 normB = np.sum(wB * (np.log(edges_cut[1:]) - np.log(edges_cut[:-1])))
#                 if normB > 0: wB = wB / normB
#             elif normalize_mode == "sum":
#                 sB = np.nansum(wB)
#                 if sB > 0: wB = wB / sB
#             # Thin line over centers
#             ax.plot(x_cut, wB, color=color, zorder=4, **line_kwargs)

#     g.map_dataframe(plot_layers)
#     g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
#     g.figure.subplots_adjust(hspace=-.25)
#     g.set_titles("")
#     g.set(yticks=[], ylabel="")
#     g.despine(bottom=True, left=True)

#     # Size labels (left, colored)
#     if show_row_label:
#         for ax, gname in zip(g.axes.flat, dfA['g']):
#             if gname.startswith('blank'): 
#                 continue
#             m = re.search(r'(\d+)', gname)
#             size_label = f"{m.group(1)} nm" if m else gname
#             ax.text(0.01, 0.08, size_label, fontweight="bold",
#                     color=palette[gname], ha="left", va="bottom",
#                     transform=ax.transAxes,
#                     fontsize=row_label_fontsize,
#                     fontname=row_label_font)

#     # Axes adjustments & ticks (exactly your vibe)
#     for ax in g.axes.flat:
#         if log_x:
#             ax.set_xscale('log')
#             major_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
#             ax.set_xticks(major_ticks)
#             ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
#             ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(1.0,10.0)*0.1))
#         if xlim: 
#             ax.set_xlim(xlim)
        
#         if ylim: ax.set_ylim(ylim)

#         ax.set_xlabel("droplet diameter [µm]", fontsize=axis_fontsize, fontname=axis_font)
#         ax.tick_params(labelsize=tick_fontsize)
#         for label in ax.get_xticklabels() + ax.get_yticklabels():
#             label.set_fontname(tick_font)
#         # Optional vertical refline for SAM
#         # if isinstance(model_fill_csv, str) and ('SAM' in model_fill_csv):
#         #     ax.axvline(vline_x, color='k', alpha=0.3, linestyle=':', linewidth=1, zorder=10)

#     if save_name:
#         g.fig.savefig(save_name, dpi=300, transparent=True)
#     return g
def plot_ridge(
    dfA,                # DataFrame with columns ['x','w','g']
    dfB=None,           # Optional second dataset, same structure
    user_colors=None, 
    log_x=True, 
    show_row_label=True, 
    vline_x=0.5, 
    save_name=None,
    xlim=None, 
    ylim=None,
    axis_fontsize=12, 
    tick_fontsize=10, 
    row_label_fontsize=10,
    axis_font="Arial", 
    tick_font="Arial", 
    row_label_font="Arial",
    height=0.5, 
    aspect=10, 
    outline_color='w',
    critical_diams=None,          # array-like, cutoff per row (optional)
    lighten_alpha=0.3,            # how much to lighten fill toward white
    x_max_allowed=40.0,
    line_kwargs=None,              # kwargs for dfB overlay line
    as_bar=True
):
    """
    Ridge-style plot of size distributions.
    Bars for dfA, optional line overlay for dfB.
    """
    groups = dfA['g'].unique().tolist()

    # Colors
    if isinstance(user_colors, list):
        palette = dict(zip(groups, user_colors))
    else:
        palette = dict(zip(groups, sns.color_palette("crest", len(groups))))
    
    # Critical diameters → map to each group (else 0)
    if critical_diams is None:
        critical_diams = np.zeros(len(groups))
    cutoff_map = dict(zip(groups, critical_diams))

    # Outline color per row
    if isinstance(outline_color, list):
        row_to_outline = dict(zip(groups, outline_color))
    else:
        row_to_outline = {g: outline_color for g in groups}

    # FacetGrid
    g = sns.FacetGrid(dfA, row="g", hue="g", aspect=aspect, height=height, palette=palette)

    def plot_layer(data, color, **kwargs):
        row_name = data['g'].iloc[0]
        outline = row_to_outline.get(row_name, 'w')

        # Sort & mask
        x = data['x'].to_numpy()
        w = data['w'].to_numpy()
        order = np.argsort(x)
        x, w = x[order], w[order]
        mask = x <= x_max_allowed
        x, w = x[mask], w[mask]

        if len(x) < 1:
            return

        # Compute bin widths from log-midpoints
        if len(x) >= 2:
            lx = np.log10(x)
            edges_log = np.concatenate((
                [lx[0] - (lx[1]-lx[0])/2.0],
                (lx[:-1] + lx[1:]) / 2.0,
                [lx[-1] + (lx[-1]-lx[-2])/2.0]
            ))
            edges = 10**edges_log
            dx_vals = np.diff(edges)
        else:
            dx_vals = np.array([x[0] * (10**0.05 - 10**-0.05)])

        # Lightened fill
        light_color = blend_colors(color, [1.,1.,1.], lighten_alpha)

        # Outline
        plt.bar(x, w, width=dx_vals, facecolor="none", edgecolor=outline,
                align='center', linewidth=2., clip_on=False, zorder=0)

        # Main fill
        plt.bar(x, w, width=dx_vals, color=light_color, edgecolor='none',
                align='center', clip_on=False, zorder=1)

        # Above-cutoff fill
        cutoff = cutoff_map.get(row_name, 0)
        mask = x >= cutoff
        if np.any(mask):
            plt.bar(x[mask], w[mask], width=dx_vals[mask], color=color,
                    edgecolor='none', align='center', clip_on=False, zorder=2)

        # Overlay dfB (if provided)
        if dfB is not None:
            subB = dfB[dfB['g'] == row_name]
            if not subB.empty:
                plt.plot(subB['x'], subB['w'], color=outline, linewidth=3.,zorder=5)
                plt.plot(subB['x'], subB['w'], color=color,
                         **(line_kwargs or dict(ls='--', lw=1.5)),zorder=6)


    def plot_layer_fillbetween(data, color, **kwargs):
        row_name = data['g'].iloc[0]
        outline = row_to_outline.get(row_name, 'w')

        # Sort & mask
        x = data['x'].to_numpy()
        w = data['w'].to_numpy()
        order = np.argsort(x)
        x, w = x[order], w[order]
        mask = x <= x_max_allowed
        x, w = x[mask], w[mask]

        if len(x) < 1:
            return

        # Compute bin widths from log-midpoints
        if len(x) >= 2:
            lx = np.log10(x)
            edges_log = np.concatenate((
                [lx[0] - (lx[1]-lx[0])/2.0],
                (lx[:-1] + lx[1:]) / 2.0,
                [lx[-1] + (lx[-1]-lx[-2])/2.0]
            ))
            edges = 10**edges_log
            dx_vals = np.diff(edges)
        else:
            dx_vals = np.array([x[0] * (10**0.05 - 10**-0.05)])

        # Lightened fill
        light_color = blend_colors(color, [1.,1.,1.], lighten_alpha)

        # Outline
        plt.fill_between(x, np.zeros(len(w)), w, 
                         facecolor="none", edgecolor=outline,
                         linewidth=2., clip_on=False, zorder=0)

        # Main fill
        # plt.bar(x, w, width=dx_vals, color=light_color, edgecolor='none',
        #         align='center', clip_on=False, zorder=1)
        plt.fill_between(x, np.zeros(len(w)), w, 
                         facecolor=light_color, edgecolor='none',
                         clip_on=False, zorder=1)


        # Above-cutoff fill
        cutoff = cutoff_map.get(row_name, 0)
        mask = x >= cutoff
        if np.any(mask):
            # plt.bar(x[mask], w[mask], width=dx_vals[mask], color=color,
            #         edgecolor='none', align='center', clip_on=False, zorder=2)
    
            plt.fill_between(x[mask], np.zeros(len(w[mask])), w[mask], 
                             facecolor=color, edgecolor='none',
                             clip_on=False, zorder=1)
        # Overlay dfB (if provided)
        if dfB is not None:
            subB = dfB[dfB['g'] == row_name]
            if not subB.empty:
                plt.plot(subB['x'], subB['w'], color=outline, linewidth=3.,zorder=5)
                plt.plot(subB['x'], subB['w'], color=color,
                         **(line_kwargs or dict(ls='--', lw=1.5)),zorder=6)
    
    if as_bar:
        g.map_dataframe(plot_layer)
    else:
        g.map_dataframe(plot_layer_fillbetween)
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)
    
    # Aesthetics
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Row labels
    if show_row_label:
        for ax, gname in zip(g.axes.flat, groups):
            ax.text(0.01, 0.08, str(gname),
                    fontweight="bold", color=palette[gname],
                    ha="left", va="bottom", transform=ax.transAxes,
                    fontsize=row_label_fontsize, fontname=row_label_font)

    # Axes formatting
    for ax in g.axes.flat:
        if log_x:
            ax.set_xscale('log')
    #         major_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    #         ax.set_xticks(major_ticks)
    #         ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    #         ax.xaxis.set_minor_locator(
    #             mticker.LogLocator(base=10.0, subs=np.arange(1.0,10.0)*0.1)
    #         )
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

        ax.set_xlabel("diameter [m]", fontsize=axis_fontsize, fontname=axis_font)
        ax.tick_params(labelsize=tick_fontsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(tick_font)
        # if vline_x:
        #     ax.axvline(vline_x, color='k', alpha=0.3, ls=':', lw=1, zorder=10)

    if save_name:
        plt.savefig(save_name, dpi=300, transparent=True)
    return g

def plot_dots(
    df,
    ax=None,
    color_map=None,
    marker='o',
    size=50,
    alpha=0.8,
    edgecolor='black'
):
    """
    Dot-style plot aligned with ridge rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['x','w','g','row'] where 'row' is a numeric
        row index used for vertical positioning. The y-position of each
        point is computed as row + w (so w acts as a vertical offset).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, uses plt.gca().
    color_map : dict, optional
        Mapping group name -> color. If None, uses the matplotlib color cycle.
    marker : str, optional
        Marker style, default 'o'.
    size : float, optional
        Marker size for scatter (points), default 50.
    alpha : float, optional
        Marker transparency, default 0.8.
    edgecolor : color, optional
        Marker edge color when using open markers, default 'black'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the plotted dots.
    """
    if ax is None:
        ax = plt.gca()

    # Validate input
    required = {'x', 'w', 'g'}
    if not required.issubset(set(df.columns)):
        raise ValueError("df must contain columns 'x', 'w', and 'g'")

    # If no explicit numeric row provided, map groups to integer rows
    if 'row' not in df.columns:
        groups = list(pd.Categorical(df['g']).categories)
        row_map = {g: i for i, g in enumerate(groups[::-1])}  # top-to-bottom
        df = df.copy()
        df['row'] = df['g'].map(row_map)

    groups = df['g'].unique().tolist()

    # Build default color map if not provided
    if color_map is None:
        cycle = plt.rcParams.get('axes.prop_cycle').by_key().get('color', None)
        if cycle is None:
            # fallback to seaborn
            palette = sns.color_palette(n_colors=len(groups))
            colors = [palette[i % len(palette)] for i in range(len(groups))]
        else:
            colors = [cycle[i % len(cycle)] for i in range(len(groups))]
        color_map = dict(zip(groups, colors))

    # Plot each group as a separate scatter series
    for g in groups:
        sub = df[df['g'] == g]
        if sub.empty:
            continue
        x = sub['x'].to_numpy()
        w = sub['w'].to_numpy()
        row_idx = sub['row'].to_numpy()
        y = row_idx + w

        color = color_map.get(g, None)

        # If user requested an open marker by specifying color 'none' or None,
        # draw marker with facecolors='none' and edgecolors=edgecolor
        if isinstance(color, str) and color.lower() in ('none', 'none'):
            sc = ax.scatter(x, y, s=size, marker=marker, facecolors='none',
                            edgecolors=edgecolor, alpha=alpha, zorder=3)
        elif color is None:
            sc = ax.scatter(x, y, s=size, marker=marker, facecolors='none',
                            edgecolors=edgecolor, alpha=alpha, zorder=3)
        else:
            # Filled marker
            sc = ax.scatter(x, y, s=size, marker=marker, c=[color],
                            edgecolors=edgecolor, alpha=alpha, zorder=3)

    # Aesthetics: hide y-axis ticks/labels to match ridge-style
    ax.set_yticks([])
    ax.set_ylabel("")

    # Keep x-axis label (units should be set by caller)
    ax.tick_params(axis='both', which='major', labelsize=10)

    return ax


# -------------------------------
# Your box plot stays as-is
# -------------------------------
def plot_box(df, value_col, row_col='row_name', particle_sizes=[50,100,250], size_col='Aerosol Size',
             user_palette=None, log_x=True, show_labels=True, draw_box=True,
             grid_style="horizontal", x_ticks_position="bottom",
             x_axis_position="bottom", y_axis_style="full",
             fig_width=3., fig_height=6., xlabel=None, save_name=None,
             axis_fontsize=12, tick_fontsize=10, axis_font="Arial", tick_font="Arial",
             outline_color='w'):
    # (unchanged from your snippet)
    # size_to_color = get_size_colors(particle_sizes, palette=user_palette)
    if row_col not in df.columns:
        df[row_col] = df['Case Run'] + " " + df['Aerosol Size']
    rows = list(df[row_col].unique())
    blank_rows = ['blank_1','blank_2','blank_3']
    rows_with_blanks = rows[:3] + ['blank_1'] + rows[6:9] + ['blank_3'] + rows[3:6] + ['blank_2'] + rows[9:]
    for blank in blank_rows:
        df = pd.concat([df, pd.DataFrame({row_col:[blank], value_col:[np.nan], size_col:[np.nan]})], ignore_index=True)
    df[row_col] = pd.Categorical(df[row_col], categories=rows_with_blanks, ordered=True)

    non_blank_rows = [r for r in rows_with_blanks if not r.startswith('blank')]
    if isinstance(outline_color, list):
        repeats = (len(non_blank_rows) // len(outline_color)) + 1
        extended_outline = (outline_color * repeats)[:len(non_blank_rows)]
        row_to_outline = {}
        idx = 0
        for r in rows_with_blanks:
            if r.startswith('blank'):
                row_to_outline[r] = 'white'
            else:
                row_to_outline[r] = extended_outline[idx]; idx += 1
    else:
        row_to_outline = {r: outline_color if not r.startswith('blank') else 'white'
                          for r in rows_with_blanks}

    nrows = len(rows_with_blanks)
    y_positions = {row: nrows - i - 1 for i, row in enumerate(rows_with_blanks)}
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for row in rows_with_blanks:
        if row.startswith('blank'): 
            continue
        subset = df[df[row_col] == row].copy()
        y_pos = y_positions[row]
        data = subset[value_col].dropna()
        if log_x: data = np.log10(data[data>0])
        color = get_size_color(subset[size_col].iloc[0], size_to_color)
        ax.boxplot(data, positions=[y_pos], vert=False, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=color, color=color, linewidth=1.5),
                   whiskerprops=dict(color=color, linewidth=1.5),
                   capprops=dict(color=color, linewidth=1.5),
                   medianprops=dict(color=row_to_outline[row], linewidth=2),
                   showfliers=False)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([r if show_labels else '' for r in rows_with_blanks], 
                       fontsize=tick_fontsize, fontname=tick_font)

    if grid_style=="full":
        ax.grid(True, which="both", linestyle="--", color="lightgray", alpha=0.5)
    elif grid_style=="horizontal":
        ax.xaxis.grid(False); ax.yaxis.grid(True, linestyle="--", color="lightgray", alpha=0.5, linewidth=0.5)
    elif grid_style =="none":
        ax.grid(False)
    
    if log_x:
        xticks = ax.get_xticks()
        ax.set_xticklabels([f"{10**x:.2g}" for x in xticks], fontsize=tick_fontsize, fontname=tick_font)
    ax.set_xlabel(xlabel if xlabel else value_col, fontsize=axis_fontsize, fontname=axis_font)

    for spine_name, spine in ax.spines.items():
        if spine_name in ["top","bottom"]:
            spine.set_visible(x_axis_position in ["top","bottom","both"])
        elif spine_name in ["left","right"]:
            spine.set_visible(y_axis_style=="full" and draw_box)

    ax.tick_params(axis="x", bottom=x_ticks_position in ["bottom","both"],
                   top=x_ticks_position in ["top","both"],
                   labelbottom=x_ticks_position in ["bottom","both"],
                   labeltop=x_ticks_position in ["top","both"])
    if y_axis_style=="none":
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)
    
    sns.despine(left=True, bottom=True)
    ax.spines['bottom'].set_visible(True)
    plt.tight_layout()
    if save_name: fig.savefig(save_name, dpi=300, transparent=True)
    return fig, ax


def get_Dlims(output_list, wetsize=True):
    D_min = np.inf
    D_max = 0.
    for output in output_list:
        for part_id in output.particle_population.ids:
            if wetsize:
                D = output.particle_population.get_particle(part_id).get_Dwet()
            else:
                D = output.particle_population.get_particle(part_id).get_Ddry()
            D_min = min([D,D_min])
            D_max = max([D,D_max])
    return D_min, D_max
    
from ambrs.partmc_dq_merge_start import retrieve_model_state as retrieve_partmc
from ambrs.mam4 import retrieve_model_state as retrieve_mam4
# from ..partmc import retrieve_model_state as retrieve_partmc
# from ..mam4 import retrieve_model_state as retrieve_mam4

# scenario_names = ['001','005','007']
def plot_ensemble_state(
        partmc_dir, mam4_dir, scenario_names, ensemble, 
        timestep=-1, repeat_num=1, species_modifications={},
        N_bins=30, D_min=None, D_max=None,
        as_bar=True, normalize=True, wetsize=True, 
        pdf_method = 'hist'):
    
    
    
    # assemble output lists
    partmc_outputs = []
    mam4_outputs = []
    for scenario_name in scenario_names:
        partmc_output = retrieve_partmc( 
            scenario_name = scenario_name,
            scenario = ensemble.member(int(scenario_name)), 
            timestep = timestep, 
            repeat_num = repeat_num, 
            species_modifications=species_modifications,
            ensemble_output_dir = partmc_dir)
        partmc_outputs.append(partmc_output)
        
        mam4_output = retrieve_mam4( 
            scenario_name = scenario_name,
            scenario = ensemble.member(int(scenario_name)), 
            timestep = timestep,  # fixme: need to sync up timesteps
            repeat_num = repeat_num, # fixme: this isn't actually needed for MAM4, but kept it for consistency
            species_modifications=species_modifications,
            ensemble_output_dir = mam4_dir)        
        mam4_outputs.append(mam4_output)
    
    
    D_min_overall, D_max_overall = get_Dlims(partmc_outputs + mam4_outputs,wetsize=True)
    if not D_min: # is None
        D_min = D_min_overall
    if not D_max:
        D_max = D_max_overall
    df_partmc = build_sizedist_df(
            partmc_outputs, group_type='scenario', # could also be timesteps
            wetsize = wetsize, normalize=normalize, method = pdf_method, 
            N_bins = N_bins, D_min = D_min, D_max = D_max)
    
    df_mam4 = build_sizedist_df(
            mam4_outputs, group_type='scenario', # could also be timesteps
            wetsize = wetsize, normalize=normalize, method = 'hist', 
            N_bins = N_bins, D_min = D_min, D_max = D_max)
    
    plot_ridge(
        df_partmc, # data frame containing size distribution details
        dfB=df_mam4, # optional second line
        user_colors=None, 
        log_x=True, show_row_label=True, vline_x=0.5, 
        save_name=None,
        # blank_rows=[], # where to add blanks
        # N_blanks = 0, # N blank rows per blank
        xlim = None,
        ylim = None,
        axis_fontsize=12, tick_fontsize=10, row_label_fontsize=10,
        axis_font="Arial", tick_font="Arial", row_label_font="Arial",
        height=0.6, aspect=10, outline_color='w',
        critical_diams=None,               # array-like, one cutoff per plotted row (same order as columns)
        # normalize_mode="sum_dx",           # 'sum_dx' (normalize by sum(w)*Δln x), 'sum' (simple sum), or None
        x_max_allowed=40.0,
        line_kwargs=None,                  # for 2nd model: dict(lw=2, ls='--', alpha=1.0)
        lighten_alpha=0.3,                 # how much to blend fill toward white
        as_bar=as_bar
    )

def plot_ensemble_dots(
        partmc_dir, mam4_dir, scenario_names, ensemble,
        varname, var_config,
        timestep=1, repeat_num=1, species_modifications={},
        as_ccn=True, as_optics=True,
        figsize=(12,5), dpi=300, save_name=None,
        # pass-through to plot_dots
        color_map=None, marker='o', size=50, alpha=0.8, edgecolor='black'):
    """
    Build CCN/optics comparison DataFrames for an ensemble and draw dot plots.

    Retrieves PartMC and MAM4 outputs for the listed scenarios, builds
    comparison DataFrames via `build_comparison_dfs`, and plots CCN (left)
    and optics (right) dot panels using `plot_dots`.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Retrieve outputs
    partmc_outputs = []
    mam4_outputs = []
    for scenario_name in scenario_names:
        partmc_output = retrieve_partmc(
            scenario_name=scenario_name,
            scenario=ensemble.member(int(scenario_name)),
            timestep=timestep,
            repeat_num=repeat_num,
            species_modifications=species_modifications,
            ensemble_output_dir=partmc_dir
        )
        partmc_outputs.append(partmc_output)

        mam4_output = retrieve_mam4(
            scenario_name=scenario_name,
            scenario=ensemble.member(int(scenario_name)),
            timestep=timestep,
            repeat_num=repeat_num,
            species_modifications=species_modifications,
            ensemble_output_dir=mam4_dir
        )
        mam4_outputs.append(mam4_output)

    # Build comparison dfs using var_config
    df = build_comparison_dfs(
        varname, var_config, partmc_outputs, mam4_outputs, scenario_names=scenario_names
    )
    
    # Prepare figure
    n_panels = 1 if (as_ccn and not as_optics) or (as_optics and not as_ccn) else 2
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    gvals = df['g'].unique().tolist()
    local_cmap = color_map or dict(zip(gvals, sns.color_palette(n_colors=len(gvals))))
    plot_dots(df, ax=ax, color_map=local_cmap, marker=marker,
              size=size, alpha=alpha, edgecolor=edgecolor)
    
    plt.tight_layout()
    if save_name:
        fig.savefig(save_name, dpi=dpi, transparent=True)
    
    return fig, ax, df

    
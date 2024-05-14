import numpy as np
import matplotlib.pyplot as plt
from paper_plots.paper1_weight_sharing.box_plot import draw_box_plot


def format_with_SI_postfix(x, decimals=2):
    postfixes = [(1e9, "b"), (1e6, "M"), (1e3, "k")]
    postfix = ""
    for v,p in postfixes:
        if x >= v:
            x/= v
            postfix = p
            break
    x = np.round(x, decimals)
    x = str(x)
    if "." in x:
        x = str(x).rstrip("0").rstrip(".")
    return x + postfix

def plot_df(ax, df, x, y, experiment, ls_dict=None, label_dict=None, color_dict=None, marker_dict=None, lw_dict=None, shade_range=False, experiment_order=None, boxplots=False, **kwargs):
    if experiment_order is None:
        experiment_order = sorted(df[experiment].unique())
    ls_dict = ls_dict or dict()
    label_dict = label_dict or dict()
    color_dict = color_dict or dict()
    marker_dict = marker_dict or dict()
    lw_dict = lw_dict or dict()

    for e in experiment_order:
        pivot = df[df[experiment] == e].groupby(x)[y].agg(['mean', 'min', 'max']).reset_index()
        if len(pivot) == 0:
            continue
        ax.plot(pivot[x].values,
                pivot['mean'].values,
                color=kwargs.get('color', color_dict.get(e, 'k')),
                label=kwargs.get('label', label_dict.get(e)),
                ls=kwargs.get('ls', ls_dict.get(e, '-')),
                marker=kwargs.get('marker', marker_dict.get(e, 'o')),
                lw=kwargs.get('lw', lw_dict.get(e)),
                **{k:v for k,v in kwargs.items() if k not in ["ls", "lw", "color", "label", "marker"]}
                )
        if boxplots:
            for x_data in pivot[x].values:
                y_data = df[df[experiment] == e].groupby(x)[y].get_group(x_data).to_list()
                draw_box_plot(ax, x_data, y_data, width=1.15, color=kwargs.get('color', color_dict.get(e, 'k')),
                              alpha=0.6, scale='log', center_line='mean', alpha_outlier=0.2,
                              marker=kwargs.get('marker', marker_dict.get(e, 'o')))
        if shade_range:
            ax.fill_between(pivot[x], pivot['min'], pivot['max'], alpha=0.4, color=color_dict[e])



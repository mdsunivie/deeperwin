import matplotlib.pyplot as plt
import numpy as np


def plot_waterfall(y_values, label_start, labels_delta,
                   label_end="Total",
                   color_delta='lightblue',
                   color_summary='C0',
                   summary_points=None,
                   show_values=True,
                   value_format_string="{:.1f}",
                   bar_width=0.8,
                   label_rotation=0,
                   ax=None,
                   y_err=None,
                   ylim=None,
                   value_position='top',
                   horizontal=False,
                   textcolors=None,
                   labeloffset=0.1,
                   label_in_bar_thr=0.15,
                   x_values=None):
    y_delta = np.diff(y_values)
    assert len(y_delta) == len(labels_delta)
    ax = ax or plt.gca()
    summary_points = summary_points or []
    summary_points.append((len(y_values) - 1, label_end))

    summary_indices = np.array([s[0] for s in summary_points])
    y_values = np.insert(y_values, summary_indices, [y_values[s[0]] for s in summary_points])
    x_labels = np.insert(np.array([label_start] + labels_delta, dtype=object), summary_indices + 1,
                         [s[1] for s in summary_points])

    is_summary = np.zeros_like(y_values, dtype=bool)
    is_summary[0] = True
    for i, s in enumerate(summary_points):
        is_summary[s[0] + i + 1] = True
    y_bottom = np.array([0 if is_sum else y_values[i - 1] for i, is_sum in enumerate(is_summary)])
    if x_values is None:
        x_values = np.arange(len(y_values))

    # color_summary = color_summary if type(color_summary) is str else np.array(color_summary)[is_summary]

    if horizontal:
        ax.barh(x_values[is_summary],
                 y_values[is_summary],
                 bar_width,
                 color=color_summary,
                 zorder=3)
        ax.barh(x_values[~is_summary],
                 y_values[~is_summary] - y_bottom[~is_summary],
                 bar_width,
                 left=y_bottom[~is_summary],
                 color=color_delta,
                 zorder=3)
    else:
        ax.bar(x_values[is_summary],
                 y_values[is_summary],
                 bar_width,
                 color=color_summary,
                 zorder=3)
        ax.bah(x_values[~is_summary],
                 y_values[~is_summary] - y_bottom[~is_summary],
                 bar_width,
                 bottom=y_bottom[~is_summary],
                 color=color_delta,
                 zorder=3)

    if ylim:
        if horizontal:
            ax.set_xlim(ylim)
        else:
            ax.set_ylim(ylim)

    if show_values:
        textcolors = textcolors or ['k' for _ in x_values]
        for x, y, y_b, color in zip(x_values, y_values, y_bottom, textcolors):
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(y_b):
                fits_in_center = (abs(y - y_b) / abs(max(y_values) - min(y_values))) > label_in_bar_thr
                if (value_position == 'top') or (not fits_in_center):
                    y_label_pos = max(y, y_b) + (labeloffset if horizontal else 0.0)
                    if horizontal:
                        va, ha = 'center', 'left'
                    else:
                        va, ha = 'bottom', 'center'
                elif value_position == 'center':
                    y_label_pos = (y + y_b) / 2
                    va,ha = 'center', 'center'
                else:
                    raise ValueError(f"Unknown position of value: {value_position}")
                if ylim:
                    if (y_label_pos < min(ylim)) or (y_label_pos > max(ylim)):
                        continue
                kwargs = dict(x=x, y=y_label_pos, s=value_format_string.format(y - y_b), ha=ha, va=va, color=color)
                if horizontal:
                    kwargs['x'], kwargs['y'] = kwargs['y'], kwargs['x']
                ax.text(**kwargs)

    for i, y in enumerate(y_values[:-1]):
        x_connector, y_connector = [i + 0.5 * bar_width, i + 1 - 0.5 * bar_width], [y, y]
        if horizontal:
            x_connector, y_connector = y_connector, x_connector
        ax.plot(x_connector, y_connector, color='dimgray', lw=1, ls='--')

    if y_err:
        y_err = np.insert(y_err, summary_indices, [y_err[i] for i in summary_indices])
        kwargs = dict(x=x_values[:-1] + 0.5, y=y_values[:-1], yerr=y_err[:-1], color='k', capsize=2, marker='None', ls='None',
                    zorder=4, alpha=0.5)
        if horizontal:
            kwargs['xerr'] = kwargs.pop('yerr')
            kwargs['x'], kwargs['y'] = kwargs['y'], kwargs['x']
        ax.errorbar(**kwargs)

    if horizontal:
        ax.set_yticks(x_values)
        ax.set_yticklabels(x_labels, rotation=label_rotation)
        ax.invert_yaxis()
    else:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=label_rotation)



if __name__ == '__main__':
    energies = [4, 3.8, 3.6, 1.5, 1.6, 1, 0.9]
    errors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    plt.close("all")
    fig, ax = plt.subplots(1, 1, dpi=100, figsize=(14, 8))
    plot_waterfall(energies,
                   label_start="FermNet published",
                   labels_delta=["Isotropic envelopes", "DPE Hyperparams", "SchNet-like embedding",
                                 "Local input features", "Full Determinant",
                                 "Initialization"],
                   summary_points=[(4, 'Improved architecture')],
                   y_err=errors,
                   show_values=True,
                   horizontal=True,
                   value_position='center'
                   )
    plt.tight_layout()
    # ax.set_ylim([None, 5.0])
    # ax.grid(axis='y', alpha=0.5, zorder=-1)

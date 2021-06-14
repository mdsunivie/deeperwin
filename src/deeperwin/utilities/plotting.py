import numpy as np
from matplotlib import pyplot as plt


def imshowDataFrame(df, title=None, xlabel=None, ylabel=None, format_string_x = None, format_string_y=None, format_string_fields=None, axis=None, clim=None):
    """
    Plot a pandas pivot as colormap/imshow.

    Args:
        df (pd.DataFrame): Data to be plotted
        title (str): Chart title
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        format_string_x (str): Format-string for x-ticks
        format_string_y (str): Format-string for y-ticks
        format_string_fields (str): Format string for data values to be shown. If None, data will only be color coded and values will not be written as text
        axis (Axis): matplotlib.axis to plot; If None, defaults to current axis (plt.gca())
        clim (list): List of 2 elements; color limits for imshow

    Returns:
        None
    """
    if axis is None:
        axis = plt.gca()
    axis.imshow(df.values, origin='lower', clim=clim)
    if format_string_fields is not None:
        addTextLabelsToImshow(df.values, color='white', format_string=format_string_fields, axis=axis)

    xticks = list(df)
    if format_string_x is not None:
        xticks = list(map(format_string_x.format, xticks))
    yticks = list(df.index)
    if format_string_y is not None:
        yticks = list(map(format_string_y.format, yticks))

    axis.set_xticks(np.arange(len(xticks)))
    axis.set_xticklabels(xticks)
    axis.set_yticks(np.arange(len(yticks)))
    axis.set_yticklabels(yticks)
    if title is not None:
        axis.set_title(title)
    if xlabel is None:
        xlabel = df.columns.name
    if ylabel is None:
        ylabel = df.index.name
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def get_extent_for_imshow(x_values, y_values, origin='upper'):
    """
    Get the correct limits to be passed to the 'extent' keyword of matplotlib.pyplot.imshow.

    Assumes the the passed data is given on an equidistant grid to add proper padding to all edges
    Args:
        x_values (np.array): x-values of the data
        y_values (np.array): y-values of the data
        origin (str): 'upper' or 'lower'; must match the origin keyword passed to imshow

    Returns:
        (list): List of 4 elements denoting x_min, x_max, y_min, y_max

    """
    dx = x_values[1]-x_values[0]
    dy = y_values[1]-y_values[0]
    if origin == 'lower':
        return (x_values[0]-0.5*dx, x_values[-1]+0.5*dx, y_values[0]-0.5*dy, y_values[-1]+0.5*dy)
    elif origin == 'upper':
        return (x_values[0]-0.5*dx, x_values[-1]+0.5*dx, y_values[-1]+0.5*dy, y_values[0]-0.5*dy)
    else:
        raise ValueError("Origin must be 'lower' or 'upper'")


def addTextLabelsToImshow(values, format_string='{:.2f}', color='black', axis=None):
    """
    Put text labels on each 'pixel' of a color coded imshow plot to denote the values.

    Args:
        values (np.array): 2D numpy array containing the values passed to imshow
        format_string (str): python format string to be used for formatting the labels (e.g. '{:.2f}')
        color (str): Color of the text labels
        axis: Matplotlib axis to plot it on. If set to None, current axis (plt.gca()) is used

    Returns:
        None
    """
    if axis is None:
        axis = plt.gca()
    values = np.array(values)
    xx, yy = np.meshgrid(np.arange(values.shape[1]),np.arange(values.shape[0]))
    labels = [format_string.format(v) for v in np.array(values).flatten()]
    for x,y,l in zip(xx.flatten(), yy.flatten(),labels):
        axis.text(x, y, l, ha='center', va='center', color=color)


def build_tabbed_colormap(n_colors=None, c_start = 0.3, c_end = 0.8):
    """
    Build a colormap that contains shades of given colors, to visually group lines in plots together.

    Examples:
        This returns an array of 5 colors: light-green, dark-green, light-blue, medium-blue, dark-blue
        >>> build_tabbed_colormap([('green', 2), ('blue', 3)])

    Args:
        n_colors (list of tuples): Each tuple contains 2 entries: Name of the base color and number of shades of this color
        c_start (float): Lightness of the first color
        c_end (float): Lightness of the last color

    Returns:
        (np.array): Array of shape [Nx3] where N is the total number of colors and 4 contains the RGBA components of each color
    """
    if n_colors is None:
        n_colors = [('purples', 3), ('blues', 3), ('greens', 3), ('oranges', 3), ('reds', 3)]
    colors = []
    for c in n_colors:
        color_name = c[0] if c[0].endswith('s') else c[0] + 's'
        color_name = color_name.capitalize()
        x = np.linspace(c_start, c_end, c[1])
        colors.append(plt.cm.get_cmap(color_name)(x))
    return np.concatenate(colors, axis=0)


def plotGroupedBarsFromDF(df, **kwargs):
    """
    Plot a pandas pivot as a grouped bar chart.

    Uses the index as groups (i.e. x-axis) and the columns as categories (i.e. legend/colors)

    Args:
        df (pd.DataFrame): Pandas dataframe, typically obtained by pd.pivot
        **kwargs: kwargs to be passed to plotGroupedBars

    Returns:
        None
    """
    plotGroupedBars(df.values, df.index, list(df), **kwargs)


def plotGroupedBars(y, group_labels=None, bar_labels=None, yerr=None, spacing=0.2, colors=None, show_legend=True, axis = None, **bar_args):
    """
    Creates a grouped/butted bar plot using matplotlib from given 2D data.

    Args:
        y (np.array): Data to be plotted. Items that should appear in the same group (with different color) should be in the same row
        group_labels (str): Labels for the groups (i.e. x-axis labels)
        bar_labels (str): Labels per category (i.e. color legend)
        yerr (np.array): Array of same shape as y, containing data to be used for errorbars
        spacing (float): Relative spacing between the bars
        colors (list or None): Colors to be used for each category; Use None for matplotlib default colors
        show_legend (bool): Whether to show a legend
        axis (Axis): Axis to plot on. Set to None, to use current axis (plt.gca())
        **bar_args: Additional kwargs to be passed to axis.bar()

    Returns:
        None
    """
    if axis is None:
        axis = plt.gca()
    n_groups, n_bars = y.shape
    width = (1-spacing) / n_bars

    for i in range(n_bars):
        x = np.arange(n_groups) - width*n_bars*0.5 + (i+0.5)*width
        if yerr is None:
            axis.bar(x, y[:,i], width=width, color=None if colors is None else colors[i], **bar_args)
        else:
            axis.bar(x, y[:,i], width=width, yerr=yerr[:,i], color=None if colors is None else colors[i], **bar_args)
    if group_labels is not None:
        axis.set_xticks(np.arange(n_groups))
        axis.set_xticklabels(group_labels)
    if (bar_labels is not None) and show_legend:
        axis.legend(bar_labels)


def autoscaleIgnoreOutliers(ax, quantile=0.05, margin=0.1):
    """
    Automatically determine min and max for the y-axis and scale the plot accordingly, but ignore rare outliers that would hijack the scaling.

    Args:
        ax (Axis): matplotlib axis to autoscale
        quantile (float): Quantile which is considered to be an outlier (0: nothing is an outlier; 1: everything is an outlier)
        margin (float): Additional margin to add on ymin and ymax (relative to determined scale)

    Returns:
        None
    """
    ymax = -np.inf
    ymin = np.inf
    for l in ax.lines:
        ymin = min(np.quantile(l._y, quantile), ymin)
        ymax = max(np.quantile(l._y, 1-quantile), ymax)
    scale = ymax - ymin
    ax.set_ylim(ymin - margin*scale, ymax + margin*scale)

if __name__ == '__main__':
    build_tabbed_colormap([('green', 2), ('blue', 3)])

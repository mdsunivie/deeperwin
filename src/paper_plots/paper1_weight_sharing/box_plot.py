import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_box_plot(ax, x, y, width, color, alpha, scale='lin', center_line='median', marker='o', alpha_outlier=None, draw_whiskers=True, draw_outliers=True):
    alpha_outlier = alpha_outlier or alpha
    if scale == 'lin':
        x_left = x - width/2
        x_right = x + width/2
        x_whisker_left = x - width/4
        x_whisker_right = x + width / 4
    elif scale == 'log':
        x_left = x / width
        x_right = x * width
        x_whisker_left = x / np.sqrt(width)
        x_whisker_right = x * np.sqrt(width)

    y = np.array(y)
    lower_q = np.quantile(y, 0.25)
    upper_q = np.quantile(y, 0.75)
    iqr = upper_q - lower_q # inter-quartile range
    is_outlier = np.logical_or(y > (upper_q + 1.5 * iqr), y < (lower_q - 1.5 * iqr))
    n_outliers = np.sum(is_outlier)
    upper_whisker = np.max(y[~is_outlier])
    lower_whisker = np.min(y[~is_outlier])
    if center_line == 'median':
        average = np.median(y)
    elif center_line == 'mean':
        average = np.mean(y)

    rect = Rectangle((x_left, lower_q), x_right - x_left, iqr, color=color, alpha=alpha)
    ax.add_patch(rect)

    if center_line != 'none':
        ax.plot([x_left, x_right], [average, average], color=color, alpha=np.sqrt(alpha))
    if draw_whiskers:
        ax.plot([x, x], [upper_q, upper_whisker], color=color, alpha=alpha)
        ax.plot([x, x], [lower_q, lower_whisker], color=color, alpha=alpha)
        ax.plot([x_whisker_left, x_whisker_right], [upper_whisker, upper_whisker], color=color, alpha=alpha)
        ax.plot([x_whisker_left, x_whisker_right], [lower_whisker, lower_whisker], color=color, alpha=alpha)
    if draw_outliers and n_outliers > 0:
        ax.scatter(np.ones(n_outliers) * x, y[is_outlier], color=color, alpha=alpha_outlier, edgecolor='None', marker=marker)

if __name__ == '__main__':
    plt.close("all")
    plt.figure(dpi=200)

    x_values = np.logspace(1, 5, 5)
    y_values = [np.random.normal(np.log10(x_), 1.0, [50]) for x_ in x_values]
    means = [np.mean(y_) for y_ in y_values]


    plt.semilogx(x_values, means)
    for x, y in zip(x_values, y_values):
        draw_box_plot(plt.gca(), x, y, 1.5, 'C0', 0.2, scale='log', marker='^')


    plt.grid(alpha=0.5)

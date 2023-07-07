import numpy as np
from typing import Literal

def _format_with_SI_postfix(x, decimals=2):
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

def set_ticks_with_SI_postfix(ax, axis: Literal["x", "y"], ticks=None):
    if axis == 'x':
        axis = ax.xaxis
    elif axis == 'y':
        axis = ax.yaxis
    if ticks:
        axis.set_ticks(ticks)
    axis.set_major_formatter(lambda x, pos: _format_with_SI_postfix(x))
# -*- coding: utf-8 -*-

__all__ = [
    "plot_CDF_with_bounds",
    "plot_ECDF",
    "add_arrows_for_limits",
]

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .utils import unbinned_empirical_cdf

log = logging.getLogger(__name__)


def plot_CDF_with_bounds(
    bins_mid, median, lower, upper, ax=None, alpha=0.3, last_length=0.1, **kwargs
):
    """
    Convenience function to elegantly plot a cumulative distribution
    function and the confidence bounds around it.
    This function is different to `plot_ECDF()` in that it plots a
    _binned_ CDF, and the user should provide an array of the middle
    points of the bins.
    """

    if ax is None:
        ax = plt.gca()

    (artist,) = ax.plot(bins_mid, median, drawstyle="steps-mid", **kwargs)

    # Add the first and last line to make the plot look better
    _add_first_and_last_lines(
        x=bins_mid, y=median, ax=ax, artist=artist, last_length=last_length
    )

    ax.fill_between(
        bins_mid,
        lower,
        upper,
        step="mid",
        color=plt.getp(artist, "color"),
        alpha=alpha,
        zorder=plt.getp(artist, "zorder") - 1,
    )
    ax.plot(
        bins_mid,
        lower,
        drawstyle="steps-mid",
        lw=plt.getp(artist, "linewidth") / 2,
        c=plt.getp(artist, "color"),
        zorder=plt.getp(artist, "zorder") - 1,
    )
    ax.plot(
        bins_mid,
        upper,
        drawstyle="steps-mid",
        lw=plt.getp(artist, "linewidth") / 2,
        c=plt.getp(artist, "color"),
        zorder=plt.getp(artist, "zorder") - 1,
    )
    return


def plot_ECDF(sample, ax=None, last_length=0.1, **kwargs):
    """
    Convenience function to elegantly plot an empirical cumulative
    distribution function.
    """

    if ax is None:
        ax = plt.gca()

    x, ECDF = unbinned_empirical_cdf(sample)
    (artist,) = ax.plot(x, ECDF, drawstyle="steps-post", **kwargs)

    # Add the first and last line to make the plot look better
    _add_first_and_last_lines(
        x=x, y=ECDF, ax=ax, artist=artist, last_length=last_length
    )
    return


def _add_first_and_last_lines(x, y, last_length=0.1, ax=None, artist=None):
    """
    Add a beginning and final line to make an empirical CDF plotted
    with drawstyle="steps-post" look better.

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data
    last_length : float, optional
        Length of the last line in percent of the data span (default 0.1)
    ax : None, optional
        ax on which to add the lines
    artist : None, optional
        artist used to copy the properties from
    """

    if ax is None:
        ax = plt.gca()

    if artist is not None:
        kwargs = {
            "color": plt.getp(artist, "color"),
            "linestyle": plt.getp(artist, "linestyle"),
            "linewidth": plt.getp(artist, "linewidth"),
            "zorder": plt.getp(artist, "zorder"),
            "alpha": plt.getp(artist, "alpha"),
        }
    else:
        kwargs = {}

    ymin = y[0]
    ymax = y[-1]
    xmin = x[0]
    xmax = x[-1]

    bottom_line = Line2D([xmin, xmin], [0, ymin], **kwargs)

    top_line = Line2D(
        [xmax, xmax + last_length * (xmax - xmin)], [ymax, ymax], **kwargs
    )
    ax.add_line(bottom_line)
    ax.add_line(top_line)
    return


def add_arrows_for_limits(
    lim_val, lim_type, ax=None, loc="bottom", logscale=False, arrow_size=None, **kwargs
):
    """
    Add arrows to a plot to represent limits. This function is meant to
    be used on a CDF plot.

    Parameters
    ----------
    lim_val : float, int or array-like
        Value of the limit (where the arrow will start).
    lim_type : str
        Type of limit ("upper" or "lower").
    ax : axes, optional, Default `None`
        Matplotlib axes, if `None` will use `plt.gca()`.
    loc : str, optional
        "top" or "bottom".
    logscale : bool, optional, Default `True`
        If the x-axis is in log scale.
    arrow_size : None, optional
        Size of the arrow.
    **kwargs
        Any additional arguments to pass to `Line2D` (e.g. for specifying
        the color).
    """

    lim_val = np.atleast_1d(lim_val)

    if ax is None:
        ax = plt.gca()

    if lim_type not in ["upper", "lower"]:
        raise ValueError("lim_type must be 'upper' or 'lower'")

    xmin, xmax = ax.get_xlim()

    if logscale:
        if arrow_size is None:
            if lim_type == "upper":
                arrow_size = 0.4
            elif lim_type == "lower":
                arrow_size = 0.8
        arrow_length = lim_val * arrow_size
    else:
        if arrow_size is None:
            arrow_size = 0.02
        arrow_length = (xmax - xmin) * arrow_size * np.ones(len(lim_val))

    if loc == "bottom":
        y_tail_start = 0.01
        y_tail_stop = 0.03
    elif loc == "top":
        y_tail_start = 0.97
        y_tail_stop = 0.99
    else:
        raise ValueError("loc must be 'top' or 'bottom'")

    sign = -1 if lim_type == "upper" else 1

    for i, lim in enumerate(lim_val):
        # the xaxis transform returns a blended transform that allow to use data coordinates
        # for the x axis and axes coordinates for the y axis
        tail = Line2D(
            xdata=[lim, lim],
            ydata=[y_tail_start, y_tail_stop],
            transform=ax.get_xaxis_transform(),
            **kwargs
        )
        ax.add_artist(tail)
        ax.annotate(
            "",
            xy=(lim + sign * arrow_length[i], 0.5 * (y_tail_start + y_tail_stop)),
            xycoords=ax.get_xaxis_transform(),
            xytext=(lim, 0.5 * (y_tail_start + y_tail_stop)),
            textcoords=ax.get_xaxis_transform(),
            arrowprops=dict(arrowstyle="-|>", color=plt.getp(tail, "color")),
        )
    return

import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)


def show_flux_integration_bounds(
    x,
    y,
    wvlg_min,
    wvlg_max,
    line_flux,
    line_err,
    continuum=0,
    ax1D=None,
    ax2D=None,
    facecolor="C0",
    **kwargs,
):
    """
    Visually represent the integration bounds over which the flux was calculated with
    vertical lines. Can also be added to the 2D plot by specifying a ax2D.
    """
    if ax2D is not None:
        ax2D.axvline(x.searchsorted(wvlg_min), **kwargs)
        ax2D.axvline(x.searchsorted(wvlg_max), **kwargs)
        ax2D.axvline(x.searchsorted(wvlg_min), **kwargs)
        ax2D.axvline(x.searchsorted(wvlg_max), **kwargs)

    if ax1D is None:
        ax1D = plt.gca()

    ax1D.axvline(wvlg_min, **kwargs)
    ax1D.axvline(wvlg_max, **kwargs)
    ax1D.fill_between(
        x,
        y,
        y2=continuum,
        where=((x >= wvlg_min) & (x <= wvlg_max)),
        label=r"$\rm Flux =$"
        + f" {line_flux:.1e} "
        + r"$\pm$"
        + f" {line_err:.1e} "
        + r"$\rm erg/s/cm^{{2}}$",
        facecolor=facecolor,
        hatch="|",
    )
    return


def plot_spectrum(wvlg, flux, error=None, ax=None, **kwargs):
    """Summary

    Parameters
    ----------
    ax : None, optional
        Description
    color : str, optional
        Description
    **kwargs
        Description
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(
        wvlg,
        flux,
        drawstyle="steps-mid",
        color=kwargs.get("color", "k"),
        lw=kwargs.get("lw", 1),
        **kwargs,
    )
    if error is not None:
        ax.fill_between(
            wvlg.value,
            error.value,
            color=kwargs.get("color", "k"),
            alpha=kwargs.get("alpha", 0.2),
            step="mid",
            **kwargs,
        )


def show_regions(regions, ax=None, **kwargs):
    """Summary

    Parameters
    ----------
    regions : TYPE
        List of 2-tuple containing the lower and upper bounds of the
        region.
    ax : None, optional
        Description
    """
    if ax is None:
        ax = plt.gca()
    for i, reg in enumerate(regions):
        ax.axvline(reg[0].value, lw=kwargs.get("lw", 0.5))
        ax.axvline(reg[1].value, lw=kwargs.get("lw", 0.5))
        ax.axvspan(
            reg[0].value,
            reg[1].value,
            alpha=kwargs.get("alpha", 0.05),
            **kwargs,
        )
    return


def fig_resid(figsize=(10, 7)):
    fig = plt.figure(figsize=figsize)
    ax_center = fig.add_axes([0.0, 0.20, 0.90, 0.80])
    ax_residuals = fig.add_axes([0.0, 0.0, 0.90, 0.20], sharex=ax_center)
    ax_residuals_hist = fig.add_axes([0.90, 0.0, 0.1, 0.20], sharey=ax_residuals)

    ax_center.tick_params(labelbottom=False)
    ax_residuals_hist.tick_params(labelleft=False, labelbottom=False)
    axes = {
        "center": ax_center,
        "residuals": ax_residuals,
        "residuals_hist": ax_residuals_hist,
    }
    return fig, axes

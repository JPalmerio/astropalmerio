import logging
import matplotlib.pyplot as plt
import numpy as np

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


def plot_fit(
    wvlg, flux, error, model, spec_plot_kw={}, model_plot_kw={}, show_legend=True
):
    fig, axes = fig_resid()

    ax = axes["center"]
    ax_res = axes["residuals"]
    ax_resh = axes["residuals_hist"]

    plot_spectrum(wvlg, flux, error, ax=ax, **spec_plot_kw)
    plot_model(wvlg, model, ax=ax, **model_plot_kw)

    # Adjust y height of plot
    ymax = np.max(
        (
            np.median(flux).value + 5 * np.std(flux).value,
            2 * np.max(model).value - np.median(model).value,
        )
    )
    ax.set_ylim(ymax=ymax)
    if show_legend:
        ax.legend()


def plot_model(wvlg, model, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(wvlg, model, color=kwargs.pop("color", "C0"), **kwargs)


def plot_continuum(wvlg, flux, regions=None, ax=None, **kwargs):
    """Summary

    Parameters
    ----------
    wvlg : TYPE
        Description
    flux : TYPE
        Description
    regions : None, optional
        Description
    ax : None, optional
        Description
    **kwargs
        Description
    """
    if ax is None:
        ax = plt.gca()

    color = kwargs.pop("color", "C1")
    label = kwargs.pop("label", "Continuum fit")
    ax.plot(
        wvlg,
        flux,
        label=label,
        color=color,
        **kwargs,
    )

    if regions is not None:
        show_regions(
            ax=ax,
            regions=regions,
            color=color,
        )


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

    color = kwargs.pop("color", "k")
    lw = kwargs.pop("lw", 1)
    ax.plot(
        wvlg,
        flux,
        drawstyle="steps-mid",
        color=color,
        lw=lw,
        **kwargs,
    )
    if error is not None:
        ax.fill_between(
            wvlg.value,
            error.value,
            color=color,
            alpha=kwargs.pop("alpha", 0.2),
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

    alpha = kwargs.pop("alpha", 0.05)

    if ax is None:
        ax = plt.gca()
    for i, reg in enumerate(regions):
        ax.axvline(reg[0].value, lw=kwargs.get("lw", 0.5))
        ax.axvline(reg[1].value, lw=kwargs.get("lw", 0.5))
        ax.axvspan(
            reg[0].value,
            reg[1].value,
            alpha=alpha,
            **kwargs,
        )
    return


def set_standard_spectral_labels(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_ylabel(r"$\rm F_{\lambda}~[erg/s/cm^{2}/\AA$]")
    ax.set_xlabel(r"Wavelength [$\rm \AA$]")
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

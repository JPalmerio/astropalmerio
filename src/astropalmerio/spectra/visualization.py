# -*- coding: utf-8 -*-

__all__ = [
    "show_flux_integration_bounds",
    "plot_fit",
    "plot_model",
    "plot_continuum",
    "plot_spectrum",
    "show_regions",
    "set_standard_spectral_labels",
    "fig_resid",
]

import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from .utils import gaussian_fct

log = logging.getLogger(__name__)


def show_flux_integration_bounds(
    x,
    y,
    wvlg_min,
    wvlg_max,
    line_flux,
    line_unc,
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
        + f" {line_unc:.1e} "
        + r"$\rm erg/s/cm^{{2}}$",
        facecolor=facecolor,
        hatch="|",
    )
    return


def _plot_residuals(
    wvlg,
    residuals,
    ax,
    ax_hist=None,
    resid_hist_kw={},
    show_unit_gauss=True,
    show_legend=True,
    **kwargs,
):
    color = kwargs.pop("color", "k")
    ax.plot(
        wvlg,
        residuals,
        color=color,
        drawstyle=kwargs.pop("drawstyle", "steps-mid"),
        lw=kwargs.pop("lw", 1),
        **kwargs,
    )

    if ax_hist is not None:
        pvalue = shapiro(residuals).pvalue
        default_label = r"$p_{\rm value}$ = " + f"{pvalue:.3f}"

        ax_hist.hist(
            residuals,
            orientation="horizontal",
            density=True,
            color=resid_hist_kw.pop("color", color),
            alpha=resid_hist_kw.pop("alpha", 0.9),
            bins=resid_hist_kw.pop("bins", np.arange(-5, 5.5, 0.5)),
            label=resid_hist_kw.pop("label", default_label),
            **resid_hist_kw,
        )

        if show_unit_gauss:
            x = np.linspace(-4, 4, 100)
            ax_hist.plot(
                gaussian_fct(x=x, mean=0, stddev=1),
                x,
                color=color,
                lw=1,
            )
        if show_legend:
            ax_hist.legend()
    return


def plot_fit(
    wvlg,
    flux,
    uncertainty,
    model,
    residuals=None,
    fit_bounds=None,
    spec_plot_kw={},
    model_plot_kw={},
    resid_plot_kw={},
    show_legend=True,
):

    if residuals is not None:
        fig, axes = fig_resid()
        ax = axes["center"]
        ax_res = axes["residuals"]
        ax_resh = axes["residuals_hist"]
    else:
        fig, ax = plt.subplots()

    plot_spectrum(wvlg, flux, uncertainty=uncertainty, ax=ax, **spec_plot_kw)
    plot_model(wvlg, model, ax=ax, **model_plot_kw)

    # Adjust y height of plot
    ymax = np.max(
        (
            np.median(flux).value + 5 * np.std(flux).value,
            2 * np.max(model).value - np.median(model).value,
        )
    )

    ax.set_ylim(ymax=ymax)

    if residuals is not None:

        if fit_bounds is None:
            # plot all residuals with same color as spectrum
            _plot_residuals(
                wvlg,
                residuals=residuals,
                ax=ax_res,
                ax_hist=ax_resh,
                label=None,
                show_unit_gauss=True,
                lw=resid_plot_kw.pop("lw", spec_plot_kw.get("lw", 1)),
                color=resid_plot_kw.pop("color", spec_plot_kw.get("color")),
                resid_hist_kw=resid_plot_kw.pop("resid_hist_kw", {}),
                **resid_plot_kw,
            )
        else:
            # Plot all residuals with same color as spectrum
            # But also overplot region between fit bounds with same color as model
            # And calculate side histograms associated with this
            if fit_bounds[0] >= fit_bounds[1]:
                raise Exception(
                    "First element of fit_bounds should be smaller than second element"
                )

            inbounds = np.where((wvlg >= fit_bounds[0]) & (wvlg <= fit_bounds[1]))[0]
            wvlg_resid = wvlg[inbounds]
            residuals_fit = residuals[inbounds]

            _plot_residuals(
                wvlg,
                residuals=residuals,
                ax=ax_res,
                ax_hist=ax_resh,
                label=None,
                show_unit_gauss=True,
                lw=spec_plot_kw.get("lw", 1),
                color=spec_plot_kw.get("color", "k"),
                resid_hist_kw={"label": None},
                show_legend=False,
            )

            _plot_residuals(
                wvlg_resid,
                residuals=residuals_fit,
                ax=ax_res,
                ax_hist=ax_resh,
                show_unit_gauss=False,
                color=resid_plot_kw.pop("color", "C0"),
                resid_hist_kw=resid_plot_kw.pop("resid_hist_kw", {}),
                **resid_plot_kw,
            )

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


def plot_spectrum(wvlg, flux, uncertainty=None, ax=None, **kwargs):
    """ """
    if ax is None:
        ax = plt.gca()

    color = kwargs.pop("color", "k")
    ax.plot(
        wvlg,
        flux,
        drawstyle="steps-mid",
        color=color,
        lw=kwargs.pop("lw", 1),
        **kwargs,
    )
    if uncertainty is not None:
        ax.fill_between(
            wvlg.value,
            uncertainty.value,
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


def set_standard_spectral_labels(ax_x=None, ax_y=None):
    if ax_x is None and ax_y is None:
        ax_x = plt.gca()
        ax_y = ax_x
    elif ax_x is None:
        ax_x = ax_y
    elif ax_y is None:
        ax_y = ax_x

    ax_y.set_ylabel(r"$\rm F_{\lambda}~[erg/s/cm^{2}/\AA$]")
    ax_x.set_xlabel(r"Wavelength [$\rm \AA$]")
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

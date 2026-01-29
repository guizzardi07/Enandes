# modulos/plotting.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_timeseries_daily_grid(
    df: pd.DataFrame,
    ylabel: str = "Nivel",
    title: str | None = None,
    figsize=(12, 4),
) -> plt.Figure:
    """
    Plot con eje X adaptativo según el largo de la ventana temporal:
    - ventanas cortas: labels horarias (dd/mm + hora)
    - ventanas medias: labels diarias / semanales
    - ventanas largas: labels mensuales / anuales
    Además:
    - grilla mayor suave
    - grilla menor fina en X
    """
    fig, ax = plt.subplots(figsize=figsize)

    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col))

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Fecha")
    if title:
        ax.set_title(title)
    ax.legend()

    # --------- EJE X DINÁMICO SEGÚN RANGO ---------
    if df is not None and len(df.index) >= 2:
        idx = pd.DatetimeIndex(df.index).sort_values()
        span = idx.max() - idx.min()
        span_days = span / pd.Timedelta(days=1)

        if span_days <= 2:
            # ticks cada 6h (labels) + grilla cada 1h
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

        elif span_days <= 14:
            # ticks diarios + grilla cada 6h
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

        elif span_days <= 90:
            # ticks semanales + grilla diaria
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

        elif span_days <= 365 * 2:
            # ticks mensuales + grilla semanal
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

        else:
            # ticks anuales + grilla mensual
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

        # estética: sin rotación y un poco más chico
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)

    else:
        # fallback (por si df vacío o 1 punto)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    # --------- GRILLAS ---------
    ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.4)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.35)

    fig.tight_layout()
    return fig


def plot_scatter_obs_fit(
    y_obs: pd.Series,
    y_fit: pd.Series,
    title: str = "Obs vs Fit",
    figsize: tuple[float, float] = (2.75, 2.75),
    s: float = 10,
    alpha: float = 0.7,
    fontsize: int = 9,
    ticksize: int = 8,
    show_11: bool = True,
) -> plt.Figure:
    df = pd.concat([y_obs.rename("obs"), y_fit.rename("fit")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df["obs"].values, df["fit"].values, s=s, alpha=alpha)

    if show_11 and not df.empty:
        vmin = float(np.nanmin([df["obs"].min(), df["fit"].min()]))
        vmax = float(np.nanmax([df["obs"].max(), df["fit"].max()]))
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)

    ax.set_title(title, fontsize=fontsize + 1)
    ax.set_xlabel("Observado", fontsize=fontsize)
    ax.set_ylabel("Ajustado", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    return fig

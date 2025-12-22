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
    Plot con:
    - ticks mayores mensuales (labels)
    - grilla menor diaria en X (líneas verticales)
    """
    fig, ax = plt.subplots(figsize=figsize)

    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col))

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Fecha")
    if title:
        ax.set_title(title)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

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

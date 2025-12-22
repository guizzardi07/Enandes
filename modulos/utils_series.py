# modulos/utils_series.py
from __future__ import annotations

from typing import Dict
import pandas as pd


def apply_lag_shift_series(up: pd.Series, step: pd.Timedelta, lag_steps: int) -> pd.Series:
    """
    Alinea upstream hacia adelante: mueve timestamps +lag*step.
    (Tu comportamiento actual en _apply_lag_align)
    """
    lag_steps = int(lag_steps or 0)
    return up.shift(lag_steps, freq=step)


def apply_lags_for_plot(
    df: pd.DataFrame,
    obs_col: str,
    lag_map: Dict[str, int],
    step: pd.Timedelta,
    upstream_suffix: str = "",
) -> pd.DataFrame:
    """
    Devuelve DF listo para plot:
      - obs_col sin shift
      - upstream shift por lag_map
    """
    out = pd.DataFrame(index=df.index)
    out[obs_col] = df[obs_col]

    for col in df.columns:
        if col == obs_col:
            continue
        lag = int(lag_map.get(col, 0))
        out[f"{col}{upstream_suffix}"] = apply_lag_shift_series(df[col], step=step, lag_steps=lag)

    return out


# modulos/utils_time.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Tuple
import pandas as pd


def today_local() -> date:
    return datetime.now().date()

def to_dt_start(d: date) -> str:
    """construir_series_union espera timestart como string."""
    return datetime.combine(d, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")

def to_dt_end(d: date) -> datetime:
    """construir_series_union espera timeend como datetime."""
    return datetime.combine(d, datetime.max.time()).replace(microsecond=0)

def parse_step(step_adopt: str) -> pd.Timedelta:
    """
    Normaliza step (ej: 'h', '1h', 'day', '1D', '60min') a pd.Timedelta.
    """
    s = str(step_adopt).strip().lower()

    # aliases comunes
    if s in {"h", "1h", "hour", "hora"}:
        return pd.Timedelta(hours=1)
    if s in {"d", "1d", "day", "dia"}:
        return pd.Timedelta(days=1)

    try:
        return pd.to_timedelta(step_adopt)
    except Exception as e:
        raise ValueError(f"Step inválido: {step_adopt!r}") from e
    
def default_download_window(days: int = 365) -> Tuple[date, date]:
    end = today_local()
    start = end - timedelta(days=days)
    return start, end

def default_plot_window_from_index(idx: pd.DatetimeIndex, days: int = 90) -> Tuple[date, date]:
    if idx is None or len(idx) == 0:
        today = today_local()
        return today - timedelta(days=days), today
    idx_min = idx.min()
    idx_max = idx.max()
    if pd.isna(idx_min) or pd.isna(idx_max):
        today = today_local()
        return today - timedelta(days=days), today
    start = max(idx_min.date(), (idx_max - pd.Timedelta(days=days)).date())
    return start, idx_max.date()

"""
Pronóstico estacional mensual por persistencia y analogías.

Este script integra en una misma estructura:
1. Descarga o lectura de serie.
2. Limpieza de outliers.
3. Regularización temporal.
4. Agregado mensual.

5. Pronóstico por persistencia de cuantiles.
6. Pronóstico por analogías.

7. Gráfico de trazas análogas: meses previos + meses pronosticados.

Notas:
- La preparación de datos es compartida por ambos métodos.
- El método de analogías usa la serie transformada logarítmica estandarizada por mes.
- El gráfico de analogías muestra la traza objetivo y las N trazas seleccionadas.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from a5client import Crud, observacionesListToDataFrame
from dotenv import load_dotenv

# Configuración general

TIMEZONE = pytz.timezone("America/Argentina/Buenos_Aires")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MONTH_NAMES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

@dataclass
class StationConfig:
    """Parámetros de la estación y de la descarga desde A5."""

    nombre: str = "Puerto Pilcomayo"
    id_serie: int = 42293 # 42293   32025
    id_serie_hist: int = 6362   # 29998 42293  
    fecha_desde: str = "1980-01-01 01:00:00"
    fecha_hasta: str = "2026-05-08 09:00:00"
    variable: str = "valor"

@dataclass
class CleaningConfig:
    """Parámetros de limpieza y regularización de la serie."""

    limite_outliers: tuple[float, float] = (1, 8.0)
    intervalo: timedelta = timedelta(days=1)
    interpolation_limit: int = 1

@dataclass
class ForecastConfig:
    """Parámetros comunes para persistencia y analogías."""

    long_busqueda: int = 6
    long_prono: int = 4
    vent_resamp: str = "month"
    min_count_mensual: int = 25
    mes_select: Optional[int] = 2
    yr_select: Optional[int] = 2015
    plot: bool = True

@dataclass
class AnalogyConfig:
    """Parámetros específicos del método de analogías."""

    orden: str = "RMSE"
    orden_ascending: bool = True
    cantidad: int = 5
    variable_log: str = "LogVar"
    variable_transf: str = "LogVar_Est"
    eps_log: float = 1e-6


# Utilidades de fecha
def parse_datetime(date_string: str) -> datetime:
    """Convierte un string a datetime."""
    return datetime.strptime(date_string, DATE_FORMAT)

def add_months(year: int, month: int, n: int) -> tuple[int, int]:
    """Suma n meses a un par year/month."""
    month_zero_based = (month - 1) + n
    new_year = year + month_zero_based // 12
    new_month = month_zero_based % 12 + 1
    return new_year, new_month

def month_to_date(year: int, month: int, day: int = 15) -> pd.Timestamp:
    """Convierte year/month a fecha. Para gráficos mensuales se usa día 15."""
    return pd.Timestamp(year=int(year), month=int(month), day=day)

def add_date_column(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    """Agrega una columna de fecha mensual usando el día 15."""
    out = df.copy()
    out[date_col] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=15))
    return out

def infer_forecast_origin(today: Optional[datetime] = None) -> tuple[int, int]:
    """
    Define el último mes con datos suficientes para emitir pronóstico.

    Si el día del mes es mayor a 27, usa el mes actual; si no, usa el mes anterior.
    """
    fecha_emision = today or datetime.now()

    if fecha_emision.day > 27:
        return fecha_emision.year, fecha_emision.month

    if fecha_emision.month == 1:
        return fecha_emision.year - 1, 12

    return fecha_emision.year, fecha_emision.month - 1


# Descarga y preprocesamiento
def get_a5_client() -> Crud:
    """Crea el cliente de A5 leyendo A5_URL y A5_TOKEN desde .env."""
    load_dotenv()
    url = os.getenv("A5_URL")
    token = os.getenv("A5_TOKEN")

    if not url or not token:
        raise RuntimeError("Faltan A5_URL o A5_TOKEN en el archivo .env")

    return Crud(url=url, token=token)

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que el índice sea DatetimeIndex en TIMEZONE.

    Usa utc=True para poder recibir índices tz-aware o tz-naive sin romper.
    """
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    idx = idx.tz_convert(TIMEZONE)

    mask_valid = ~idx.isna()
    out = out.loc[mask_valid].copy()
    out.index = idx[mask_valid]
    return out.sort_index()

def download_a5_series(client: Crud, station: StationConfig) -> pd.DataFrame:
    """Descarga una serie de A5 y la convierte a DataFrame."""
    timestart = parse_datetime(station.fecha_desde)
    timeend = parse_datetime(station.fecha_hasta)

    raw = client.readSerie(station.id_serie, timestart, timeend)
    df = observacionesListToDataFrame(raw["observaciones"])

    if df.empty:
        raise ValueError(f"La serie {station.id_serie} no tiene datos")    

    raw_hist = client.readSerie(station.id_serie_hist, timestart, timeend)
    df_hist = observacionesListToDataFrame(raw_hist["observaciones"])

    if df_hist.empty:
        raise ValueError(f"La serie {station.id_serie_hist} no tiene datos")

    return ensure_datetime_index(df), ensure_datetime_index(df_hist)

def normalize_value_column(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Asegura que la columna principal exista.

    En algunas descargas la columna se llama 'valor'. Si value_col no existe y sí existe
    'valor', se renombra a value_col.
    """
    out = df.copy()
    if value_col not in out.columns and "valor" in out.columns:
        out = out.rename(columns={"valor": value_col})

    if value_col not in out.columns:
        raise KeyError(f"No existe la columna {value_col!r}")

    return out

def remove_outliers(
    df: pd.DataFrame,
    limits: tuple[float, float],
    column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reemplaza por NaN los valores fuera de limits.

    Devuelve:
    - outliers detectados
    - serie limpia
    """
    if column not in df.columns:
        raise KeyError(f"No existe la columna {column!r}")

    limit_inf, limit_sup = limits
    out = df.copy()

    mask_outlier = (out[column] < limit_inf) | (out[column] > limit_sup)
    outliers = out.loc[mask_outlier].copy()
    out.loc[mask_outlier, column] = np.nan

    logger.info("Outliers detectados: %s", len(outliers))
    logger.info("Límites usados: %.2f a %.2f", limit_inf, limit_sup)

    return outliers, out

def create_regular_index(
    start: Any,
    end: Any,
    freq: timedelta | str,
    timezone=TIMEZONE,
) -> pd.DatetimeIndex:
    """
    Crea un índice regular manteniendo la hora local original.

    Se genera primero un rango tz-naive y luego se localiza. Esto evita errores por
    horarios inexistentes durante cambios históricos de horario en Argentina.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)

    regular_index = pd.date_range(start=start_ts, end=end_ts, freq=freq)
    regular_index = regular_index.tz_localize(
        timezone,
        nonexistent="shift_forward",
        ambiguous="NaT",
    )
    return regular_index.dropna()

def regularize_series(
    df: pd.DataFrame,
    freq: timedelta | str,
    column: str,
    timestart: Optional[Any] = None,
    timeend: Optional[Any] = None,
    interpolate: bool = True,
    interpolation_limit: int = 1,
    daily_normalize: bool = True,
) -> pd.DataFrame:
    """Lleva la serie a paso regular e interpola huecos cortos."""

    if column not in df.columns:
        raise KeyError(f"No existe la columna {column!r}")

    data = ensure_datetime_index(df)

    if daily_normalize:
        data = data.copy()

        # Quitar timezone antes de normalizar evita NonExistentTimeError
        # por cambios históricos de huso horario.
        data.index = data.index.tz_localize(None).normalize()

        data = (
            data.groupby(data.index)
            .agg({column: "mean"})
            .sort_index()
        )

    start = timestart if timestart is not None else data.index.min()
    end = timeend if timeend is not None else data.index.max()

    if daily_normalize:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        if start.tzinfo is not None:
            start = start.tz_localize(None)
        if end.tzinfo is not None:
            end = end.tz_localize(None)

        start = start.normalize()
        end = end.normalize()

        regular_index = pd.date_range(
            start=start,
            end=end,
            freq=freq,
        )

    else:
        regular_index = create_regular_index(start, end, freq)

    out = pd.DataFrame(index=regular_index)
    out = out.join(data[[column]], how="left")

    if interpolate:
        out[column] = out[column].interpolate(
            method="time",
            limit=interpolation_limit,
            limit_direction="both",
        )

    return out.round(2)

def add_time_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables temporales usadas para agrupar y filtrar."""
    out = df.copy()
    out.insert(0, "year", out.index.year)
    out.insert(1, "month", out.index.month)
    out.insert(2, "day", out.index.day)
    out.insert(3, "yrDay", out.index.dayofyear)
    out.insert(4, "wkDay", out.index.isocalendar().week.astype(int))
    return out

def hyd_year(dts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Año hidrológico que arranca el 1/6: Jun-Dic -> año+1, Ene-May -> año."""
    dts = pd.to_datetime(dts)
    return np.where(dts.month >= 6, dts.year + 1, dts.year)

def merge_current_and_historical(
    current: pd.DataFrame,
    historical: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """
    Combina serie actual e histórica.

    Prioridad:
    1. Serie actual.
    2. Serie histórica para completar huecos.
    """
    cur = current[[value_col]].copy().rename(columns={value_col: "actual"})
    hist = historical[[value_col]].copy().rename(columns={value_col: "historica"})

    out = hist.join(cur, how="outer")
    out[value_col] = out["actual"].combine_first(out["historica"])

    return out[[value_col]].sort_index()

def get_last_complete_hyd_year(
    df: pd.DataFrame,
    value_col: str,
    min_valid_days: int = 300,
) -> int:
    """
    Devuelve el último año hidrológico cerrado con datos suficientes.
    """
    today = pd.Timestamp.now(tz=TIMEZONE)

    # Año hidrológico actual: si estamos entre junio y diciembre, es año+1.
    current_hy = today.year + 1 if today.month >= 6 else today.year

    valid_counts = (
        df.loc[df["hyd_year"] < current_hy]
        .groupby("hyd_year")[value_col]
        .count())
    
    valid_counts = valid_counts[valid_counts >= min_valid_days]

    if valid_counts.empty:
        raise ValueError("No hay años hidrológicos cerrados con datos suficientes")

    return int(valid_counts.index.max())

def compute_hyd_year_minimum_trend(
    df: pd.DataFrame,
    value_col: str,
    q: float = 0.02,
    min_valid_days: int = 300,
    rolling_window: int = 3,
) -> pd.DataFrame:
    """
    Calcula el percentil bajo anual y una tendencia suave mediante mediana móvil.
    """
    stats = (
        df.groupby("hyd_year")[value_col]
        .agg(
            n_valid="count",
            p02=lambda s: s.quantile(q),
        )
        .reset_index()
    )

    stats.loc[stats["n_valid"] < min_valid_days, "p02"] = np.nan

    stats["p02_suavizado"] = (
        stats["p02"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .median()
    )

    return stats

def adjust_series_to_current_hydraulic_level(
    df: pd.DataFrame,
    value_col: str,
    q: float = 0.02,
    min_valid_days: int = 300,
    rolling_window: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corrige la serie por desplazamiento vertical usando la tendencia suavizada
    del mínimo anual hidrológico.

    La referencia es el último año hidrológico cerrado con datos suficientes.
    """
    out = df.copy()

    out["hyd_year"] = hyd_year(out.index)

    stats = compute_hyd_year_minimum_trend(
        out,
        value_col=value_col,
        q=q,
        min_valid_days=min_valid_days,
        rolling_window=rolling_window,
    )

    ref_hy = get_last_complete_hyd_year(
        out,
        value_col=value_col,
        min_valid_days=min_valid_days,
    )

    ref_value = stats.loc[
        stats["hyd_year"] == ref_hy,
        "p02_suavizado"
    ].iloc[0]

    stats["hyd_year_ref"] = ref_hy
    stats["p02_ref"] = ref_value
    stats["offset"] = stats["p02_ref"] - stats["p02_suavizado"]

    out = out.merge(
        stats[["hyd_year", "p02", "p02_suavizado", "offset"]],
        on="hyd_year",
        how="left",
    )

    out.index = df.index
    out[f"{value_col}_corr"] = out[value_col] + out["offset"]

    return out, stats

def plot_hydrological_minimums(
    stats: pd.DataFrame,
    station_name: str,
    q: float = 0.02,
) -> None:
    """
    Grafica el percentil bajo anual y su tendencia suavizada.
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        stats["hyd_year"],
        stats["p02"],
        marker="o",
        linewidth=1.5,
        label=f"P{int(q*100)} anual",
    )

    ax.plot(
        stats["hyd_year"],
        stats["p02_suavizado"],
        marker="s",
        linewidth=3,
        label="Tendencia suavizada",
    )

    # Año de referencia
    if "hyd_year_ref" in stats.columns:
        ref_year = stats["hyd_year_ref"].iloc[0]

        ax.axvline(
            ref_year,
            linestyle="--",
            linewidth=1.5,
            label=f"Año referencia: {ref_year}",
        )

    ax.set_title(
        f"{station_name} - Evolución del mínimo hidráulico"
    )

    ax.set_xlabel("Año hidrológico")
    ax.set_ylabel("Nivel [m]")

    ax.grid(
        True,
        which="both",
        color="0.75",
        linestyle="-.",
        linewidth=0.4,
    )

    ax.legend(loc="best")

    # plt.show()
    # plt.close(fig)
    return fig

def prepare_regular_series(
    station: StationConfig,
    cleaning: CleaningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Descarga, combina serie actual e histórica, limpia, regulariza,
    corrige por mínimo hidráulico y agrega variables temporales.

    Devuelve:
    - serie regular corregida
    - outliers detectados
    - tabla de estadísticos por año hidrológico
    """
    client = get_a5_client()
    raw_series, raw_hist = download_a5_series(client, station)

    raw_series = normalize_value_column(raw_series, station.variable)
    raw_hist = normalize_value_column(raw_hist, station.variable)

    # 1. Combinar histórica + actual, priorizando la actual.
    combined_series = merge_current_and_historical(
        current=raw_series,
        historical=raw_hist,
        value_col=station.variable,
    )

    # 2. Limpiar outliers sobre la serie combinada original.
    outliers, clean_series = remove_outliers(
        combined_series,
        limits=cleaning.limite_outliers,
        column=station.variable,
    )

    # 3. Regularizar e interpolar huecos cortos.
    regular_series = regularize_series(
        clean_series,
        freq=cleaning.intervalo,
        column=station.variable,
        interpolation_limit=cleaning.interpolation_limit,
    )

    # print(combined_series)
    # print(clean_series)
    # print(regular_series)

    # fig, ax = plt.subplots(figsize=(15, 8))
    # ax.plot(combined_series.index, combined_series[station.variable], label="combined_series")
    # ax.scatter(clean_series.index, clean_series[station.variable], label="clean_series")
    # ax.plot(regular_series.index, regular_series[station.variable], label="regular_series")


    # ax.grid(True, which="both", color="0.75", linestyle="-.", linewidth=0.4)
    # ax.legend(loc="best")
    # plt.show()
    # quit()

    # 4. Corregir la serie regularizada al nivel hidráulico actual.
    corrected_series, hyd_stats = adjust_series_to_current_hydraulic_level(
        regular_series,
        value_col=station.variable,
        q=0.02,
        min_valid_days=300,
        rolling_window=3,
    )

    # logger.info("Año hidrológico de referencia: %s", hyd_stats["hyd_year_ref"].iloc[0])
    # logger.info("\n%s", hyd_stats.tail(10))

    # Grafica el minimo anual elegido y su tendencia suavizada
    # plot_hydrological_minimums(
    #     hyd_stats,
    #     station_name=station.nombre,
    #     q=0.02,
    # )

    # 5. Usar la columna corregida como variable principal para el resto del flujo.
    value_col_corr = f"{station.variable}_corr"
    
    corrected_series[station.variable] = corrected_series[value_col_corr]

    # 6. Dejar solo columnas útiles para el resto del script.
    corrected_series = corrected_series[[station.variable]].copy()

    # 7. Agregar variables temporales.
    corrected_series = add_time_variables(corrected_series)

    return corrected_series, outliers, hyd_stats

# Resampleo mensual
def resample_series(
    df: pd.DataFrame,
    group_var: str,
    value_col: str,
    min_count: int = 25,
) -> pd.DataFrame:
    """
    Calcula el promedio por año y ventana temporal: mes.

    Para el caso mensual, group_var='month'. Si la cantidad de datos válidos de una
    ventana es menor que min_count, el promedio se reemplaza por NaN.
    """
    required_cols = {"year", group_var, value_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")

    df_resamp = (
        df.groupby(["year", group_var])
        .agg(**{value_col: (value_col, "mean"), "Count": (value_col, "count")})
        .reset_index()
    )

    df_resamp.loc[df_resamp["Count"] < min_count, value_col] = np.nan
    return df_resamp.round(2)

# Persistencia de cuantiles
def get_selected_row(
    df: pd.DataFrame,
    year: int,
    period: int,
    period_col: str,
) -> tuple[int, pd.Series]:
    """Obtiene la fila correspondiente al año y período seleccionados."""
    mask = (df["year"] == year) & (df[period_col] == period)
    selected = df.loc[mask]

    if selected.empty:
        raise ValueError(f"No se encontró year={year}, {period_col}={period}")

    idx = int(selected.index[0])
    return idx, selected.iloc[0]

def forecast_persistence(
    df: pd.DataFrame,
    value_col: str,
    selected_month: int,
    selected_year: int,
    search_length: int,
    forecast_length: int,
    period_col: str = "month",
) -> pd.DataFrame:
    """Pronostica por persistencia de cuantiles."""
    if period_col != "month":
        raise NotImplementedError("Esta versión está implementada solo para period_col='month'")

    idx_selected, selected_row = get_selected_row(
        df,
        year=selected_year,
        period=selected_month,
        period_col=period_col,
    )

    selected_value = selected_row[value_col]
    if pd.isna(selected_value):
        raise ValueError("El valor seleccionado es NaN; no se puede calcular el cuantil")

    historical = df.loc[: idx_selected - 1].copy()
    same_month_values = historical.loc[historical[period_col] == selected_month, value_col].dropna()

    if same_month_values.empty:
        raise ValueError("No hay datos históricos para calcular el cuantil")

    selected_quantile = (same_month_values < selected_value).mean()
    # logger.info("Cuantil del mes seleccionado: %.3f", selected_quantile)

    records = []
    for horizon in range(1, forecast_length + 1):
        forecast_year, forecast_month = add_months(selected_year, selected_month, horizon)
        historical_values = historical.loc[historical[period_col] == forecast_month, value_col].dropna()
        forecast_value = historical_values.quantile(selected_quantile) if not historical_values.empty else np.nan

        records.append(
            {
                "metodo": "Persistencia",
                "horizonte": horizon,
                "year": forecast_year,
                "month": forecast_month,
                "Prono": forecast_value,
                "cuantil_base": selected_quantile,
            }
        )

        # logger.info(
        #     "Mes %s | q=%.6f | prono=%.3f | q_check=%.3f | n=%s",
        #     forecast_month,
        #     selected_quantile,
        #     forecast_value,
        #     (historical_values < forecast_value).mean(),
        #     len(historical_values),
        # )



    return add_date_column(pd.DataFrame(records).round(2))


# Analogías
def transform_for_analogy(
    df: pd.DataFrame,
    value_col: str,
    config: AnalogyConfig,
) -> pd.DataFrame:
    """
    Agrega columnas de transformación para analogías.

    - LogVar = log(variable)
    - LogVar_Est = variable logarítmica estandarizada por mes
    """
    out = df.copy()

    if (out[value_col].dropna() <= 0).any():
        logger.warning(
            "La serie tiene valores <= 0. Se usa eps_log=%s antes del log.",
            config.eps_log,
        )

    out[config.variable_log] = np.log(out[value_col].clip(lower=config.eps_log))

    monthly_stats = out.groupby("month")[config.variable_log].agg(["mean", "std"])
    out = out.merge(
        monthly_stats.rename(columns={"mean": "log_month_mean", "std": "log_month_std"}),
        left_on="month",
        right_index=True,
        how="left",
    )

    out[config.variable_transf] = (
        out[config.variable_log] - out["log_month_mean"]
    ) / out["log_month_std"]

    return out

def compute_fit_metrics(
    df: pd.DataFrame,
    obs_col: str,
    sim_col: str,
    month_selected: int,
    obs_name: Any = None,
    sim_name: Any = None,
) -> pd.DataFrame:
    """Calcula indicadores de similitud entre dos trazas."""
    valid = df[[obs_col, sim_col]].dropna()
    if valid.empty:
        return pd.DataFrame()

    obs = valid[obs_col].to_numpy(dtype=float)
    sim = valid[sim_col].to_numpy(dtype=float)

    obs_mean = round(float(np.mean(obs)), 3)
    sim_mean = round(float(np.mean(sim)), 3)

    f = float(np.square(sim - obs).sum())
    f0 = float(np.square(obs - obs.mean()).sum())
    nash = np.nan if f0 == 0 else round(100 * (f0 - f) / f0, 3)

    corr = np.nan if len(valid) < 2 else round(float(np.corrcoef(obs, sim)[0, 1]), 4)
    rmse = round(float(np.sqrt(np.square(sim - obs).mean())), 3)

    obs_diff = np.diff(obs)
    sim_diff = np.diff(sim)
    speds = np.nan if len(obs_diff) == 0 else round(float(100 * np.mean(obs_diff * sim_diff >= 0)), 2)

    vol_obs = float(obs.sum())
    vol_sim = float(sim.sum())
    err_vol = np.nan if vol_obs == 0 else round(100 * (vol_sim - vol_obs) / vol_obs, 3)

    return pd.DataFrame(
        {
            "YrObs": [obs_name if obs_name is not None else obs_col],
            "MesObs": [month_selected],
            "YrSim": [sim_name if sim_name is not None else sim_col],
            "nobs": [len(valid)],
            "Vobs_media": [obs_mean],
            "Vsim_media": [sim_mean],
            "Nash": [nash],
            "CoefC": [corr],
            "RMSE": [rmse],
            "SPEDS": [speds],
            "ErrVol": [err_vol],
        }
    )

def normalize_metrics(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Normaliza columnas métricas entre 0 y 1, controlando rangos nulos."""
    out = df.copy()
    for col in cols:
        rng = out[col].max() - out[col].min()
        if pd.isna(rng) or rng == 0:
            out[f"{col}_norm"] = 0.0
        else:
            out[f"{col}_norm"] = (out[col] - out[col].min()) / rng
    return out

def calc_indicators_by_date(
    df: pd.DataFrame,
    year_obj: int,
    month_obj: int,
    search_length: int,
    config: AnalogyConfig,
) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    """Calcula indicadores de similitud del año objetivo contra años históricos."""
    var_transf = config.variable_transf

    idx_selected, _ = get_selected_row(df, year_obj, month_obj, "month")
    idx_start = idx_selected - search_length + 1

    if idx_start < 0:
        return False, pd.DataFrame(), pd.DataFrame()

    df_obj = df.iloc[idx_start : idx_selected + 1].copy()
    if df_obj[var_transf].isna().any():
        return False, df_obj, pd.DataFrame()

    results = []
    for year_sim in sorted(df["year"].dropna().unique()):
        year_sim = int(year_sim)
        if year_sim == year_obj:
            continue

        try:
            idx_sim, _ = get_selected_row(df, year_sim, month_obj, "month")
        except ValueError:
            continue

        idx_sim_start = idx_sim - search_length + 1
        if idx_sim_start < 0:
            continue

        df_sim = df.iloc[idx_sim_start : idx_sim + 1].copy()
        if df_sim[var_transf].isna().any():
            continue

        df_union = pd.merge(
            df_obj[["month", var_transf]],
            df_sim[["month", var_transf]],
            on="month",
            suffixes=("_obs", "_sim"),
        )

        if len(df_union) < search_length:
            continue

        metrics = compute_fit_metrics(
            df_union,
            f"{var_transf}_obs",
            f"{var_transf}_sim",
            month_selected=month_obj,
            obs_name=year_obj,
            sim_name=year_sim,
        )
        if not metrics.empty:
            results.append(metrics)

    if not results:
        return False, df_obj, pd.DataFrame()

    df_ind = pd.concat(results, ignore_index=True)
    metric_cols = ["Nash", "CoefC", "RMSE", "SPEDS", "ErrVol"]
    df_ind = normalize_metrics(df_ind, metric_cols)

    df_ind["Score"] = (
        df_ind["Nash_norm"]
        + df_ind["CoefC_norm"]
        - df_ind["RMSE_norm"]
        + df_ind["SPEDS_norm"]
        - df_ind["ErrVol_norm"]
    )

    return True, df_obj, df_ind.sort_values("Score", ascending=False).reset_index(drop=True)

def build_forecast_calendar(
    selected_year: int,
    selected_month: int,
    forecast_length: int,
) -> pd.DataFrame:
    """Arma calendario mensual futuro a partir del mes seleccionado."""
    records = []
    for horizon in range(1, forecast_length + 1):
        year_i, month_i = add_months(selected_year, selected_month, horizon)
        records.append({"horizonte": horizon, "year": year_i, "month": month_i})
    return pd.DataFrame(records)

def get_window_by_origin(
    df: pd.DataFrame,
    origin_year: int,
    origin_month: int,
    before: int,
    after: int,
) -> pd.DataFrame:
    """
    Extrae una ventana mensual antes y después de un origen.

    rel_step = 0 corresponde al mes de origen.
    rel_step > 0 corresponde a meses pronosticados.
    """
    idx_origin, _ = get_selected_row(df, origin_year, origin_month, "month")
    idx_ini = idx_origin - before + 1
    idx_fin = idx_origin + after

    if idx_ini < 0 or idx_fin >= len(df):
        return pd.DataFrame()

    out = df.iloc[idx_ini : idx_fin + 1].copy()
    out["rel_step"] = range(-before + 1, after + 1)
    return out

def forecast_analogy(
    df: pd.DataFrame,
    value_col: str,
    selected_month: int,
    selected_year: int,
    forecast_cfg: ForecastConfig,
    analogy_cfg: AnalogyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pronostica por analogías.

    Devuelve:
    - forecast: pronóstico final ponderado
    - selected_analogs: años análogos seleccionados con métricas y pesos
    - traces: trazas para graficar, incluyendo objetivo y análogos
    - df_obj: ventana observada del objetivo
    """
    df_transf = transform_for_analogy(df, value_col=value_col, config=analogy_cfg)

    ok, df_obj, indicators = calc_indicators_by_date(
        df_transf,
        year_obj=selected_year,
        month_obj=selected_month,
        search_length=forecast_cfg.long_busqueda,
        config=analogy_cfg,
    )

    if not ok or indicators.empty:
        raise ValueError("No se pudieron calcular analogías para la fecha seleccionada")

    ordered = indicators.sort_values(
        by=analogy_cfg.orden,
        ascending=analogy_cfg.orden_ascending,
    ).reset_index(drop=True)

    selected_rows = []
    forecast_calendar = build_forecast_calendar(
        selected_year,
        selected_month,
        forecast_cfg.long_prono,
    )
    analog_values_transf = []
    traces = []

    # Traza objetivo: meses previos observados y meses posteriores si ya existen.
    target_trace = get_window_by_origin(
        df_transf,
        selected_year,
        selected_month,
        before=forecast_cfg.long_busqueda,
        after=forecast_cfg.long_prono,
    )
    if not target_trace.empty:
        target_trace = target_trace[["year", "month", "rel_step", value_col]].copy()
        target_trace["tipo"] = "Objetivo"
        target_trace["traza"] = str(selected_year)
        traces.append(target_trace)

    for _, row in ordered.iterrows():
        year_sim = int(row["YrSim"])

        analog_window = get_window_by_origin(
            df_transf,
            year_sim,
            selected_month,
            before=forecast_cfg.long_busqueda,
            after=forecast_cfg.long_prono,
        )
        if analog_window.empty:
            continue

        future = analog_window.loc[analog_window["rel_step"] > 0].copy()
        if len(future) < forecast_cfg.long_prono:
            logger.info("Año análogo %s omitido: faltan meses futuros", year_sim)
            continue

        if future[[value_col, analogy_cfg.variable_transf]].isna().any().any():
            logger.info("Año análogo %s omitido: contiene NaN", year_sim)
            continue

        selected_rows.append(row)
        analog_values_transf.append(future[analogy_cfg.variable_transf].to_numpy(dtype=float))

        trace_i = analog_window[["year", "month", "rel_step", value_col]].copy()
        trace_i["tipo"] = "Análogo"
        trace_i["traza"] = str(year_sim)
        traces.append(trace_i)

        if len(selected_rows) == analogy_cfg.cantidad:
            break

    if not selected_rows:
        raise ValueError("No se encontraron años análogos completos para pronosticar")

    selected_analogs = pd.DataFrame(selected_rows).reset_index(drop=True)

    # Peso inverso al RMSE, con protección ante RMSE=0.
    rmse = selected_analogs["RMSE"].astype(float).replace(0, np.nan)
    if rmse.isna().all():
        selected_analogs["wi"] = 1.0 / len(selected_analogs)
    else:
        min_positive = rmse[rmse > 0].min()
        rmse = rmse.fillna(min_positive * 0.1)
        selected_analogs["wi"] = (1.0 / rmse) / (1.0 / rmse).sum()

    analog_matrix = np.vstack(analog_values_transf)
    weights = selected_analogs["wi"].to_numpy(dtype=float)
    prono_transf = np.average(analog_matrix, axis=0, weights=weights)

    forecast = forecast_calendar.copy()
    forecast[analogy_cfg.variable_transf] = prono_transf

    # Inversión de la transformación mensual.
    stats = df_transf.groupby("month")[["log_month_mean", "log_month_std"]].first()
    forecast = forecast.merge(stats, left_on="month", right_index=True, how="left")
    forecast["Prono"] = np.exp(
        forecast[analogy_cfg.variable_transf] * forecast["log_month_std"]
        + forecast["log_month_mean"]
    )
    forecast["metodo"] = "Analogía"
    forecast = add_date_column(forecast.round(2))

    traces_df = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()

    return forecast, selected_analogs.round(4), traces_df, df_obj


# Gráficos
def get_axis_label(value_col: str) -> str:
    """Devuelve una etiqueta de eje según la variable."""
    labels = {
        "Caudal": r"Caudal [m$^3$/s]",
        "Nivel": "Nivel [m]",
        "valor": "Nivel [m]",
    }
    return labels.get(value_col, value_col)

def plot_forecast_boxplot(
    station_name: str,
    recent_obs: pd.DataFrame,
    forecast: pd.DataFrame,
    historical: pd.DataFrame,
    period_col: str,
    value_col: str,
    search_length: int,
    forecast_col: str = "Prono",
) -> None:
    """Grafica boxplots históricos mensuales, observaciones recientes y pronóstico."""

    months = list(range(1, 13))

    box_plot_data = [
        historical.loc[historical[period_col] == month, value_col].dropna()
        for month in months
    ]

    box_plot_labels = [
        MONTH_NAMES.get(month, str(month))
        for month in months
    ]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.boxplot(
        box_plot_data,
        patch_artist=True,
        labels=box_plot_labels,
        boxprops={"fill": None},
    )

    # Observaciones recientes
    if not recent_obs.empty:
        ax.scatter(
            recent_obs[period_col],
            recent_obs[value_col],
            s=60,
            label=f"Últimos {search_length} meses obs.",
            zorder=3,
        )

    # Pronóstico
    if not forecast.empty:
        ax.scatter(
            forecast[period_col],
            forecast[forecast_col],
            s=70,
            marker="s",
            label="Pronóstico",
            zorder=4,
        )

    ax.set_title(station_name)
    ax.grid(
        True,
        axis="y",
        which="both",
        color="0.75",
        linestyle="-.",
        linewidth=0.3,
    )

    ax.set_xlabel("Mes", size=18)
    ax.set_ylabel(get_axis_label(value_col), size=18)
    ax.tick_params(axis="x", labelsize=14, rotation=20)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(prop={"size": 14}, loc="best")
    # plt.show()
    # plt.close(fig)
    # print(forecast[["fecha", "Prono"]])
    # print(months)
    return fig

def plot_analogy_traces(
    traces: pd.DataFrame,
    forecast: pd.DataFrame,
    selected_analogs: pd.DataFrame,
    station_name: str,
    value_col: str,
    selected_year: int,
    selected_month: int,
) -> None:
    """
    Grafica las trazas análogas y la traza objetivo.

    El eje x está en meses relativos:
    - valores negativos: meses previos usados para buscar analogías
    - 0: mes de emisión/origen
    - valores positivos: meses pronosticados
    """
    if traces.empty:
        logger.warning("No hay trazas para graficar")
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    analog_years = selected_analogs["YrSim"].astype(int).astype(str).tolist()
    weights = dict(zip(selected_analogs["YrSim"].astype(int).astype(str), selected_analogs["wi"]))

    # Análogos seleccionados.
    for year_i in analog_years:
        trace_i = traces.loc[traces["traza"] == year_i].sort_values("rel_step")
        if trace_i.empty:
            continue
        label = f"Análogo {year_i} (w={weights.get(year_i, np.nan):.2f})"
        ax.plot(trace_i["rel_step"], trace_i[value_col], marker="o", linewidth=1.5, alpha=0.75, label=label)

    # Objetivo observado.
    target = traces.loc[traces["tipo"] == "Objetivo"].sort_values("rel_step")
    if not target.empty:
        ax.plot(
            target["rel_step"],
            target[value_col],
            marker="o",
            linewidth=3.0,
            label=f"Objetivo {selected_year}",
        )

    # Pronóstico final por analogía.
    if not forecast.empty:
        ax.plot(
            forecast["horizonte"],
            forecast["Prono"],
            marker="s",
            linewidth=3.0,
            linestyle="--",
            label="Pronóstico analogía",
        )

    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.set_title(f"{station_name} - Analogías desde {MONTH_NAMES[selected_month]} {selected_year}")
    ax.set_xlabel("Mes relativo al origen del pronóstico")
    ax.set_ylabel(get_axis_label(value_col))
    ax.grid(True, which="both", color="0.75", linestyle="-.", linewidth=0.4)
    ax.legend(loc="best")
    # plt.show()
    # plt.close(fig)
    return fig

def plot_forecasts_comparison(
    df: pd.DataFrame,
    persistence: pd.DataFrame,
    analogy: pd.DataFrame,
    station_name: str,
    value_col: str,
    selected_year: int,
    selected_month: int,
    history_months: int = 12,
) -> None:
    """Grafica observados recientes y pronósticos de persistencia y analogía."""
    idx_selected, _ = get_selected_row(df, selected_year, selected_month, "month")
    hist = df.iloc[max(0, idx_selected - history_months + 1) : idx_selected + 1].copy()
    hist = add_date_column(hist)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist["fecha"], hist[value_col], marker="o", label="Observado")

    if not persistence.empty:
        ax.plot(persistence["fecha"], persistence["Prono"], marker="s", linestyle="--", label="Persistencia")
    if not analogy.empty:
        ax.plot(analogy["fecha"], analogy["Prono"], marker="s", linestyle="--", label="Analogía")

    ax.set_title(f"{station_name} - Pronóstico mensual")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(get_axis_label(value_col))
    ax.grid(True, which="both", color="0.75", linestyle="-.", linewidth=0.4)
    ax.legend(loc="best")
    # plt.show()
    # plt.close(fig)
    return fig


def get_last_valid_forecast_origin(
    df: pd.DataFrame,
    value_col: str,
    period_col: str = "month",
) -> tuple[int, int]:
    """Devuelve el último mes con valor mensual válido."""
    valid = df.dropna(subset=[value_col]).copy()

    if valid.empty:
        raise ValueError("No hay meses válidos para definir el origen del pronóstico")

    row = valid.iloc[-1]
    return int(row["year"]), int(row[period_col])


# Ejecución principal
def main() -> None:
    station = StationConfig()
    cleaning = CleaningConfig()
    forecast_cfg = ForecastConfig()
    analogy_cfg = AnalogyConfig()

    # Si no se fija manualmente el mes/año de emisión, se infiere con la regla operativa.
    if forecast_cfg.mes_select is None or forecast_cfg.yr_select is None:
        forecast_cfg.yr_select, forecast_cfg.mes_select = infer_forecast_origin()

    logger.info("Estación: %s", station.nombre)
    logger.info("Origen del pronóstico: %02d/%04d", forecast_cfg.mes_select, forecast_cfg.yr_select)

    serie_reg, outliers, hyd_stats = prepare_regular_series(station, cleaning)

    # Resampleo mensual
    df_resamp = resample_series(
        serie_reg,
        group_var=forecast_cfg.vent_resamp,
        value_col=station.variable,
        min_count=forecast_cfg.min_count_mensual,
    )

    logger.info(
        "Serie mensual desde %s/%s hasta %s/%s",
        int(df_resamp.loc[0, forecast_cfg.vent_resamp]),
        int(df_resamp.loc[0, "year"]),
        int(df_resamp.loc[len(df_resamp) - 1, forecast_cfg.vent_resamp]),
        int(df_resamp.loc[len(df_resamp) - 1, "year"]),
    )
    logger.info("NaN mensuales: %s", df_resamp[station.variable].isna().sum())


    # Pronóstico por persistencia de cuantiles.
    prono_persistencia = forecast_persistence(
        df=df_resamp,
        value_col=station.variable,
        selected_month=forecast_cfg.mes_select,
        selected_year=forecast_cfg.yr_select,
        search_length=forecast_cfg.long_busqueda,
        forecast_length=forecast_cfg.long_prono,
        period_col=forecast_cfg.vent_resamp,
    )

    # Pronóstico por analogía.
    prono_analogia, analogos, trazas, _ = forecast_analogy(
        df=df_resamp,
        value_col=station.variable,
        selected_month=forecast_cfg.mes_select,
        selected_year=forecast_cfg.yr_select,
        forecast_cfg=forecast_cfg,
        analogy_cfg=analogy_cfg,
    )

    print("\nPronóstico por persistencia:\n")
    print(prono_persistencia[["fecha", "horizonte", "Prono", "cuantil_base"]])

    print("\nPronóstico por analogía:\n")
    print(prono_analogia[["fecha", "horizonte", "Prono"]])

    print("\nAños análogos seleccionados:\n")
    print(analogos[["YrSim", "RMSE", "CoefC", "Nash", "ErrVol", "wi"]])


    if forecast_cfg.plot:
        idx_selected, _ = get_selected_row(
            df_resamp,
            forecast_cfg.yr_select,
            forecast_cfg.mes_select,
            forecast_cfg.vent_resamp,
        )

        recent_obs = df_resamp.iloc[
            max(0, idx_selected - forecast_cfg.long_busqueda + 1): idx_selected + 1
        ].copy()

        plot_forecast_boxplot(
            station_name=station.nombre,
            recent_obs=recent_obs,
            forecast=prono_persistencia, # prono_persistencia   prono_analogia
            historical=df_resamp,
            period_col=forecast_cfg.vent_resamp,
            value_col=station.variable,
            search_length=forecast_cfg.long_busqueda,
        )

        plot_analogy_traces(
            traces=trazas,
            forecast=prono_analogia, # prono_persistencia   prono_analogia
            selected_analogs=analogos,
            station_name=station.nombre,
            value_col=station.variable,
            selected_year=forecast_cfg.yr_select,
            selected_month=forecast_cfg.mes_select,
        )

        plot_forecasts_comparison(
            df=df_resamp,
            persistence=prono_persistencia,
            analogy=prono_analogia,
            station_name=station.nombre,
            value_col=station.variable,
            selected_year=forecast_cfg.yr_select,
            selected_month=forecast_cfg.mes_select,
        )


# if __name__ == "__main__":
#     main()

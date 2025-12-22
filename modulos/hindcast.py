# modulos/hindcast.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.api import OLS

from .CrosCorrAnalysis_mod import get_response_time, add_constant

logger = logging.getLogger(__name__)

def evaluar_estaciones_individuales(
    df_union: pd.DataFrame,
    estaciones: Tuple[str, str] = ("Villa Montes", "Puente Aruma"),
    obs_col: str = "Misión La Paz",
    max_lag: int = 72,
    ini_lag: int = 2,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Ajusta un modelo lineal con UNA sola estación a la vez (con su lag óptimo)
    y calcula R² para cada una.

    Devuelve:
    orden : list[str]
        Estaciones ordenadas de mejor a peor (por R²).
    df_res : DataFrame
        Resumen con lag óptimo, R² y número de muestras por estación.
    """
    resultados: List[Dict] = []

    for est in estaciones:
        up = df_union[est].copy()
        down = df_union[obs_col].copy()

        df_cal = pd.concat([up, down], axis=1).dropna()
        df_cal.columns = ["up", "down"]

        if df_cal.empty:
            logger.warning("No hay datos simultáneos para %s - %s", est, obs_col)
            continue
        lag = get_response_time(up_series=df_cal["up"],down_series=df_cal["down"],max_lag=max_lag,ini=ini_lag,)     
        
        try:
            lag = get_response_time(
                up_series=df_cal["up"],
                down_series=df_cal["down"],
                max_lag=max_lag,
                ini=ini_lag,
            )
        except Exception as exc:
            logger.warning(
                "Error estimando lag global para %s: %s", est, exc
            )
            continue

        # Crear predictor lagueado
        df_lag = df_cal.copy()
        df_lag["up_lag"] = df_lag["up"].shift(lag)
        data = df_lag.dropna(subset=["up_lag", "down"])

        if len(data) < 50:
            logger.warning("Datos insuficientes para ajustar modelo con %s", est)
            continue

        X = data[["up_lag"]]
        X = add_constant(X)
        y = data["down"]

        try:
            modelo = OLS(y, X).fit()
        except Exception as exc:
            logger.warning("Error ajustando modelo global con %s: %s", est, exc)
            continue

        r2 = float(modelo.rsquared)
        resultados.append(
            {
                "estacion": est,
                "lag_optimo": lag,
                "r2": r2,
                "n_muestras": len(data),
            }
        )

        logger.info(
            "[GLOBAL] Estación %s: lag=%d steps, R²=%.3f, n=%d",
            est, lag, r2, len(data),
        )

    if not resultados:
        raise RuntimeError("No se pudo evaluar ninguna estación de forma individual.")

    df_res = pd.DataFrame(resultados).sort_values("r2", ascending=False)
    orden = df_res["estacion"].tolist()
    return orden, df_res

def ajustar_estacion_con_lag(
    df_union: pd.DataFrame,
    est: str,
    obs_col: str,
    lag: int,
    min_muestras: int = 50,
) -> tuple[pd.Series, pd.Series, object]:
    """
    Ajusta un modelo lineal obs ~ upstream_lag.
    
    Devuelve:
    - y_obs (Serie observada)
    - y_fit (Serie ajustada)
    - modelo (OLS)
    """
    up = df_union[est]
    down = df_union[obs_col]

    df_cal = pd.concat([up, down], axis=1).dropna()
    df_cal.columns = ["up", "down"]

    df_cal["up_lag"] = df_cal["up"].shift(lag)
    data = df_cal.dropna(subset=["up_lag", "down"])

    if len(data) < min_muestras:
        raise ValueError(f"Datos insuficientes tras aplicar lag={lag} (n={len(data)}).")

    X = data[["up_lag"]]
    X = add_constant(X)
    y = data["down"]

    modelo = OLS(y, X).fit()

    y_fit = modelo.predict(X)
    y_fit.index = y.index 

    return y, y_fit, modelo

def calibrar_y_pronosticar_ventana(
    df_union: pd.DataFrame,
    t_emision: pd.Timestamp,
    primary: str = "Villa Montes",
    secondary: str = "Puente Aruma",
    obs_col: str = "Misión La Paz",
    ventana: pd.DateOffset = pd.DateOffset(years=1, months=3),  # no se usa directo
    ventana_lag: pd.DateOffset = pd.DateOffset(years=1, months=3),
    ventana_ML: pd.DateOffset = pd.DateOffset(years=1, months=3),
    freq: str = "1h",
    horizonte_horas: int = 42,
    max_lag: int = 72,
    ini_lag: int = 2,
    min_muestras: int = 200,
) -> pd.DataFrame:
    """
    Para una fecha de emisión `t_emision`:

    - Toma una ventana de calibración previa a la emisión.
    - Ajusta modelos de UNA sola estación (primary y secondary), cada uno
      con su lag óptimo dentro de la ventana.
    - Pronostica las próximas horas usando:
        * primero la estación principal (primary)
        * si no hay datos, usa la estación secundaria (secondary)

    NOTA: el horizonte de pronóstico se fija igual
    al lag de la estación primaria (lead_obj).

    Devuelve:
    df_prono : DataFrame
        Columnas:
        - Fecha          (tiempo pronosticado)
        - Valor          (pronóstico en obs_col)
        - Fecha_emitido  (t_emision)
    """
    df = df_union.sort_index().copy()
    step = pd.to_timedelta(freq)

    # Ventana de calibración para lag y modelo
    t_ini_lag = t_emision - ventana_lag
    t_fin = t_emision - step

    df_cal_lag = df.loc[(df.index >= t_ini_lag) & (df.index <= t_fin)].copy()

    if df_cal_lag.empty or len(df_cal_lag) < min_muestras:
        logger.warning(
            "No hay datos suficientes para calibrar en ventana %s — %s (len=%d)",
            t_ini_lag, t_fin, len(df_cal_lag),
        )
        return pd.DataFrame(columns=["Fecha", "Valor", "Fecha_emitido"])

    # Ventana para Misión La Paz
    t_ini_ml = t_emision - ventana_ML
    df_cal = df.loc[(df.index >= t_ini_ml) & (df.index <= t_fin)].copy()

    y_cal = df_cal[obs_col]
    modelos: Dict[str, Dict] = {}  # est -> {"modelo":..., "lag":...}

    # para ajustar un modelo con UNA estación
    def _ajustar_un_est(est: str) -> Optional[Dict[str, object]]:
        up = df_cal[est]
        base = pd.concat([up, y_cal], axis=1).dropna()
        base.columns = ["up", "down"]

        if len(base) < min_muestras:
            logger.info(
                "Datos insuficientes para %s en ventana %s — %s (len=%d)",
                est, t_ini_ml, t_fin, len(base),
            )
            return None

        try:
            lag = get_response_time(
                up_series=base["up"],
                down_series=base["down"],
                max_lag=max_lag,
                ini=ini_lag,
            )
        except Exception as exc:
            logger.warning(
                "Error estimando lag para %s en ventana %s — %s: %s",
                est, t_ini_ml, t_fin, exc,
            )
            return None

        base["up_lag"] = base["up"].shift(lag)
        data = base.dropna(subset=["up_lag", "down"])

        if len(data) < min_muestras:
            logger.info(
                "Datos insuficientes tras aplicar lag=%d para %s (len=%d)",
                lag, est, len(data),
            )
            return None

        X = data[["up_lag"]]
        X = add_constant(X)
        y = data["down"]

        try:
            modelo = OLS(y, X).fit()
        except Exception as exc:
            logger.error("Error ajustando modelo con %s: %s", est, exc)
            return None

        logger.info(
            "Ventana %s — %s: modelo %s con lag=%d, R²=%.3f, n=%d",
            t_ini_ml, t_fin, est, lag, float(modelo.rsquared), len(data),
        )
        return {"modelo": modelo, "lag": lag}

    # Ajustar primary y secondary
    modelos_primary = _ajustar_un_est(primary)
    if modelos_primary is not None:
        modelos[primary] = modelos_primary

    modelos_secondary = _ajustar_un_est(secondary)
    if modelos_secondary is not None:
        modelos[secondary] = modelos_secondary

    if not modelos:
        logger.warning(
            "No se pudieron ajustar modelos para ninguna estación en ventana %s — %s",
            t_ini_ml, t_fin,
        )
        return pd.DataFrame(columns=["Fecha", "Valor", "Fecha_emitido"])

    logger.info(
        "Fecha emisión %s - modelos disponibles: %s",
        t_emision, list(modelos.keys()),
    )
    
    # Pronosticar las próximas horizonte_horas horas ---
    df_hist = df.loc[:t_emision].copy()

    registros: List[Dict] = []

    # Definir el lead objetivo a pronosticar
    lead_obj: Optional[int] = None
    if primary in modelos:
        lead_obj = modelos[primary]["lag"]
    elif secondary in modelos:
        lead_obj = modelos[secondary]["lag"]

    if lead_obj is None:
        logger.warning(
            "No se encontró lag para primary ni secondary en t_emision=%s", t_emision
        )
        return pd.DataFrame(columns=["Fecha", "Valor", "Fecha_emitido"])

    # En la versión actual, el horizonte se fuerza a ser igual al lag
    horizonte_horas = lead_obj

    for h in range(1, horizonte_horas + 1):
        t_forecast = t_emision + h * step

        # No pronosticamos más allá de la última observación disponible
        if t_forecast > df.index.max():
            break

        y_hat: Optional[float] = None

        # Intentar primero con la estación principal
        if primary in modelos:
            lag_p = modelos[primary]["lag"]
            modelo_p = modelos[primary]["modelo"]
            t_up_p = t_forecast - lag_p * step
            if t_up_p in df_hist.index:
                up_p = df_hist.loc[t_up_p, primary]
                if not pd.isna(up_p):
                    x_p = pd.Series({"const": 1.0, "up_lag": up_p})
                    x_p = x_p[modelo_p.params.index]
                    y_hat = float(np.dot(x_p.values, modelo_p.params.values))

        # Si no se pudo con la principal, probar con la secundaria
        if (y_hat is None) and (secondary in modelos):
            lag_s = modelos[secondary]["lag"]
            modelo_s = modelos[secondary]["modelo"]
            t_up_s = t_forecast - lag_s * step
            if t_up_s in df_hist.index:
                up_s = df_hist.loc[t_up_s, secondary]
                if not pd.isna(up_s):
                    x_s = pd.Series({"const": 1.0, "up_lag": up_s})
                    x_s = x_s[modelo_s.params.index]
                    y_hat = float(np.dot(x_s.values, modelo_s.params.values))

        if y_hat is None:
            # Para este lead no hay información disponible ni en primary ni en secondary
            continue

        registros.append(
            {
                "Fecha": t_forecast,
                "Valor": y_hat,
                "Fecha_emitido": t_emision,
            }
        )

    if not registros:
        return pd.DataFrame(columns=["Fecha", "Valor", "Fecha_emitido"])

    return pd.DataFrame(registros)

def hindcast_diario(
    df_union: pd.DataFrame,
    primary: str,
    secondary: str,
    obs_col: str = "Misión La Paz",
    fecha_ini_eval: str | None = None,
    fecha_fin_eval: str | None = None,
    freq: str = "1h",
    ventana_lag: pd.DateOffset = pd.DateOffset(years=1, months=3),
    ventana_ML: pd.DateOffset = pd.DateOffset(years=1, months=3),
    horizonte_horas: int = 42,
    min_muestras: int = 200,
) -> pd.DataFrame:
    """
    Hindcast diario:

    Para cada día entre fecha_ini_eval y fecha_fin_eval:
      - Usa una ventana histórica (por defecto 1 año + 3 meses) para calibrar.
      - Ajusta modelos de UNA estación (primary y secondary).
      - Pronostica las próximas horas usando:
          * primero la principal (primary)
          * si no hay datos, la secundaria (secondary)

    Devuelve
    --------
    df_hind : DataFrame
        Todas las corridas concatenadas, con columnas:
        - Fecha
        - Valor
        - Fecha_emitido
    """
    df = df_union.sort_index().copy()

    if fecha_ini_eval is None:
        fecha_ini_eval = df.index.min().normalize()
    else:
        fecha_ini_eval = pd.to_datetime(fecha_ini_eval).normalize()

    if fecha_fin_eval is None:
        fecha_fin_eval = df.index.max().normalize()
    else:
        fecha_fin_eval = pd.to_datetime(fecha_fin_eval).normalize()

    fechas_emision = pd.date_range(
        start=fecha_ini_eval,
        end=fecha_fin_eval,
        freq="D",
    )

    logger.info(
        "Iniciando hindcast diario desde %s hasta %s (%d emisiones) usando primary=%s, secondary=%s",
        fecha_ini_eval.date(), fecha_fin_eval.date(), len(fechas_emision),
        primary, secondary,
    )

    todos_registros: List[pd.DataFrame] = []

    for t_emision in fechas_emision:
        # Se emite a las 09:00 de cada día
        t_emision_ts = pd.Timestamp(t_emision).replace(hour=9, minute=0, second=0)

        if t_emision_ts <= df.index.min():
            continue

        logger.info("Emitir pronóstico para %s", t_emision_ts)

        df_prono = calibrar_y_pronosticar_ventana(
            df_union=df,
            t_emision=t_emision_ts,
            primary=primary,
            secondary=secondary,
            obs_col=obs_col,
            ventana_lag=ventana_lag,
            ventana_ML=ventana_ML,
            freq=freq,
            horizonte_horas=horizonte_horas,
            max_lag=72,
            ini_lag=2,
            min_muestras=min_muestras,
        )

        if not df_prono.empty:
            todos_registros.append(df_prono)

    if not todos_registros:
        logger.warning("No se generó ningún pronóstico en el período dado.")
        return pd.DataFrame(columns=["Fecha", "Valor", "Fecha_emitido"])

    df_hind = pd.concat(todos_registros, ignore_index=True)
    return df_hind

def forecast_from_upstream(
    df: pd.DataFrame,
    est: str,
    obs_col: str,
    lag: int,
    modelo,
    freq: str = "1h",
) -> pd.Series:
    """
    y_hat(t) = const + beta * up(t - lag*step)
    Devuelve Serie indexada por t (tiempo objetivo).
    """
    step = pd.to_timedelta(freq)

    up = df[est].copy()
    idx_t = df.index

    t_up = idx_t - lag * step
    up_at = up.reindex(t_up).to_numpy()

    const = float(modelo.params.get("const", 0.0))
    beta = float(modelo.params.get("up_lag", np.nan))

    y_hat = const + beta * up_at
    return pd.Series(y_hat, index=idx_t, name=f"y_hat_{est}_lag{lag}")

"""
Streamlit - Tablero paso a paso (v2)

Enfoque de esta versión:
1) Descargar series (rango editable; default hoy-1 año → hoy)
2) Limpiar series (usa el flujo existente en construir_series_union)
3) Exportar series limpias a Excel
4) Estimar lag óptimo por estación en una ventana elegida
5) Aplicar lag (auto o manual) y graficar 3 series: upstream cruda, upstream alineada por lag, downstream obs


Uso:
  streamlit run app_streamlit_v2.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd
import streamlit as st

from dotenv import load_dotenv

# Import
def _import_project_functions():
    """
    Importa funciones/clases del paquete `modulos/`.
    """
    try:
        from modulos.series import construir_series_union
        from modulos.hindcast import evaluar_estaciones_individuales
        from a5client import Crud
        return construir_series_union, evaluar_estaciones_individuales, Crud
    except Exception as e:
        raise ImportError(
            "No se pudo importar desde `modulos/`. Verificá que exista `modulos/series.py`, "
            "`modulos/hindcast.py` y `a5client.py`, y que estés corriendo Streamlit "
            "desde la raíz del proyecto."
        ) from e

construir_series_union, evaluar_estaciones_individuales, Crud = _import_project_functions()

# Utils
def _today_local() -> date:
    # Streamlit corre donde corre Python: usamos la fecha local del sistema
    return datetime.now().date()

def _to_dt_start(d: date) -> str:
    # construir_series_union espera timestart como string
    return datetime.combine(d, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")

def _to_dt_end(d: date) -> datetime:
    # construir_series_union espera timeend como datetime
    return datetime.combine(d, datetime.max.time()).replace(microsecond=0)

def _df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "series") -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name)
    return out.getvalue()

def _parse_step(step_adopt: str) -> pd.Timedelta:
    """
    step_adopt típicamente "1H" o "1D".
    """
    try:
        return pd.to_timedelta(step_adopt)
    except Exception:
        # fallback: valores comunes
        if step_adopt.lower() in {"h", "1h", "hour", "hora"}:
            return pd.Timedelta(hours=1)
        if step_adopt.lower() in {"d", "1d", "day", "dia"}:
            return pd.Timedelta(days=1)
        raise

def _apply_lag_align(up: pd.Series, step: pd.Timedelta, lag_steps: int) -> pd.Series:
    """
    Alinea la serie upstream hacia adelante (mueve timestamps +lag*step),
    para que quede temporalmente comparable con la downstream en el mismo eje.
    """
    if lag_steps is None:
        lag_steps = 0
    return up.shift(lag_steps, freq=step)

# UI
st.set_page_config(page_title="Pilcomayo - Tablero de control", layout="wide")

st.title("Tablero de control")

with st.sidebar:
    st.header("Configuración")

    # A5 fijo para este proyecto
    load_dotenv()
    a5_url = "https://alerta.ina.gob.ar/a6"
    st.text_input("A5_URL", value=a5_url, disabled=True)

    # Token: por env o ingreso manual
    a5_token_env = os.getenv("A5_TOKEN", "")
    a5_token = st.text_input("A5_TOKEN", value=a5_token_env, type="password")

    st.divider()

    # Paso temporal fijo (1 hora)
    step_adopt = "h"
    st.text_input("Paso temporal (fijo)", value=step_adopt, disabled=True)

    carpeta_figuras = st.text_input(
        "Carpeta de figuras",
        value=str(Path("resultados") / "figuras"),
        help="Dónde guardar los PNG que genera el flujo de limpieza/análisis",
    )

    archivo_salida_csv = st.text_input("Salida CSV (series limpias)", value="series_nivel_union_h.csv")
    archivo_salida_resumen = st.text_input("Salida Excel (resumen)", value="resumen_series_niveles_h.xlsx")

    st.divider()
    #st.caption("")

if "df_union" not in st.session_state:
    st.session_state.df_union = None
if "df_resumen" not in st.session_state:
    st.session_state.df_resumen = None
if "lags_df" not in st.session_state:
    st.session_state.lags_df = None
if "obs_col" not in st.session_state:
    st.session_state.obs_col = "Misión La Paz"
if "stations" not in st.session_state:
    # valores por defecto (editable)
    st.session_state.stations = [
        {"Estacion": "Misión La Paz", "serie_id": 42293},
        {"Estacion": "Villa Montes", "serie_id": 42291},
        {"Estacion": "Puente Aruma", "serie_id": 42294},
    ]

# Estaciones
st.subheader("Estaciones")

colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.write("Editá la lista de estaciones (nombre + `serie_id`).")
    stations_df = pd.DataFrame(st.session_state.stations)
    stations_df = st.data_editor(
        stations_df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Estacion": st.column_config.TextColumn(required=True),
            "serie_id": st.column_config.NumberColumn(required=True, step=1),
        },
        key="stations_editor",
    )
    # persistimos
    st.session_state.stations = stations_df.to_dict("records")

with colB:
    # Downstream/observada fija para este proyecto
    st.session_state.obs_col = "Misión La Paz"
    st.info("Estación objetivo: **Misión La Paz**.")


# Paso 1: Descargar + limpiar series
st.subheader("Paso 1 — Descargar y limpiar series (rango editable)")

today = _today_local()
default_start = today - timedelta(days=365)
default_end = today

col1, col2, col3 = st.columns([1, 1, 2], gap="large")

with col1:
    d_start = st.date_input("Desde", value=default_start, format="DD/MM/YYYY")
with col2:
    d_end = st.date_input("Hasta", value=default_end, format="DD/MM/YYYY")

with col3:
    st.write("")
    st.caption("Por defecto: hoy − 1 año → hoy. El flujo descarga, limpia y arma el dataframe unificado.")

run_build = st.button("Descargar + limpiar (construir df_union)", type="primary")

if run_build:
    if not a5_url or not a5_token:
        st.error("Falta A5_URL o A5_TOKEN.")
    else:
        try:
            step_td = _parse_step(step_adopt)

            estaciones_dict: Dict[str, int] = {
                r["Estacion"]: int(r["serie_id"])
                for r in st.session_state.stations
                if r.get("Estacion") and pd.notna(r.get("serie_id"))
            }

            # cliente A5
            client = Crud(a5_url, token=a5_token)

            timestart = _to_dt_start(d_start)
            timeend = _to_dt_end(d_end)

            Path(carpeta_figuras).mkdir(parents=True, exist_ok=True)

            with st.spinner("Descargando, limpiando y unificando series..."):
                df_union, df_resumen = construir_series_union(
                    Estaciones=estaciones_dict,
                    timestart=timestart,
                    timeend=timeend,
                    step_adopt=step_adopt,
                    client=client,
                    carpeta_figuras=carpeta_figuras,
                    archivo_salida=archivo_salida_csv,
                    archivo_salida_resumen=archivo_salida_resumen,
                )

            st.session_state.df_union = df_union
            st.session_state.df_resumen = df_resumen

            st.success(f"Listo. df_union: {df_union.shape[0]} filas × {df_union.shape[1]} columnas.")
        except Exception as e:
            st.exception(e)

if st.session_state.df_union is not None:
    df_union: pd.DataFrame = st.session_state.df_union
    st.write("Vista rápida (últimas filas):")
    st.dataframe(df_union.tail(10), width="stretch")

    c1, c2, c3 = st.columns([1, 1, 2], gap="large")

    with c1:
        st.download_button(
            "Descargar CSV (series limpias)",
            data=df_union.to_csv().encode("utf-8"),
            file_name=Path(archivo_salida_csv).name,
            mime="text/csv",
        )
    with c2:
        excel_bytes = _df_to_excel_bytes(df_union, sheet_name="series_limpias")
        st.download_button(
            "Descargar Excel (series limpias)",
            data=excel_bytes,
            file_name=Path(archivo_salida_csv).with_suffix(".xlsx").name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c3:
        st.caption("Agregar descarga de las series *crudas* (sin limpieza) además de las limpias.")


# --------------------------------------------------------------------------------------
# Paso 3: Estimar lag óptimo (ventana) + editar manual + plot 3 series
# --------------------------------------------------------------------------------------
st.subheader("Paso 2 — Lag óptimo por estación + gráfico (auto y manual)")

if st.session_state.df_union is None:
    st.info("Primero ejecutá el Paso 2 para tener df_union.")
else:
    df_union = st.session_state.df_union
    step_td = _parse_step(step_adopt)

    # ventana para estimar lag (subset del df_union)
    st.write("Elegí la **ventana temporal** para estimar lag (puede ser distinta del rango descargado).")

    idx_min = df_union.index.min()
    idx_max = df_union.index.max()
    # defaults: último año o lo que haya
    default_lag_start = max(idx_min.date(), (idx_max - pd.Timedelta(days=365)).date())
    default_lag_end = idx_max.date()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="large")
    with c1:
        lag_start = st.date_input("Ventana lag - desde", value=default_lag_start, key="lag_start", format="DD/MM/YYYY")
    with c2:
        lag_end = st.date_input("Ventana lag - hasta", value=default_lag_end, key="lag_end", format="DD/MM/YYYY")
    with c3:
        max_lag = st.number_input("max_lag (pasos)", min_value=1, max_value=500, value=72, step=1)
    with c4:
        ini_lag = st.number_input("ini_lag (pasos)", min_value=0, max_value=200, value=2, step=1)

    run_lag = st.button("Estimar lag óptimo", type="secondary")

    if run_lag:
        try:
            obs_col = st.session_state.obs_col
            # upstream = todas menos la obs
            upstream_list = [c for c in df_union.columns if c != obs_col]

            df_sub = df_union.loc[
                datetime.combine(lag_start, datetime.min.time()):
                datetime.combine(lag_end, datetime.max.time())
            ].copy()

            with st.spinner("Estimando lag óptimo por estación..."):
                orden, df_lags = evaluar_estaciones_individuales(
                    df_union=df_sub,
                    estaciones=tuple(upstream_list),
                    obs_col=obs_col,
                    max_lag=int(max_lag),
                    ini_lag=int(ini_lag),
                )

            df_lags = df_lags.copy()
            # columna editable para manual
            if "lag_optimo" in df_lags.columns:
                df_lags["lag_manual"] = df_lags["lag_optimo"].astype(int)
            elif "lag" in df_lags.columns:
                df_lags["lag_manual"] = df_lags["lag"].astype(int)
            else:
                # por si cambia el nombre
                df_lags["lag_manual"] = 0

            st.session_state.lags_df = df_lags

            st.success("Lag estimado. Podés editar 'lag_manual' y el gráfico se actualiza.")
        except Exception as e:
            st.exception(e)

    if st.session_state.lags_df is not None:
        df_lags = st.session_state.lags_df

        st.write("Resultado (editable):")
        df_lags_edit = st.data_editor(
            df_lags,
            width="stretch",
            column_config={
                "lag_manual": st.column_config.NumberColumn(
                    "lag_manual",
                    help="Editá a mano si querés. Se interpreta en 'pasos' del step_adopt (ej: horas si step=1H).",
                    step=1,
                )
            },
            key="lags_editor",
        )
        st.session_state.lags_df = df_lags_edit

        # Selector de estación para plot
        obs_col = st.session_state.obs_col
        upstream_candidates = [c for c in df_union.columns if c != obs_col]

        left, right = st.columns([1, 2], gap="large")

        with left:
            station_to_plot = st.selectbox("Estación upstream para graficar", options=upstream_candidates)
            # rango de plot
            plot_start = st.date_input(
                "Plot desde",
                value=max(df_union.index.min().date(), (df_union.index.max() - pd.Timedelta(days=90)).date()),
                key="plot_start",
                format="DD/MM/YYYY",
            )
            plot_end = st.date_input(
                "Plot hasta",
                value=df_union.index.max().date(),
                key="plot_end",
                format="DD/MM/YYYY",
            )

        with right:
            # lag manual para esa estación
            lag_row = None
            if "Estacion" in df_lags_edit.columns:
                # algunos dataframes devuelven 'Estacion'
                lag_row = df_lags_edit[df_lags_edit["Estacion"] == station_to_plot]
            elif "estacion" in df_lags_edit.columns:
                lag_row = df_lags_edit[df_lags_edit["estacion"] == station_to_plot]
            else:
                # Si el índice son nombres de estación
                if station_to_plot in df_lags_edit.index:
                    lag_row = df_lags_edit.loc[[station_to_plot]]

            lag_manual = 0
            if lag_row is not None and len(lag_row) > 0:
                if "lag_manual" in lag_row.columns:
                    lag_manual = int(lag_row["lag_manual"].iloc[0])
                elif "lag_optimo" in lag_row.columns:
                    lag_manual = int(lag_row["lag_optimo"].iloc[0])

            df_plot = df_union.loc[
                datetime.combine(plot_start, datetime.min.time()):
                datetime.combine(plot_end, datetime.max.time())
            ][[station_to_plot, obs_col]].copy()

            up_raw = df_plot[station_to_plot]
            up_aligned = _apply_lag_align(up_raw, step=step_td, lag_steps=lag_manual)
            down = df_plot[obs_col]

            plot_df = pd.DataFrame(
                {
                    f"{station_to_plot} (raw)": up_raw,
                    f"{station_to_plot} (aligned +{lag_manual} pasos)": up_aligned,
                    f"{obs_col} (obs)": down,
                }
            )

            st.line_chart(plot_df, height=380, width="stretch")

            st.caption(
                "Interpretación: la serie upstream 'aligned' se desplaza hacia adelante "
                "para comparar con la downstream en el mismo eje temporal."
            )

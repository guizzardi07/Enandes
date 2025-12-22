"""
Streamlit - Tablero paso a paso (v3)

Cambios v3:
- Luego de "Descargar + limpiar": gráfico de TODAS las series (ventana temporal elegible).
- Luego de "Estimar lag óptimo": gráfico con TODAS las series upstream alineadas por su lag (ventana elegible).
- Guarda automáticamente la serie limpia en carpeta `resultados/` al presionar el botón.
- El nombre del archivo se define junto al botón (no en el sidebar).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Imports del proyecto (desde `modulos/`)
def _import_project_functions():
    """
    Importa funciones/clases del paquete `modulos/`.
    Requiere ejecutar Streamlit desde la raíz del proyecto.
    """
    try:
        from modulos.series import construir_series_union
        from modulos.hindcast import evaluar_estaciones_individuales,ajustar_estacion_con_lag
        from modulos.CrosCorrAnalysis_mod import add_constant
        from a5client import Crud
        return construir_series_union, evaluar_estaciones_individuales, add_constant, ajustar_estacion_con_lag, Crud
    except Exception as e:
        raise ImportError(
            "No se pudo importar desde `modulos/`. Verificá que exista `modulos/series.py`, "
            "`modulos/hindcast.py` y `a5client.py`, y que estés corriendo Streamlit "
            "desde la raíz del proyecto."
        ) from e

construir_series_union, evaluar_estaciones_individuales, add_constant, ajustar_estacion_con_lag, Crud = _import_project_functions()

# Utils
def _today_local() -> date:
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
    try:
        return pd.to_timedelta(step_adopt)
    except Exception:
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
    lag_steps = int(lag_steps or 0)
    return up.shift(lag_steps, freq=step)

def _default_plot_window(df: pd.DataFrame, days: int = 90) -> tuple[date, date]:
    """
    Define una ventana temporal por defecto para graficar la serie.

    Devuelve un rango de fechas que abarca como máximo los últimos
    días disponibles en el DataFrame. Si el índice está vacío
    o contiene solo NaN, utiliza la fecha actual como referencia.
    """
    # Fechas mínima y máxima del índice temporal
    idx_min = df.index.min()
    idx_max = df.index.max()

    # Si no hay fechas válidas en el índice, usar una ventana relativa a hoy
    if pd.isna(idx_min) or pd.isna(idx_max):
        today = _today_local()
        return today - timedelta(days=days), today

    # La ventana comienza en el máximo entre:
    # - la fecha mínima disponible
    # - los últimos `days` días respecto de la fecha máxima
    start = max(
        idx_min.date(),
        (idx_max - pd.Timedelta(days=days)).date()
    )

    # La ventana termina en la fecha máxima disponible
    return start, idx_max.date()

# UI config
st.set_page_config(page_title="Pilcomayo - Tablero de control", layout="wide")
st.title("Tablero de control")

# Sidebar (solo config estable)
with st.sidebar:
    st.header("Configuración")

    load_dotenv()
    a5_url = "https://alerta.ina.gob.ar/a6"
    st.text_input("A5_URL (fijo)", value=a5_url, disabled=True)

    a5_token_env = os.getenv("A5_TOKEN", "")
    a5_token = st.text_input("A5_TOKEN", value=a5_token_env, type="password")

    st.divider()

    step_adopt = "h"  # fijo
    st.text_input("Paso temporal (fijo)", value=step_adopt, disabled=True)

    carpeta_figuras = st.text_input(
        "Carpeta de figuras",
        value=str(Path("resultados") / "figuras"),
        help="Dónde guardar los PNG que genera el flujo de limpieza/análisis",
    )

# Session state init
if "df_union" not in st.session_state:
    st.session_state.df_union = None
if "df_resumen" not in st.session_state:
    st.session_state.df_resumen = None
if "lags_df" not in st.session_state:
    st.session_state.lags_df = None
if "obs_col" not in st.session_state:
    st.session_state.obs_col = "Misión La Paz"
if "stations" not in st.session_state:
    st.session_state.stations = [
        {"Estacion": "Misión La Paz", "serie_id": 42293},
        {"Estacion": "Villa Montes", "serie_id": 42291},
        {"Estacion": "Puente Aruma", "serie_id": 42294},
    ]
if "plot_download_start" not in st.session_state:
    st.session_state.plot_download_start = None
if "plot_download_end" not in st.session_state:
    st.session_state.plot_download_end = None
if "plot_aligned_start" not in st.session_state:
    st.session_state.plot_aligned_start = None
if "plot_aligned_end" not in st.session_state:
    st.session_state.plot_aligned_end = None

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
    st.session_state.stations = stations_df.to_dict("records")

with colB:
    st.session_state.obs_col = "Misión La Paz"
    st.info("Estación objetivo: **Misión La Paz**.")

# Paso 1: Descargar + limpiar (y guardar a resultados/)
st.subheader("Paso 1 — Descargar y limpiar series")

today = _today_local()
default_start = today - timedelta(days=365)
default_end = today

c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.8], gap="large")

with c1:
    d_start = st.date_input("Desde", value=default_start, format="DD/MM/YYYY")
with c2:
    d_end = st.date_input("Hasta", value=default_end, format="DD/MM/YYYY")
with c3:
    # Nombre archivo en el cuerpo (no en sidebar)
    archivo_salida_csv = st.text_input(
        "Nombre archivo (CSV)",
        value="series_nivel_union_1H.csv",
        help="Se guardará en resultados/ y también se usará para descarga.",
    )
with c4:
    archivo_salida_resumen = st.text_input(
        "Nombre resumen (Excel)",
        value="resumen_series_niveles_1H.xlsx",
        help="Se guardará en resultados/ (si el flujo lo genera).",
    )

st.caption("Por defecto: hoy − 1 año → hoy.")

run_build = st.button("Descargar + limpiar (construir df_union)", type="primary")

if run_build:
    if not a5_token:
        st.error("Falta A5_TOKEN.")
    else:
        try:
            step_td = _parse_step(step_adopt)

            estaciones_dict: Dict[str, int] = {
                r["Estacion"]: int(r["serie_id"])
                for r in st.session_state.stations
                if r.get("Estacion") and pd.notna(r.get("serie_id"))
            }

            client = Crud(a5_url, token=a5_token)

            timestart = _to_dt_start(d_start)
            timeend = _to_dt_end(d_end)

            # carpetas de salida
            out_dir = Path("resultados")
            out_dir.mkdir(parents=True, exist_ok=True)
            Path(carpeta_figuras).mkdir(parents=True, exist_ok=True)

            # rutas
            out_csv_path = out_dir / Path(archivo_salida_csv).name
            out_resumen_path = out_dir / Path(archivo_salida_resumen).name
            out_xlsx_path = out_csv_path.with_suffix(".xlsx")

            with st.spinner("Descargando, limpiando y unificando series..."):
                df_union, df_resumen = construir_series_union(
                    Estaciones=estaciones_dict,
                    timestart=timestart,
                    timeend=timeend,
                    step_adopt=step_adopt,
                    client=client,
                    carpeta_figuras=str(carpeta_figuras),
                    archivo_salida=str(out_csv_path),
                    archivo_salida_resumen=str(out_resumen_path),
                )

            # guardamos en sesión
            st.session_state.df_union = df_union
            st.session_state.df_resumen = df_resumen

            # guardamos también un Excel “directo” de series limpias (más cómodo para intercambio)
            try:
                df_union.to_csv(out_csv_path, index=True)
                df_union.to_excel(out_xlsx_path, sheet_name="series_limpias")
            except Exception:
                # si el flujo ya lo guarda y hay permisos raros, no cortamos el tablero
                pass

            # inicializamos ventanas por defecto de plot
            pstart, pend = _default_plot_window(df_union, days=90)
            st.session_state.plot_download_start = pstart
            st.session_state.plot_download_end = pend
            st.session_state.plot_aligned_start = pstart
            st.session_state.plot_aligned_end = pend

            st.success(
                f"Listo. df_union: {df_union.shape[0]} filas × {df_union.shape[1]} columnas. "
                f"Guardado en: resultados/{out_csv_path.name}"
            )
        except Exception as e:
            st.exception(e)

# Luego de descargar: preview + descargas + gráfico de series (sin lag)
if st.session_state.df_union is not None:
    df_union: pd.DataFrame = st.session_state.df_union

    st.write("Vista rápida (últimas filas):")
    st.dataframe(df_union.tail(10), width="stretch")

    dcol1, dcol2, dcol3 = st.columns([1, 1, 2], gap="large")

    with dcol1:
        st.download_button(
            "Descargar CSV (series limpias)",
            data=df_union.to_csv().encode("utf-8"),
            file_name="series_limpias.csv",
            mime="text/csv",
        )
    with dcol2:
        excel_bytes = _df_to_excel_bytes(df_union, sheet_name="series_limpias")
        st.download_button(
            "Descargar Excel (series limpias)",
            data=excel_bytes,
            file_name="series_limpias.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dcol3:
        st.caption("Se guardan automáticamente en `resultados/` cuando presionás el botón de descarga+limpieza.")

    st.markdown("Series descargadas")
    if st.session_state.plot_download_start is None or st.session_state.plot_download_end is None:
        pstart, pend = _default_plot_window(df_union, days=90)
        st.session_state.plot_download_start = pstart
        st.session_state.plot_download_end = pend

    g1, g2 = st.columns([1, 1], gap="large")
    with g1:
        plot_download_start = st.date_input(
            "Ventana gráfico (sin lag) - desde",
            value=st.session_state.plot_download_start,
            key="plot_download_start_input",
            format="DD/MM/YYYY",
        )
    with g2:
        plot_download_end = st.date_input(
            "Ventana gráfico (sin lag) - hasta",
            value=st.session_state.plot_download_end,
            key="plot_download_end_input",
            format="DD/MM/YYYY",
        )

    st.session_state.plot_download_start = plot_download_start
    st.session_state.plot_download_end = plot_download_end

    df_plot = df_union.loc[
        datetime.combine(plot_download_start, datetime.min.time()):
        datetime.combine(plot_download_end, datetime.max.time())
    ].copy()

    # si por ventana queda vacío, avisamos
    if df_plot.empty:
        st.warning("La ventana seleccionada no tiene datos.")
    else:
        #st.line_chart(df_plot, height=380, width="stretch")
        fig, ax = plt.subplots(figsize=(12, 4))

        # Graficar todas las series
        for col in df_plot.columns:
            ax.plot(df_plot.index, df_plot[col], label=col)

        ax.set_ylabel("Nivel")
        ax.set_xlabel("Fecha")
        ax.legend()
        
        # Ticks mayores: 1 por mes (labels)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Ticks menores: 1 por día (líneas verticales)
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

        # Grilla mayor (más suave)
        ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.4)

        # Grilla menor SOLO en X (líneas verticales diarias)
        ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.35)

        fig.tight_layout()

        # Mostrar en Streamlit
        st.pyplot(fig, width="stretch")

# Paso 2: Estimar lag óptimo (ventana) + gráfico con lags aplicados a TODAS las upstream
st.subheader("Paso 2 — Lag óptimo por estación + gráfico con lag aplicado")

if st.session_state.df_union is None:
    st.info("Primero ejecutá el Paso 1 para tener df_union.")
else:
    df_union = st.session_state.df_union
    step_td = _parse_step(step_adopt)

    st.write("Elegí la **ventana temporal** para estimar el lag (puede ser menor al rango descargado).")

    idx_min = df_union.index.min()
    idx_max = df_union.index.max()
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

    run_lag = st.button("Estimar lag óptimo", type="primary")

    if run_lag:
        try:
            obs_col = st.session_state.obs_col
            upstream_list = [c for c in df_union.columns if c != obs_col]

            df_sub = df_union.loc[
                datetime.combine(lag_start, datetime.min.time()):
                datetime.combine(lag_end, datetime.max.time())
            ].copy()

            with st.spinner("Estimando lag óptimo por estación..."):
                _, df_lags = evaluar_estaciones_individuales(
                    df_union=df_sub,
                    estaciones=tuple(upstream_list),
                    obs_col=obs_col,
                    max_lag=int(max_lag),
                    ini_lag=int(ini_lag),
                )

            df_lags = df_lags.copy()
            if "lag_optimo" in df_lags.columns:
                df_lags["lag_manual"] = df_lags["lag_optimo"].astype(int)
            elif "lag" in df_lags.columns:
                df_lags["lag_manual"] = df_lags["lag"].astype(int)
            else:
                df_lags["lag_manual"] = 0

            st.session_state.lags_df = df_lags
            st.success("Lag estimado. Podés editar 'lag_manual' y el gráfico de abajo se actualiza.")
        except Exception as e:
            st.exception(e)

    if st.session_state.lags_df is not None:
        df_lags = st.session_state.lags_df

        st.write("Resultado:")
        df_lags_edit = st.data_editor(
            df_lags,
            width="stretch",
            column_config={
                "lag_manual": st.column_config.NumberColumn(
                    "lag_manual",
                    help="Editá a mano si querés. Se interpreta en pasos del step (1H).",
                    step=1,
                )
            },
            key="lags_editor",
        )
        st.session_state.lags_df = df_lags_edit

        st.markdown("### Gráfico — Series con lag aplicado")

        if st.session_state.plot_aligned_start is None or st.session_state.plot_aligned_end is None:
            pstart, pend = _default_plot_window(df_union, days=90)
            st.session_state.plot_aligned_start = pstart
            st.session_state.plot_aligned_end = pend

        g1, g2 = st.columns([1, 1], gap="large")
        with g1:
            plot_aligned_start = st.date_input(
                "Ventana gráfico (con lag) - desde",
                value=st.session_state.plot_aligned_start,
                key="plot_aligned_start_input",
                format="DD/MM/YYYY",
            )
        with g2:
            plot_aligned_end = st.date_input(
                "Ventana gráfico (con lag) - hasta",
                value=st.session_state.plot_aligned_end,
                key="plot_aligned_end_input",
                format="DD/MM/YYYY",
            )

        st.session_state.plot_aligned_start = plot_aligned_start
        st.session_state.plot_aligned_end = plot_aligned_end

        df_plot = df_union.loc[
            datetime.combine(plot_aligned_start, datetime.min.time()):
            datetime.combine(plot_aligned_end, datetime.max.time())
        ].copy()

        if df_plot.empty:
            st.warning("La ventana seleccionada no tiene datos.")
        else:
            obs_col = st.session_state.obs_col
            upstream_cols = [c for c in df_plot.columns if c != obs_col]

            # diccionario estación -> lag_manual
            lag_map = {}

            if "Estacion" in df_lags_edit.columns:
                for _, r in df_lags_edit.iterrows():
                    lag_map[str(r["Estacion"])] = int(r.get("lag_manual", 0))
            elif "estacion" in df_lags_edit.columns:
                for _, r in df_lags_edit.iterrows():
                    lag_map[str(r["estacion"])] = int(r.get("lag_manual", 0))
            else:
                # si los nombres vienen como índice
                for idx, r in df_lags_edit.iterrows():
                    lag_map[str(idx)] = int(r.get("lag_manual", 0))

            aligned_dict = {f"{obs_col} (obs)": df_plot[obs_col]}

            for col in upstream_cols:
                lag = int(lag_map.get(col, 0))
                aligned = _apply_lag_align(df_plot[col], step=step_td, lag_steps=lag)
                aligned_dict[f"{col} (aligned +{lag})"] = aligned

            aligned_df = pd.DataFrame(aligned_dict)
            #st.line_chart(aligned_df, height=420, width="stretch")
            

            fig, ax = plt.subplots(figsize=(12, 4))

            for col in aligned_df.columns:
                ax.plot(aligned_df.index, aligned_df[col], label=col)

            ax.set_ylabel("Nivel")
            ax.set_xlabel("Fecha")
            ax.legend()

            # Ticks mayores: 1 por mes (labels)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            # Ticks menores: 1 por día (líneas verticales)
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

            # Grilla mayor (suave)
            ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.4)

            # Grilla menor SOLO en X (vertical diaria)
            ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.35)

            fig.tight_layout()
            st.pyplot(fig, width="stretch")

            st.caption(
                "Todas las upstream se desplazan según su `lag_manual` (en pasos de 1H) para compararlas con Misión La Paz."
            )


# Paso 3: Ajuste por estación (con lag fijo) + gráficos
st.subheader("Paso 3 — Ajuste por estación (con lag fijo)")

if st.session_state.df_union is None or st.session_state.lags_df is None:
    st.info("Primero ejecutá el Paso 1 y el Paso 2 para tener series y lags.")
else:
    df_union = st.session_state.df_union
    df_lags_edit = st.session_state.lags_df
    obs_col = st.session_state.obs_col

    estaciones_up = [c for c in df_union.columns if c != obs_col]
    if not estaciones_up:
        st.warning("No hay estaciones upstream para ajustar.")
    else:
        csel1, csel2, csel3 = st.columns([1.2, 1, 1], gap="large")
        with csel1:
            est_sel = st.selectbox("Elegí estación upstream", estaciones_up, key="fit_est_sel")
        with csel2:
            min_muestras_fit = st.number_input("min_muestras", min_value=20, max_value=2000, value=200, step=10)
        with csel3:
            # lag manual (solo lectura acá) — se edita arriba
            pass

        # Buscar lag_manual para la estación seleccionada
        lag_sel = 0
        try:
            if "Estacion" in df_lags_edit.columns:
                lag_sel = int(df_lags_edit.loc[df_lags_edit["Estacion"].astype(str) == str(est_sel), "lag_manual"].iloc[0])
            elif "estacion" in df_lags_edit.columns:
                lag_sel = int(df_lags_edit.loc[df_lags_edit["estacion"].astype(str) == str(est_sel), "lag_manual"].iloc[0])
            else:
                # si vienen como índice
                if str(est_sel) in df_lags_edit.index.astype(str):
                    lag_sel = int(df_lags_edit.loc[str(est_sel), "lag_manual"])
        except Exception:
            lag_sel = 0

        st.info(f"Usando lag={lag_sel} pasos (1H) para **{est_sel}**.")

        # Ventana temporal para graficar el ajuste
        pstart, pend = _default_plot_window(df_union[[obs_col, est_sel]].dropna(how="all"), days=90)
        g1, g2 = st.columns([1, 1], gap="large")
        with g1:
            fit_start = st.date_input("Ventana ajuste - desde", value=pstart, key="fit_plot_start", format="DD/MM/YYYY")
        with g2:
            fit_end = st.date_input("Ventana ajuste - hasta", value=pend, key="fit_plot_end", format="DD/MM/YYYY")

        # Ajustar modelo con lag fijo
        try:
            y_obs, y_fit, modelo = ajustar_estacion_con_lag(
                df_union=df_union,
                est=est_sel,
                obs_col=obs_col,
                lag=lag_sel,
                min_muestras=int(min_muestras_fit),
            )

            # Recortar ventana
            mask = (
                (y_obs.index >= datetime.combine(fit_start, datetime.min.time())) &
                (y_obs.index <= datetime.combine(fit_end, datetime.max.time()))
            )
            y_obs_p = y_obs.loc[mask]
            y_fit_p = y_fit.loc[mask]

            if y_obs_p.empty:
                st.warning("La ventana seleccionada no tiene datos simultáneos para el ajuste.")
            else:
                # ---- Gráfico temporal: Observado vs Ajustado ----
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(y_obs_p.index, y_obs_p, label=f"{obs_col} (obs)", linewidth=2)
                ax.plot(y_fit_p.index, y_fit_p, label=f"{est_sel} → {obs_col} (fit, lag={lag_sel})", linestyle="--")

                ax.set_ylabel("Nivel")
                ax.set_xlabel("Fecha")
                ax.set_title(f"Ajuste lineal con lag fijo — {est_sel} (lag={lag_sel} h)")
                ax.legend()

                # Eje tiempo + grilla diaria (como los otros gráficos)
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

                ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.4)
                ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.35)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")

                # ---- Scatter Obs vs Fit ----
                st.markdown("**Diagnóstico — Observado vs Ajustado**")
                fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
                ax2.scatter(y_obs, y_fit, s=8, alpha=0.5)
                # línea 1:1
                vmin = float(np.nanmin([y_obs.min(), y_fit.min()]))
                vmax = float(np.nanmax([y_obs.max(), y_fit.max()]))
                ax2.plot([vmin, vmax], [vmin, vmax], linestyle="--")
                ax2.set_xlabel("Observado")
                ax2.set_ylabel("Ajustado")
                ax2.set_title(f"Obs vs Fit — {est_sel} (R²={modelo.rsquared:.3f})")
                ax2.grid(True, linestyle=":", alpha=0.3)
                fig2.tight_layout()
                st.pyplot(fig2, width="content")

                # ---- Métricas ----
                st.markdown("**Métricas**")
                st.write({
                    "R²": round(float(modelo.rsquared), 4),
                    "n": int(modelo.nobs),
                    "intercepto (const)": round(float(modelo.params.get("const", np.nan)), 4),
                    "pendiente (up_lag)": round(float(modelo.params.get("up_lag", np.nan)), 4),
                })

        except Exception as e:
            st.warning(f"No se pudo ajustar el modelo para {est_sel}: {e}")

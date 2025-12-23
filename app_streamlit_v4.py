from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import matplotlib.dates as mdates

# Uso:  streamlit run app_streamlit_v4.py

# Imports del proyecto (módulos)
def _import_project_functions():
    """
    Importa funciones/clases del paquete `modulos/`.
    Requiere ejecutar Streamlit desde la raíz del proyecto.
    """
    try:
        from modulos.series import construir_series_union
        from modulos.hindcast import (
            evaluar_estaciones_individuales,
            ajustar_estacion_con_lag,
            forecast_from_upstream,
            estimar_lags_por_estacion,
            get_lag_for_station,
            forecast_horizon_from_upstream_last

        )
        from a5client import Crud
        return (
            construir_series_union,
            evaluar_estaciones_individuales,
            ajustar_estacion_con_lag,
            forecast_from_upstream,
            estimar_lags_por_estacion,
            get_lag_for_station,
            forecast_horizon_from_upstream_last,
            Crud,
        )
    except Exception as e:
        raise ImportError(
            "No se pudo importar desde `modulos/`. Verificá que exista `modulos/series.py`, "
            "`modulos/hindcast.py` y `a5client.py`, y que estés corriendo Streamlit "
            "desde la raíz del proyecto."
        ) from e

(   construir_series_union,
    evaluar_estaciones_individuales,
    ajustar_estacion_con_lag,
    forecast_from_upstream,
    estimar_lags_por_estacion,
    get_lag_for_station,
    forecast_horizon_from_upstream_last,
    Crud,
) = _import_project_functions()

from modulos.io_utils import df_to_excel_bytes, df_to_csv_bytes, safe_filename
from modulos.utils_time import (
    today_local,
    to_dt_start,
    to_dt_end,
    parse_step,
    default_plot_window_from_index)
from modulos.utils_series import apply_lag_shift_series
from modulos.plotting import plot_timeseries_daily_grid, plot_scatter_obs_fit

# UI config
st.set_page_config(page_title="Pilcomayo - Tablero de control", layout="wide")
st.title("Tablero de control")

# Config estable (sidebar)
load_dotenv()
A5_URL_FIJO = "https://alerta.ina.gob.ar/a6"

with st.sidebar:
    st.header("Configuración")
    st.text_input("A5_URL (fijo)", value=A5_URL_FIJO, disabled=True)

    a5_token_env = os.getenv("A5_TOKEN", "")
    a5_token = st.text_input("A5_TOKEN", value=a5_token_env, type="password")

# Session state init  / Estado persistente entre interacciones

if "df_union" not in st.session_state:              # Estado persistente entre interacciones
    st.session_state.df_union = None                
if "df_resumen" not in st.session_state:            # resumen de limpieza
    st.session_state.df_resumen = None
if "lags_df" not in st.session_state:               # DataFrame con lag_optimo y lag_manual
    st.session_state.lags_df = None

# downstream fijo
if "obs_col" not in st.session_state:               # estación objetivo
    st.session_state.obs_col = "Misión La Paz"

# estaciones editables
if "stations" not in st.session_state:              # lista editable de estaciones con Estacion y serie_id
    st.session_state.stations = [
        {"Estacion": "Misión La Paz", "serie_id": 42293},
        {"Estacion": "Villa Montes", "serie_id": 42291},
        {"Estacion": "Puente Aruma", "serie_id": 42294},]

# step fijo, visible
if "step_adopt" not in st.session_state:
    st.session_state.step_adopt = "h"  # 1H fijo

# carpeta figuras
if "carpeta_figuras" not in st.session_state:
    st.session_state.carpeta_figuras = str(Path("resultados") / "figuras")

# ventanas de plot
if "plot_download_start" not in st.session_state:
    st.session_state.plot_download_start = None
if "plot_download_end" not in st.session_state:
    st.session_state.plot_download_end = None
if "plot_aligned_start" not in st.session_state:
    st.session_state.plot_aligned_start = None
if "plot_aligned_end" not in st.session_state:
    st.session_state.plot_aligned_end = None

# nombres para descargas
if "archivo_descarga_csv" not in st.session_state:
    st.session_state.archivo_descarga_csv = "series_limpias.csv"
if "archivo_descarga_xlsx" not in st.session_state:
    st.session_state.archivo_descarga_xlsx = "series_limpias.xlsx"

# Estaciones
st.subheader("Estaciones")

colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.write("Editá la lista de estaciones (nombre + `serie_id`).")
    stations_df = pd.DataFrame(st.session_state.stations)
    stations_df = st.data_editor(
        stations_df,
        num_rows="dynamic",
        width='stretch',
        column_config={
            "Estacion": st.column_config.TextColumn(required=True),
            "serie_id": st.column_config.NumberColumn(required=True, step=1),
        },
        key="stations_editor",
    )
    st.session_state.stations = stations_df.to_dict("records")

with colB:
    st.info("Estación objetivo : **Misión La Paz** (fija).")

# Paso 1
st.subheader("Paso 1 — Descargar y limpiar series")

p1a, p1b = st.columns([1, 2], gap="large")
with p1a:
    st.text_input(
        "Paso temporal",
        value=st.session_state.step_adopt,
        disabled=True,
        help="Paso temporal fijo del sistema (1 hora).",
    )
with p1b:
    st.session_state.carpeta_figuras = st.text_input(
        "Carpeta de figuras",
        value=st.session_state.carpeta_figuras,
        help="Dónde guardar los PNG que genera el flujo de limpieza/análisis.",
        key="carpeta_figuras_input",
    )

today = today_local()
default_start = today - timedelta(days=365)
default_end = today

c1, c2, c3 = st.columns([1, 1, 2], gap="large")
with c1:
    d_start = st.date_input("Desde", value=default_start, format="DD/MM/YYYY", key="download_from")
with c2:
    d_end = st.date_input("Hasta", value=default_end, format="DD/MM/YYYY", key="download_to")
with c3:
    st.caption("Por defecto: hoy − 1 año → hoy.")

run_build = st.button("Descargar + limpiar (construir df_union)", type="primary")

if run_build:
    if not a5_token:
        st.error("Falta A5_TOKEN.")
    else:
        try:
            step_adopt = st.session_state.step_adopt
            _ = parse_step(step_adopt)  # valida, aunque sea fijo

            estaciones_dict: Dict[str, int] = {
                r["Estacion"]: int(r["serie_id"])
                for r in st.session_state.stations
                if r.get("Estacion") and pd.notna(r.get("serie_id"))
            }

            client = Crud(A5_URL_FIJO, token=a5_token)

            timestart = to_dt_start(d_start)
            timeend = to_dt_end(d_end)

            # carpetas de salida
            out_dir = Path("resultados")
            out_dir.mkdir(parents=True, exist_ok=True)
            Path(st.session_state.carpeta_figuras).mkdir(parents=True, exist_ok=True)

            # nombres de salida para guardado
            out_csv_path = out_dir / "series_nivel_union_1H.csv"
            out_resumen_path = out_dir / "resumen_series_niveles_1H.xlsx"
            out_xlsx_path = out_csv_path.with_suffix(".xlsx")

            with st.spinner("Descargando, limpiando y unificando series..."):
                df_union, df_resumen = construir_series_union(
                    Estaciones=estaciones_dict,
                    timestart=timestart,
                    timeend=timeend,
                    step_adopt=step_adopt,
                    client=client,
                    carpeta_figuras=str(st.session_state.carpeta_figuras),
                    archivo_salida=str(out_csv_path),
                    archivo_salida_resumen=str(out_resumen_path),
                )

            st.session_state.df_union = df_union
            st.session_state.df_resumen = df_resumen

            # Guardado adicional
            try:
                df_union.to_csv(out_csv_path, index=True)
                df_union.to_excel(out_xlsx_path, sheet_name="series_limpias")
            except Exception:
                pass

            # inicializar ventanas por defecto de plot
            pstart, pend = default_plot_window_from_index(df_union.index, days=90)
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

# Paso 1 outputs
if st.session_state.df_union is not None:
    df_union: pd.DataFrame = st.session_state.df_union

    st.write("Vista rápida (últimas filas):")
    st.dataframe(df_union.tail(10), width='stretch')

    # nombres de descarga
    n1, n2 = st.columns([1.2, 1.8], gap="large")
    with n1:
        st.session_state.archivo_descarga_csv = st.text_input(
            "Nombre archivo (CSV)",
            value=st.session_state.archivo_descarga_csv,
            key="archivo_descarga_csv_input",
        )
    with n2:
        st.session_state.archivo_descarga_xlsx = st.text_input(
            "Nombre resumen (Excel)",
            value=st.session_state.archivo_descarga_xlsx,
            key="archivo_descarga_xlsx_input",
        )

    csv_name = safe_filename(st.session_state.archivo_descarga_csv, "series_limpias.csv")
    xlsx_name = safe_filename(st.session_state.archivo_descarga_xlsx, "series_limpias.xlsx")

    dcol1, dcol2, dcol3 = st.columns([1, 1, 2], gap="large")
    with dcol1:
        st.download_button(
            "Descargar CSV (series limpias)",
            data=df_to_csv_bytes(df_union, index=True),
            file_name=csv_name,
            mime="text/csv",
        )
    with dcol2:
        excel_bytes = df_to_excel_bytes(df_union, sheet_name="series_limpias")
        st.download_button(
            "Descargar Excel (series limpias)",
            data=excel_bytes,
            file_name=xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dcol3:
        st.caption("Se guardan automáticamente en `resultados/` al presionar Descargar + limpiar.")

    st.markdown("### Series descargadas (sin lag)")

    if st.session_state.plot_download_start is None or st.session_state.plot_download_end is None:
        pstart, pend = default_plot_window_from_index(df_union.index, days=90)
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

    if df_plot.empty:
        st.warning("La ventana seleccionada no tiene datos.")
    else:
        fig = plot_timeseries_daily_grid(df_plot, ylabel="Nivel")
        st.pyplot(fig, width='stretch')

# Paso 2
st.subheader("Paso 2 — Lag óptimo por estación + gráfico con lag aplicado")

if st.session_state.df_union is None:
    st.info("Primero ejecutá el Paso 1 para tener df_union.")
else:
    df_union = st.session_state.df_union
    step_td = parse_step(st.session_state.step_adopt)

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
        max_lag = st.number_input("max_lag (pasos)", min_value=1, max_value=500, value=72, step=1, key="max_lag")
    with c4:
        ini_lag = st.number_input("ini_lag (pasos)", min_value=0, max_value=200, value=2, step=1, key="ini_lag")

    run_lag = st.button("Estimar lag óptimo", type="primary", key="run_lag_btn")

    if run_lag:
        try:
            obs_col = st.session_state.obs_col
            upstream_list = [c for c in df_union.columns if c != obs_col]

            df_sub = df_union.loc[
                datetime.combine(lag_start, datetime.min.time()):
                datetime.combine(lag_end, datetime.max.time())
            ].copy()

            with st.spinner("Estimando lag óptimo por estación..."):
                df_lags = estimar_lags_por_estacion(
                    df_union=df_sub,
                    estaciones=tuple(upstream_list),
                    obs_col=obs_col,
                    max_lag=int(max_lag),
                    ini_lag=int(ini_lag),
                )

            # Unificamos: lag_manual arranca igual a lag_optimo
            df_lags = df_lags.copy()
            if "lag_optimo" not in df_lags.columns:
                raise KeyError("estimar_lags_por_estacion debe devolver columna 'lag_optimo'")
            if "Estacion" not in df_lags.columns and "estacion" not in df_lags.columns:
                # si viniera con índice, lo “subimos” a columna Estacion
                df_lags = df_lags.reset_index().rename(columns={"index": "Estacion"})

            df_lags["lag_manual"] = df_lags["lag_optimo"].astype(int)

            st.session_state.lags_df = df_lags
            st.success("Lag estimado. Podés editar 'lag_manual' y el gráfico de abajo se actualiza.")
        except Exception as e:
            st.exception(e)

    if st.session_state.lags_df is not None:
        df_lags = st.session_state.lags_df

        st.write("Resultado:")
        df_lags_edit = st.data_editor(
            df_lags,
            width='stretch',
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
            pstart, pend = default_plot_window_from_index(df_union.index, days=90)
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
            lag_map: Dict[str, int] = {}

            if "Estacion" in df_lags_edit.columns:
                for _, r in df_lags_edit.iterrows():
                    lag_map[str(r["Estacion"])] = int(r.get("lag_manual", 0))
            elif "estacion" in df_lags_edit.columns:
                for _, r in df_lags_edit.iterrows():
                    lag_map[str(r["estacion"])] = int(r.get("lag_manual", 0))
            else:
                for idx, r in df_lags_edit.iterrows():
                    lag_map[str(idx)] = int(r.get("lag_manual", 0))

            aligned_dict = {f"{obs_col} (obs)": df_plot[obs_col]}

            for col in upstream_cols:
                lag = int(lag_map.get(col, 0))
                aligned = apply_lag_shift_series(df_plot[col], step=step_td, lag_steps=lag)
                aligned_dict[f"{col} (aligned +{lag})"] = aligned

            aligned_df = pd.DataFrame(aligned_dict)
            fig = plot_timeseries_daily_grid(
                aligned_df,
                ylabel="Nivel",
                title="Series alineadas por lag_manual",
            )
            st.pyplot(fig, width='stretch')

            st.caption("Todas las upstream se desplazan según su `lag_manual` (en pasos de 1H).")

# Paso 3 — Ajuste
st.subheader("Paso 3 — Ajuste y pronóstico (modelo + lag)")

if (st.session_state.df_union is None) or (st.session_state.lags_df is None):
    st.info("Ejecutá Paso 1 y Paso 2 para habilitar el ajuste (df_union + lags_df).")
else:
    df_union = st.session_state.df_union
    df_lags_edit = st.session_state.lags_df
    obs_col = st.session_state.obs_col

    upstream_cols_all = [c for c in df_union.columns if c != obs_col]
    est_sel = st.selectbox("Elegí estación upstream", upstream_cols_all, key="fit_station_select")

    lag_sel = get_lag_for_station(st.session_state.lags_df, est_sel, default=0)
    st.caption(f"Lag usado (lag_manual): **{lag_sel} h**")

    pstart, pend = default_plot_window_from_index(df_union.index, days=90)

    st.markdown("### Ventana de ajuste (calibración)")
    fs1, fs2 = st.columns(2, gap="large")
    with fs1:
        fit_start = st.date_input(
            "AJUSTE - desde",
            value=pstart,
            key="fit_start",
            format="DD/MM/YYYY",
        )
    with fs2:
        fit_end = st.date_input(
            "AJUSTE - hasta",
            value=pend,
            key="fit_end",
            format="DD/MM/YYYY",
        )

    st.markdown("### Ventana de gráfico / evaluación")
    same_window = st.checkbox("Usar la misma ventana para el gráfico", value=True, key="fit_same_window")

    if same_window:
        plot_start, plot_end = fit_start, fit_end
    else:
        ps1, ps2 = st.columns(2, gap="large")
        with ps1:
            plot_start = st.date_input(
                "GRÁFICO - desde",
                value=fit_start,
                key="plot_fit_start",
                format="DD/MM/YYYY",
            )
        with ps2:
            plot_end = st.date_input(
                "GRÁFICO - hasta",
                value=fit_end,
                key="plot_fit_end",
                format="DD/MM/YYYY",
            )

    df_fit = df_union.loc[
        datetime.combine(fit_start, datetime.min.time()):
        datetime.combine(fit_end, datetime.max.time())
    ].copy()

    df_plot = df_union.loc[
        datetime.combine(plot_start, datetime.min.time()):
        datetime.combine(plot_end, datetime.max.time())
    ].copy()

    if df_fit.empty:
        st.warning("La ventana de AJUSTE no tiene datos.")
        st.stop()

    try:
        y_obs_fit, y_fit_fit, modelo = ajustar_estacion_con_lag(
            df_union=df_fit,
            est=est_sel,
            obs_col=obs_col,
            lag=lag_sel,
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    # Métricas del ajuste
    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        st.metric("R² (ajuste)", f"{float(getattr(modelo, 'rsquared', float('nan'))):.3f}")
    with m2:
        st.metric("n (ajuste)", f"{int(getattr(modelo, 'nobs', 0))}")
    with m3:
        try:
            st.metric("Intercepto", f"{float(modelo.params.get('const', float('nan'))):.3f}")
        except Exception:
            st.metric("Intercepto", "—")
    with m4:
        try:
            st.metric("Pendiente", f"{float(modelo.params.get('up_lag', float('nan'))):.3f}")
        except Exception:
            st.metric("Pendiente", "—")

    st.markdown("### Ajuste")
    df_fit_plot = pd.DataFrame(
        {
            f"{obs_col} (obs)": y_obs_fit,
            f"Fit (desde {est_sel}, lag {lag_sel}h)": y_fit_fit,
        }
    )
    fig_fit = plot_timeseries_daily_grid(
        df_fit_plot,
        ylabel="Nivel",
        title=f"Ajuste {est_sel} → {obs_col} (ventana de ajuste)",
    )
    st.pyplot(fig_fit, width='stretch')

    st.markdown("### Scatter — Obs vs Fit")
    fig_sc_fit = plot_scatter_obs_fit(
        y_obs_fit,
        y_fit_fit,
        title="Obs vs Fit",
        figsize=(2.75, 2.75),
        s=10,
        fontsize=9,
        ticksize=8,
    )
    st.pyplot(fig_sc_fit, width='content')

    # Pronóstico aplicado a df_plot
    # st.markdown("### Pronóstico aplicado (ventana de GRÁFICO / evaluación)")

    # if df_plot.empty:
    #     st.warning("La ventana de GRÁFICO no tiene datos.")
    #     st.stop()

    # y_hat_plot = forecast_from_upstream(
    #     df=df_plot,
    #     est=est_sel,
    #     obs_col=obs_col,
    #     lag=lag_sel,
    #     modelo=modelo,
    #     freq="1h",
    # )

    # y_obs_plot = df_plot[obs_col]

    # df_pred_plot = pd.DataFrame(
    #     {
    #         f"{obs_col} (obs)": y_obs_plot,
    #         f"Pronóstico (desde {est_sel}, lag {lag_sel}h)": y_hat_plot,
    #     }
    # )

    # fig_pred = plot_timeseries_daily_grid(
    #     df_pred_plot,
    #     ylabel="Nivel",
    #     title="Observado vs Pronóstico (ventana de gráfico)",
    # )
    # st.pyplot(fig_pred, width='stretch')

    # st.markdown("### Scatter — Obs vs Pronóstico")
    # fig_sc_eval = plot_scatter_obs_fit(
    #     y_obs_plot,
    #     y_hat_plot,
    #     title="Obs vs Pronóstico (evaluación)",
    #     figsize=(2.75, 2.75),
    #     s=10,
    #     fontsize=9,
    #     ticksize=8,
    # )
    # st.pyplot(fig_sc_eval, width='content')

# Paso 4 — Pronóstico operativo
st.subheader("Paso 4 — Última semana + ajuste + pronóstico operativo")

if (st.session_state.df_union is None) or (st.session_state.lags_df is None):
    st.info("Ejecutá Paso 1 y Paso 2.")
else:
    df_union = st.session_state.df_union
    lags_df = st.session_state.lags_df
    obs_col = st.session_state.obs_col

    upstream_all = [c for c in df_union.columns if c != obs_col]
    upstream_sel = st.multiselect(
        "Estaciones upstream para pronóstico",
        options=upstream_all,
        default=upstream_all[:2],
        max_selections=2,
        key="fcst_upstream_sel",
    )
    if len(upstream_sel) == 0:
        st.stop()

    # Ventana de ajuste (calibración)
    pstart, pend = default_plot_window_from_index(df_union.index, days=90)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fit_start = st.date_input("AJUSTE (batch) - desde", value=pstart, key="fit4_start", format="DD/MM/YYYY")
    with c2:
        fit_end = st.date_input("AJUSTE (batch) - hasta", value=pend, key="fit4_end", format="DD/MM/YYYY")

    df_fit = df_union.loc[
        datetime.combine(fit_start, datetime.min.time()):
        datetime.combine(fit_end, datetime.max.time())
    ].copy()

    if df_fit.empty:
        st.warning("La ventana de AJUSTE no tiene datos.")
        st.stop()
    
    # Ventana "última semana" (para ver ajuste reciente)
    obs = df_union[obs_col].dropna()
    if obs.empty:
        st.warning("No hay observado en estación objetivo.")
        st.stop()

    t_emit = obs.index.max()  # fecha de emisión del prono (último obs disponible)
    t_start = t_emit - pd.Timedelta(days=7)

    df_week = df_union.loc[t_start:t_emit].copy()
    obs_lastweek = df_week[obs_col].rename(f"{obs_col} (obs)")

    forecasts = []
    meta_rows = []

    for est in upstream_sel:
        lag = int(get_lag_for_station(lags_df, est, default=0))

        # 1) Ajuste del modelo en df_fit
        y_obs_fit, y_fit_fit, modelo = ajustar_estacion_con_lag(
            df_union=df_fit,
            est=est,
            obs_col=obs_col,
            lag=lag,
        )

        # 2) "Ajuste reciente" (hindcast) sobre la última semana
        y_hist_week = forecast_from_upstream(
            df=df_week,
            est=est,
            obs_col=obs_col,
            lag=lag,
            modelo=modelo,
            freq="1h",
        ).rename(f"Ajuste semana ({est}, lag {lag}h)")

        # 3) Pronóstico hacia adelante (futuro)
        y_fcst = forecast_horizon_from_upstream_last(
            df_union=df_union,
            est=est,
            obs_col=obs_col,
            lag=lag,
            modelo=modelo,
            freq="1h",
        ).rename(f"Pronóstico ({est}, lag {lag}h)")

        if y_fcst.empty:
            st.warning(
                f"No se generó pronóstico futuro con {est} (lag {lag}h). "
                "Probablemente upstream no tiene datos más recientes que downstream."
            )

        # 4) Curva continua: semana previa (ajuste) + futuro (pronóstico)
        y_full = pd.concat(
            [
                y_hist_week.loc[:t_emit],
                y_fcst.loc[y_fcst.index > t_emit],
            ]
        )
        y_full = y_full[~y_full.index.duplicated(keep="first")]
        y_full = y_full.rename(f"Modelo ({est}, lag {lag}h)")

        forecasts.append(y_full)

        meta_rows.append({
            "Estacion": est,
            "lag_manual_h": lag,
            "R2_ajuste": float(getattr(modelo, "rsquared", float("nan"))),
            "n_ajuste": int(getattr(modelo, "nobs", 0)),
            "const": float(modelo.params.get("const", np.nan)),
            "beta_up_lag": float(modelo.params.get("up_lag", np.nan)),
        })

    meta = pd.DataFrame(meta_rows)
    st.markdown("### Resumen modelos (batch)")
    st.dataframe(meta, width='stretch')

    # Dataset final para plot y export
    df_final = pd.concat([obs_lastweek] + forecasts, axis=1)

    # Plot final + línea vertical de emisión
    st.markdown("### Última semana (obs) + ajuste + pronóstico")

    st.pyplot(fig, width='stretch')

    fig = plot_timeseries_daily_grid(
        df_final,
        ylabel="Nivel",
        title="Observado (última semana) + modelo (ajuste + pronóstico)",
    )
    ax = fig.axes[0]

    # EJE X:
    # Ticks mayores SOLO a las 00 y 12
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    # Día arriba, hora abajo
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    # Ticks menores cada 3 horas (solo grilla)
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=0)

    # Grillas
    ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.4)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.35)

    # Línea vertical de emisión
    ax.axvline(t_emit, linestyle="--", linewidth=1)
    ax.text(
        t_emit,
        ax.get_ylim()[1],
        " Emisión prono",
        va="top",
        ha="left",
        fontsize=8,
    )

    fig.tight_layout()
    st.pyplot(fig, width='content')

    # Descargas
    st.markdown("### Descargar series (obs + modelo)")
    df_export = df_final.copy()
    df_export.index.name = "Fecha"
    df_export = df_export.reset_index()

    st.download_button(
        "Descargar CSV",
        data=df_to_csv_bytes(df_export, index=False),
        file_name="ultima_semana_ajuste_y_pronostico.csv",
        mime="text/csv",
    )

    st.download_button(
        "Descargar Excel",
        data=df_to_excel_bytes(df_export, sheet_name="series"),
        file_name="ultima_semana_ajuste_y_pronostico.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


 

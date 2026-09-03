from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

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
from modulos.limpieza_series import get_params_limpieza
from modulos import plan_builder

try:
    from modulos.subestacional import (
        StationConfig,
        CleaningConfig,
        ForecastConfig,
        AnalogyConfig,
        prepare_regular_series,
        resample_series,
        forecast_persistence,
        forecast_analogy,
        plot_analogy_traces,
        plot_forecasts_comparison,
        plot_forecast_boxplot,
        get_selected_row,
        infer_forecast_origin,
        get_last_valid_forecast_origin,
        MONTH_NAMES,
    )
except Exception as e:
    raise ImportError(
        "No se pudo importar `modulos/subestacional.py`. "
        "Copiá el módulo subestacional dentro de la carpeta `modulos/`."
    ) from e
def round_numeric_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Redondea columnas numéricas sin afectar el índice."""
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].round(decimals)
    return df_out


def clean_plotly_label(col: object) -> str:
    """Devuelve una etiqueta corta para leyenda y tooltip de Plotly."""
    label = str(col)

    # Casos conocidos de estaciones. Sirve aunque la columna tenga sufijos
    # como "(obs)", "aligned", "lag" o "shift".
    known_stations = ["Misión La Paz", "Villa Montes", "Puente Aruma"]
    for station in known_stations:
        if station in label:
            return station

    # Caso general: Modelo (Nombre estación, lag 30h, shift +0.000m)
    if label.startswith("Modelo ("):
        clean = label.replace("Modelo (", "", 1)
        clean = clean.split(",", 1)[0]
        return clean.strip().rstrip(")")

    # Fallback: cortar sufijos entre paréntesis.
    return label.split("(", 1)[0].strip()


def build_interactive_timeseries(df: pd.DataFrame, title: str, ylabel: str = "Nivel") -> go.Figure:
    """Arma un gráfico interactivo con Plotly a partir de un DataFrame indexado por fecha."""
    fig = go.Figure()
    for col in df.columns:
        clean_name = clean_plotly_label(col)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=clean_name,
                hovertemplate=f"{clean_name}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=ylabel,
        hovermode="x unified",
        legend_title_text="Series",
        template="plotly_white",
        height=420,
    )
    return fig

def build_interactive_subseasonal_forecast(
    df_hist: pd.DataFrame,
    persistence: pd.DataFrame,
    analogy: pd.DataFrame,
    value_col: str,
) -> go.Figure:

    fig = go.Figure()

    # ---------------------------------------------------------
    # Observado
    # ---------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df_hist["fecha"],
            y=df_hist[value_col],
            mode="lines+markers",
            name="Observado",
            hovertemplate="Observado: %{y:.2f}<extra></extra>",
        )
    )

    # ---------------------------------------------------------
    # Conectar pronósticos con último observado
    # ---------------------------------------------------------

    last_obs_date = df_hist["fecha"].iloc[-1]
    last_obs_value = df_hist[value_col].iloc[-1]

    persistence_plot = pd.concat(
        [
            pd.DataFrame(
                {
                    "fecha": [last_obs_date],
                    "Prono": [last_obs_value],
                }
            ),
            persistence[["fecha", "Prono"]],
        ],
        ignore_index=True,
    )

    analogy_plot = pd.concat(
        [
            pd.DataFrame(
                {
                    "fecha": [last_obs_date],
                    "Prono": [last_obs_value],
                }
            ),
            analogy[["fecha", "Prono"]],
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Persistencia
    # ---------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=persistence_plot["fecha"],
            y=persistence_plot["Prono"],
            mode="lines+markers",
            name="Persistencia",
            hovertemplate="Persistencia: %{y:.2f}<extra></extra>",
        )
    )

    # ---------------------------------------------------------
    # Analogía
    # ---------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=analogy_plot["fecha"],
            y=analogy_plot["Prono"],
            mode="lines+markers",
            name="Analogía",
            hovertemplate="Analogía: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Pronóstico mensual",
        xaxis_title="Fecha",
        yaxis_title="Nivel [m]",
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )

    return fig


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
if "archivo_descarga_nombre" not in st.session_state:
    st.session_state.archivo_descarga_nombre = "series_limpias"

# shift vertical final por estación (operativo)
if "shift_map" not in st.session_state:
    st.session_state.shift_map = {}


def render_pronostico_operativo() -> None:
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
    default_start = today - timedelta(days=90)
    default_end = today
    st.markdown("**Seleccionar fechas de descarga**")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        d_start = st.date_input("Desde", value=default_start, format="DD/MM/YYYY", key="download_from")
    with c2:
        d_end = st.date_input("Hasta", value=default_end, format="DD/MM/YYYY", key="download_to")

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

                df_union = round_numeric_df(df_union, decimals=2)

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
        st.dataframe(df_union.tail(5), width='stretch')

        # nombres de descarga
        # n1 = st.columns([1.2], gap="large")
        # with n1:
        st.session_state.archivo_descarga_nombre = st.text_input(
            "Nombre del archivo de salida",
            value=st.session_state.archivo_descarga_nombre,
            key="archivo_descarga_nombre_input",
        )

        formato_descarga = st.radio(
            "Formato de descarga",
            options=["csv", "excel"],
            horizontal=True,
            key="formato_descarga_paso1",
        )

        base_name = safe_filename(st.session_state.archivo_descarga_nombre, "series_limpias")
        base_name = Path(base_name).stem
        df_union_export = round_numeric_df(df_union, decimals=2)

        if formato_descarga == "csv":
            file_name = f"{base_name}.csv"
            file_bytes = df_to_csv_bytes(df_union_export, index=True)
            mime = "text/csv"
            button_label = "Descargar CSV (series limpias)"
        else:
            file_name = f"{base_name}.xlsx"
            file_bytes = df_to_excel_bytes(df_union_export, sheet_name="series_limpias")
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            button_label = "Descargar Excel (series limpias)"

        dcol1, dcol2 = st.columns([1, 2], gap="large")
        with dcol1:
            st.download_button(
                button_label,
                data=file_bytes,
                file_name=file_name,
                mime=mime,
            )
        with dcol2:
            st.caption("Guarda en la carpeta Descargas del navegador.")

        st.markdown("### Series descargadas")
        st.caption("Seleccione fechas para visualizar la serie:")
        if st.session_state.plot_download_start is None or st.session_state.plot_download_end is None:
            pstart, pend = default_plot_window_from_index(df_union.index, days=90)
            st.session_state.plot_download_start = pstart
            st.session_state.plot_download_end = pend

        g1, g2 = st.columns([1, 1], gap="large")
        with g1:
            plot_download_start = st.date_input(
                "Ventana gráfico - desde",
                value=st.session_state.plot_download_start,
                key="plot_download_start_input",
                format="DD/MM/YYYY",
            )
        with g2:
            plot_download_end = st.date_input(
                "Ventana gráfico - hasta",
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
    st.subheader("Paso 2 — Lag óptimo por estación.")

    if st.session_state.df_union is None:
        st.info("Primero ejecutá el Paso 1 para tener df_union.")
    else:
        df_union = st.session_state.df_union
        step_td = parse_step(st.session_state.step_adopt)

        st.markdown("<p style='font-size:1.15rem; font-weight:600;'>Elegí la ventana temporal para estimar el lag (dentro de la ventana descargada).</p>", unsafe_allow_html=True)

        idx_min = df_union.index.min()
        idx_max = df_union.index.max()
        default_lag_start = max(idx_min.date(), (idx_max - pd.Timedelta(days=365)).date())
        default_lag_end = idx_max.date()

        # Sync lag window -> plot+fit
        def _sync_windows_from_lag():
            ls = st.session_state.get("lag_start")
            le = st.session_state.get("lag_end")
            if ls is None or le is None:
                return
            
            ls_ord = min(ls, le)
            le_ord = max(ls, le)

            st.session_state.plot_aligned_start = ls_ord
            st.session_state.plot_aligned_end = le_ord
            st.session_state.fit_start_unif = ls_ord
            st.session_state.fit_end_unif = le_ord

        # defaults
        if "lag_start" not in st.session_state:
            st.session_state.lag_start = default_lag_start
        if "lag_end" not in st.session_state:
            st.session_state.lag_end = default_lag_end

        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            lag_start = st.date_input(
                "Ventana lag - desde",
                key="lag_start",
                format="DD/MM/YYYY",
                on_change=_sync_windows_from_lag,
            )
        with c2:
            lag_end = st.date_input(
                "Ventana lag - hasta",
                key="lag_end",
                format="DD/MM/YYYY",
                on_change=_sync_windows_from_lag,
            )

        ini_lag = 4
        max_lag = 72

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
                st.session_state.plot_aligned_start = st.session_state.get("lag_start", default_lag_start)
                st.session_state.plot_aligned_end = st.session_state.get("lag_end", default_lag_end)

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

    st.subheader("Paso 3 — Ajuste por regresión lineal")

    if (st.session_state.df_union is None) or (st.session_state.lags_df is None):
        st.info("Ejecutá Paso 1 y Paso 2.")
    else:
        df_union = st.session_state.df_union
        lags_df = st.session_state.lags_df
        obs_col = st.session_state.obs_col

        upstream_all = [c for c in df_union.columns if c != obs_col]

        # 1) Elegir estaciones (como en el Paso 4)
        upstream_sel = st.multiselect(
            "Estaciones aguas arriba para ajuste (regresión lineal)",
            options=upstream_all,
            default=upstream_all[:2],
            max_selections=2,
            key="upstream_sel_unificado",
        )
        if len(upstream_sel) == 0:
            st.stop()

        # 2) Mostrar lag adoptado para cada estación seleccionada
        lag_rows = []
        for est in upstream_sel:
            lag = int(get_lag_for_station(lags_df, est, default=0))
            lag_rows.append({"Estacion": est, "lag_adoptado_h": lag})
        st.markdown("### Lags adoptados")
        st.dataframe(pd.DataFrame(lag_rows), width="stretch")

        # 3) Ventana de ajuste (calibración) — ÚNICA (se usa para todo)
        pstart, pend = default_plot_window_from_index(df_union.index, days=90)

        # Inicializar AJUSTE una sola vez (y permitir que se sincronice desde la ventana lag)
        if "fit_start_unif" not in st.session_state:
            st.session_state.fit_start_unif = pstart
        if "fit_end_unif" not in st.session_state:
            st.session_state.fit_end_unif = pend

        st.markdown("### Ventana de ajuste (calibración)")
        
        st.caption("Seleccione fechas para definir la ventana de calibración. " \
        "Esta ventana se usará para ajustar el modelo de regresión lineal. " \
        "Sugerencia: elegí una ventana reciente, para priorizar los datos más actuales.")

        fs1, fs2 = st.columns(2, gap="large")
        with fs1:
            fit_start = st.date_input(
                "AJUSTE - desde",
                #value=st.session_state.fit_start_unif,
                key="fit_start_unif",
                format="DD/MM/YYYY",
            )
        with fs2:
            fit_end = st.date_input(
                "AJUSTE - hasta",
                #value=st.session_state.fit_end_unif,
                key="fit_end_unif",
                format="DD/MM/YYYY",
            )

        df_fit = df_union.loc[
            datetime.combine(fit_start, datetime.min.time()):
            datetime.combine(fit_end, datetime.max.time())
        ].copy()

        if df_fit.empty:
            st.warning("La ventana de AJUSTE no tiene datos.")
            st.stop()

        # -------------------------------------------------------------------------
        # 4) Diagnóstico (opcional, colapsado) — usa df_fit (la ventana de arriba)
        # -------------------------------------------------------------------------
        with st.expander("Diagnóstico (ajuste + evaluación) — opcional", expanded=False):
            st.caption("El diagnóstico usa la ventana de AJUSTE seleccionada arriba.")

            est_diag = st.selectbox(
                "Estación para diagnóstico",
                options=upstream_sel,      # solo entre las elegidas
                key="est_diag_select",
            )

            lag_diag = int(get_lag_for_station(lags_df, est_diag, default=0))
            st.caption(f"Lag usado (lag_manual): **{lag_diag} h**")

            # Ajuste en df_fit
            try:
                y_obs_fit, y_fit_fit, modelo = ajustar_estacion_con_lag(
                    df_union=df_fit,
                    est=est_diag,
                    obs_col=obs_col,
                    lag=lag_diag,
                )
            except Exception as e:
                st.exception(e)
                st.stop()

            # Métricas
            m1, m2, m3, m4 = st.columns(4, gap="large")
            with m1:
                st.metric("R² (ajuste)", f"{float(getattr(modelo, 'rsquared', float('nan'))):.3f}")
            with m2:
                st.metric("n (ajuste)", f"{int(getattr(modelo, 'nobs', 0))}")
            with m3:
                st.metric("Intercepto", f"{float(modelo.params.get('const', float('nan'))):.3f}")
            with m4:
                st.metric("Pendiente", f"{float(modelo.params.get('up_lag', float('nan'))):.3f}")

            # Gráfico ajuste temporal
            st.markdown("#### Ajuste temporal")
            df_fit_plot = pd.DataFrame(
                {
                    f"{obs_col} (obs)": y_obs_fit,
                    f"Fit (desde {est_diag}, lag {lag_diag}h)": y_fit_fit,
                }
            )
            fig_fit = plot_timeseries_daily_grid(
                df_fit_plot,
                ylabel="Nivel",
                title=f"Ajuste {est_diag} → {obs_col}",
            )
            st.pyplot(fig_fit, width="stretch")

            # Scatter
            st.markdown("#### Scatter — Obs vs Fit")
            fig_sc_fit = plot_scatter_obs_fit(
                y_obs_fit,
                y_fit_fit,
                title="Obs vs Fit",
                figsize=(2.75, 2.75),
                s=10,
                fontsize=9,
                ticksize=8,
            )
            st.pyplot(fig_sc_fit, width="content")

        # 5) Operativo — Última semana + ajuste + pronóstico (SIN pedir fechas)
        #     - usa df_fit (ventana seleccionada arriba) para calibrar
        st.markdown("### Operativo — Última semana")


        # --- Ajuste final manual (shift vertical) ---
        st.markdown("#### Ajuste final (shift vertical)")
        st.caption("Los siguientes valores representan el desplazamiento vertical (en metros) para cada estación."
        " Si el modelo ajusta con un sesgo, podés aplicar un corrimiento vertical para corregirlo. ")
        if "shift_map" not in st.session_state:
            st.session_state.shift_map = {}

        if len(upstream_sel) == 1:
            est0 = upstream_sel[0]
            v0 = float(st.session_state.shift_map.get(est0, 0.0))
            v0 = st.number_input(
                f"Shift [m] para {est0}",
                value=v0,
                step=0.01,
                format="%.3f",
                key=f"shift_m__{est0}",
            )
            st.session_state.shift_map[est0] = float(v0)
        else:
            sh_cols = st.columns(len(upstream_sel), gap="large")
            for i, est_i in enumerate(upstream_sel):
                with sh_cols[i]:
                    vi = float(st.session_state.shift_map.get(est_i, 0.0))
                    vi = st.number_input(
                        f"Shift [m]\n{est_i}",
                        value=vi,
                        step=0.01,
                        format="%.3f",
                        key=f"shift_m__{est_i}",
                    )
                    st.session_state.shift_map[est_i] = float(vi)

        obs = df_union[obs_col].dropna()
        if obs.empty:
            st.warning("No hay observado en estación objetivo.")
            st.stop()

        t_emit = obs.index.max()
        t_start = t_emit - pd.Timedelta(days=7)

        df_week = df_union.loc[t_start:t_emit].copy()
        obs_lastweek = df_week[obs_col].rename(f"{obs_col} (obs)")

        forecasts = []
        meta_rows = []

        for est in upstream_sel:
            lag = int(get_lag_for_station(lags_df, est, default=0))

            # Ajustar modelo usando la ventana df_fit (la seleccionada arriba)
            y_obs_fit, y_fit_fit, modelo = ajustar_estacion_con_lag(
                df_union=df_fit,
                est=est,
                obs_col=obs_col,
                lag=lag,
            )

            # Ajuste reciente (última semana)
            y_hist_week = forecast_from_upstream(
                df=df_week,
                est=est,
                obs_col=obs_col,
                lag=lag,
                modelo=modelo,
                freq="1h",
            )

            # Pronóstico futuro
            y_fcst = forecast_horizon_from_upstream_last(
                df_union=df_union,
                est=est,
                obs_col=obs_col,
                lag=lag,
                modelo=modelo,
                freq="1h",
            )

            # Curva continua: semana previa + futuro
            y_full = pd.concat([y_hist_week.loc[:t_emit], y_fcst.loc[y_fcst.index > t_emit]])
            y_full = y_full[~y_full.index.duplicated(keep="first")]

            # aplicar shift vertical final (si el usuario lo definió)
            shift_m = float(st.session_state.get("shift_map", {}).get(est, 0.0))
            if shift_m != 0.0:
                y_full = y_full + shift_m

            y_full = round_numeric_df(y_full.to_frame(name=f"Modelo ({est}, lag {lag}h, shift {shift_m:+.3f}m)"), decimals=2).iloc[:, 0]
            forecasts.append(y_full)

            meta_rows.append(
                {
                    "Estacion": est,
                    "lag_adoptado_h": lag,
                    "R2_ajuste": float(getattr(modelo, "rsquared", float("nan"))),
                    "n_ajuste": int(getattr(modelo, "nobs", 0)),
                    "const": float(modelo.params.get("const", np.nan)),
                    "beta_up_lag": float(modelo.params.get("up_lag", np.nan)),
                    "shift_m": float(st.session_state.get("shift_map", {}).get(est, 0.0)),
                }
            )

        meta = pd.DataFrame(meta_rows)
        st.markdown("#### Resumen modelos")
        st.dataframe(meta, width="stretch")

        df_final = pd.concat([obs_lastweek] + forecasts, axis=1)
        df_final = round_numeric_df(df_final, decimals=2)

        fig = plot_timeseries_daily_grid(
            df_final,
            ylabel="Nivel",
            title="Observado (última semana) + modelo",
        )
        ax = fig.axes[0]
        ax.axvline(t_emit, linestyle="--", linewidth=1)
        ax.text(
            t_emit,
            ax.get_ylim()[1],
            " Emisión prono",
            va="top",
            ha="left",
            fontsize=8,
        )
        st.pyplot(fig, width="stretch")

        st.markdown("#### Gráfico interactivo")
        fig_interactive = build_interactive_timeseries(
            df_final,
            title="Observado (última semana) + modelo",
            ylabel="Nivel",
        )
        fig_interactive.add_vline(x=t_emit, line_dash="dash", line_width=1)
        st.plotly_chart(fig_interactive, width="stretch")

        st.markdown("#### Descargar resultados")
        df_export = round_numeric_df(df_final.copy(), decimals=2)
        df_export.index.name = "Fecha"
        df_export = df_export.reset_index()

        formato_descarga_final = st.radio(
            "Formato de descarga de resultados",
            options=["csv", "excel"],
            horizontal=True,
            key="formato_descarga_final",
        )

        if formato_descarga_final == "csv":
            st.download_button(
                "Descargar CSV",
                data=df_to_csv_bytes(df_export, index=False),
                file_name="ultima_semana_modelo.csv",
                mime="text/csv",
            )
        else:
            st.download_button(
                "Descargar Excel",
                data=df_to_excel_bytes(df_export.set_index("Fecha"), sheet_name="prono_operativo"),
                file_name="ultima_semana_modelo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

def render_pronostico_subestacional() -> None:
    st.subheader("Pronóstico hidrológico subestacional")
    st.caption(
        "Pronóstico mensual para Misión La Paz basado en persistencia de cuantiles "
        "y analogías históricas. La serie se arma combinando la serie histórica y la serie actual."
    )

    if "sub_resultados" not in st.session_state:
        st.session_state.sub_resultados = None

    with st.expander("Configuración", expanded=True):

        st.markdown("#### Parámetros del pronóstico")

        p1, p2, p3, p4 = st.columns(4, gap="large")

        with p1:
            long_busqueda = st.number_input(
                "Meses de búsqueda",
                value=6,
                min_value=1,
                step=1,
                key="sub_long_busqueda",
                help=(
                    "Cantidad de meses previos usados para comparar la situación actual "
                    "contra años históricos y buscar analogías."
                ),
            )

        with p2:
            long_prono = st.number_input(
                "Meses a pronosticar",
                value=4,
                min_value=1,
                step=1,
                key="sub_long_prono",
                help=(
                    "Horizonte temporal del pronóstico mensual hacia adelante."
                ),
            )

        with p3:
            cantidad_analogos = st.number_input(
                "Cantidad de análogos",
                value=5,
                min_value=1,
                step=1,
                key="sub_cant_analogos",
                help=(
                    "Cantidad de años históricos similares que se usan "
                    "para construir el pronóstico por analogía."
                ),
            )

        with p4:
            orden_analogos = st.selectbox(
                "Ordenar analogías por",
                options=["RMSE", "Score", "CoefC", "Nash", "ErrVol"],
                index=0,
                key="sub_orden_analogos",
                help=(
                    "Métrica utilizada para seleccionar los años análogos más similares."
                ),
            )

        with st.expander("Descripción de métricas", expanded=False):

            st.markdown("""
        **RMSE**: Error cuadrático medio.

        **CoefC**: Coeficiente de correlación lineal.

        **Nash**: Eficiencia de Nash-Sutcliffe.

        **ErrVol**: Error porcentual de volumen acumulado.

        **Score**:Indicador combinado que resume varias métricas de ajuste.
        """)


        usar_origen_auto = st.checkbox(
            "Inferir mes/año de emisión automáticamente",
            value=True,
            key="sub_origen_auto",
        )

        if usar_origen_auto:
            yr_select, mes_select = infer_forecast_origin()
            st.info(f"Origen inferido: {MONTH_NAMES.get(mes_select, mes_select)} {yr_select}")
        else:
            o1, o2 = st.columns(2, gap="large")
            with o1:
                yr_select = st.number_input("Año origen", value=2015, step=1, key="sub_yr_select")
            with o2:
                mes_select = st.number_input(
                    "Mes origen",
                    value=2,
                    min_value=1,
                    max_value=12,
                    step=1,
                    key="sub_mes_select",
                )

    run_sub = st.button(
        "Ejecutar pronóstico subestacional",
        type="primary",
        key="sub_run",
    )

    if run_sub:
        st.session_state.sub_resultados = None

        if not a5_token:
            st.error("Falta A5_TOKEN.")
        else:
            try:
                os.environ["A5_URL"] = A5_URL_FIJO
                os.environ["A5_TOKEN"] = a5_token

                # Configuración operativa fija. No se expone al usuario.
                station = StationConfig(
                    fecha_desde="1980-01-01 01:00:00",
                    fecha_hasta=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                cleaning = CleaningConfig()

                forecast_cfg = ForecastConfig(
                    long_busqueda=int(long_busqueda),
                    long_prono=int(long_prono),
                    mes_select=None,
                    yr_select=None,
                    plot=False,
                )

                analogy_cfg = AnalogyConfig(
                    orden=str(orden_analogos),
                    orden_ascending=(str(orden_analogos) in {"RMSE", "ErrVol"}),
                    cantidad=int(cantidad_analogos),
                )

                forecast_cfg.yr_select, forecast_cfg.mes_select = infer_forecast_origin()

                with st.spinner("Descargando series, armando serie mensual y calculando pronóstico..."):
                    serie_reg, outliers, hyd_stats = prepare_regular_series(station, cleaning)

                    df_resamp = resample_series(
                        serie_reg,
                        group_var=forecast_cfg.vent_resamp,
                        value_col=station.variable,
                        min_count=forecast_cfg.min_count_mensual,
                    )

                    if usar_origen_auto:
                        yr_select, mes_select = get_last_valid_forecast_origin(
                            df_resamp,
                            value_col=station.variable,
                            period_col=forecast_cfg.vent_resamp,
                        )
                    else:
                        yr_select = int(yr_select)
                        mes_select = int(mes_select)

                    forecast_cfg.yr_select = int(yr_select)
                    forecast_cfg.mes_select = int(mes_select)
                    


                    # st.write("Origen usado persistencia:")
                    # st.write(forecast_cfg.yr_select, forecast_cfg.mes_select)

                    # row_sel = df_resamp[
                    #     (df_resamp["year"] == forecast_cfg.yr_select)
                    #     & (df_resamp["month"] == forecast_cfg.mes_select)
                    # ]

                    # st.dataframe(row_sel.tail(1))

                    # same_month = df_resamp[
                    #     df_resamp["month"] == forecast_cfg.mes_select
                    # ][station.variable]

                    # selected_value = row_sel.iloc[0][station.variable]

                    # q = (same_month < selected_value).mean()

                    # st.write("Cuantil base:", q)
                    
                    # st.write("Cuantiles marzo:", df_resamp.loc[df_resamp["month"] == 3, "valor"].quantile([0.25, 0.5, 0.574, 0.75, 0.9]))
                    
                    # vals_mayo = df_resamp.loc[
                    #     df_resamp["month"] == 5,
                    #     "valor"
                    # ].dropna()
                    # st.write("Q25:", vals_mayo.quantile(0.25))
                    # st.write("Q50:", vals_mayo.quantile(0.50))
                    # st.write("Q57:", vals_mayo.quantile(q))
                    # st.write("Q75:", vals_mayo.quantile(0.75))
                    

                    # idx_selected, _ = get_selected_row(
                    #     df_resamp,
                    #     2026,
                    #     3,
                    #     "month",
                    # )




                    prono_persistencia = forecast_persistence(
                        df=df_resamp,
                        value_col=station.variable,
                        selected_month=forecast_cfg.mes_select,
                        selected_year=forecast_cfg.yr_select,
                        search_length=forecast_cfg.long_busqueda,
                        forecast_length=forecast_cfg.long_prono,
                        period_col=forecast_cfg.vent_resamp,
                    )
                    # st.write("Pronóstico de persistencia:", prono_persistencia)
                    # st.write("Pronóstico de persistencia:", prono_persistencia[["fecha", "Prono"]])


                    prono_analogia, analogos, trazas, df_obj = forecast_analogy(
                        df=df_resamp,
                        value_col=station.variable,
                        selected_month=forecast_cfg.mes_select,
                        selected_year=forecast_cfg.yr_select,
                        forecast_cfg=forecast_cfg,
                        analogy_cfg=analogy_cfg,
                    )

                st.session_state.sub_resultados = {
                    "station": station,
                    "forecast_cfg": forecast_cfg,
                    "analogy_cfg": analogy_cfg,
                    "serie_reg": serie_reg,
                    "df_resamp": df_resamp,
                    "outliers": outliers,
                    "hyd_stats": hyd_stats,
                    "prono_persistencia": prono_persistencia,
                    "prono_analogia": prono_analogia,
                    "analogos": analogos,
                    "trazas": trazas,
                    "df_obj": df_obj,
                }
                st.success("Pronóstico subestacional calculado.")
            except Exception as e:
                st.exception(e)

    resultados = st.session_state.sub_resultados
    if resultados is None:
        st.info("Ejecutá el pronóstico para ver los resultados.")
        return

    station = resultados["station"]
    forecast_cfg = resultados["forecast_cfg"]
    df_resamp = resultados["df_resamp"]
    outliers = resultados["outliers"]
    hyd_stats = resultados["hyd_stats"]
    prono_persistencia = resultados["prono_persistencia"]
    prono_analogia = resultados["prono_analogia"]
    analogos = resultados["analogos"]
    trazas = resultados["trazas"]

    st.markdown("### Resumen")

    fecha_ini = pd.to_datetime(
        dict(
            year=[int(df_resamp.iloc[0]["year"])],
            month=[int(df_resamp.iloc[0]["month"])],
            day=[1],
        )
    ).iloc[0]

    fecha_fin = pd.to_datetime(
        dict(
            year=[int(df_resamp.iloc[-1]["year"])],
            month=[int(df_resamp.iloc[-1]["month"])],
            day=[1],
        )
    ).iloc[0]

    m1, m2, m3 = st.columns(3, gap="large")

    with m1:
        st.metric(
            "Origen",
            f"{MONTH_NAMES.get(forecast_cfg.mes_select, forecast_cfg.mes_select)} {forecast_cfg.yr_select}"
        )

    with m2:
        st.metric(
            "Serie histórica",
            f"{fecha_ini.year}–{fecha_fin.year}"
        )

    with m3:
        st.metric(
            "Meses sin datos",
            int(df_resamp[station.variable].isna().sum())
        )

    # ------------------------------------------------------------------
    # Línea 2 — parámetros del modelo
    # ------------------------------------------------------------------

    m4, m5, m6 = st.columns(3, gap="large")

    with m4:
        st.metric(
            "Ventana búsqueda",
            f"{forecast_cfg.long_busqueda} meses"
        )

    with m5:
        st.metric(
            "Horizonte prono",
            f"{forecast_cfg.long_prono} meses"
        )

    with m6:
        st.metric(
            "Cantidad de Análogos",
            f"{analogy_cfg.cantidad}"
        )

    st.caption(
        f"La serie mensual utilizada por el modelo abarca desde "
        f"{fecha_ini.strftime('%m/%Y')} hasta {fecha_fin.strftime('%m/%Y')}."
    )

    st.caption(
        f"La búsqueda de analogías utiliza los "
        f"{forecast_cfg.long_busqueda} meses previos al origen del pronóstico."
    )


    st.markdown("### Pronóstico")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Persistencia")
        st.dataframe(
            round_numeric_df(prono_persistencia[["fecha", "horizonte", "Prono", "cuantil_base"]], decimals=2),
            width="stretch",
        )
    with c2:
        st.markdown("#### Analogías")
        st.dataframe(
            round_numeric_df(prono_analogia[["fecha", "horizonte", "Prono"]], decimals=2),
            width="stretch",
        )
    
    st.markdown("### Gráficos")

    idx_selected, _ = get_selected_row(
        df_resamp,
        forecast_cfg.yr_select,
        forecast_cfg.mes_select,
        forecast_cfg.vent_resamp,
    )

    hist_plot = df_resamp.iloc[
        max(0, idx_selected - 12): idx_selected + 1
    ].copy()

    hist_plot["fecha"] = pd.to_datetime(
        dict(
            year=hist_plot["year"],
            month=hist_plot["month"],
            day=15,
        )
    )

    st.markdown("#### Pronóstico mensual interactivo")

    fig_interactive_sub = build_interactive_subseasonal_forecast(
        df_hist=hist_plot,
        persistence=prono_persistencia,
        analogy=prono_analogia,
        value_col=station.variable,
    )

    st.plotly_chart(fig_interactive_sub, width="stretch")


    recent_obs = df_resamp.iloc[
        max(0, idx_selected - forecast_cfg.long_busqueda + 1): idx_selected + 1
    ].copy()

    with st.expander("Boxplot histórico + persistencia", expanded=False):
        fig_box_pers = plot_forecast_boxplot(
            station_name=f"{station.nombre} - Persistencia",
            recent_obs=recent_obs,
            forecast=prono_persistencia,
            historical=df_resamp,
            period_col=forecast_cfg.vent_resamp,
            value_col=station.variable,
            search_length=forecast_cfg.long_busqueda,
        )
        if fig_box_pers is not None:
            st.pyplot(fig_box_pers, width="stretch")

    # with st.expander("Boxplot histórico + analogía", expanded=False):
    #     fig_box_ana = plot_forecast_boxplot(
    #         station_name=f"{station.nombre} - Analogía",
    #         recent_obs=recent_obs,
    #         forecast=prono_analogia,
    #         historical=df_resamp,
    #         period_col=forecast_cfg.vent_resamp,
    #         value_col=station.variable,
    #         search_length=forecast_cfg.long_busqueda,
    #     )
    #     if fig_box_ana is not None:
    #         st.pyplot(fig_box_ana, width="stretch")

    with st.expander("Trazas análogas seleccionadas", expanded=False):
        fig1 = plot_analogy_traces(
            traces=trazas,
            forecast=prono_analogia,
            selected_analogs=analogos,
            station_name=station.nombre,
            value_col=station.variable,
            selected_year=forecast_cfg.yr_select,
            selected_month=forecast_cfg.mes_select,
        )
        if fig1 is not None:
            st.pyplot(fig1, width="stretch")

    with st.expander("Comparación mensual Matplotlib", expanded=False):
        fig2 = plot_forecasts_comparison(
            df=df_resamp,
            persistence=prono_persistencia,
            analogy=prono_analogia,
            station_name=station.nombre,
            value_col=station.variable,
            selected_year=forecast_cfg.yr_select,
            selected_month=forecast_cfg.mes_select,
        )
        if fig2 is not None:
            st.pyplot(fig2, width="stretch")




    st.markdown("### Años análogos seleccionados")
    cols_analogos = [c for c in ["YrSim", "RMSE", "CoefC", "Nash", "ErrVol", "wi"] if c in analogos.columns]
    st.dataframe(round_numeric_df(analogos[cols_analogos], decimals=3), width="stretch")

    with st.expander("Serie mensual usada por el modelo", expanded=False):
        st.dataframe(round_numeric_df(df_resamp.tail(24), decimals=2), width="stretch")

    # with st.expander("Corrección por mínimo hidráulico", expanded=False):
    #     cols_hyd = [c for c in ["hyd_year", "n_valid", "p02", "p02_suavizado", "offset", "hyd_year_ref", "p02_ref"] if c in hyd_stats.columns]
    #     st.dataframe(round_numeric_df(hyd_stats[cols_hyd].tail(15), decimals=3), width="stretch")

    st.markdown("### Descargas")
    df_export = pd.concat(
        [
            prono_persistencia.assign(metodo_export="persistencia"),
            prono_analogia.assign(metodo_export="analogia"),
        ],
        ignore_index=True,
        sort=False,
    )

    d1, d2, d3 = st.columns(3, gap="large")
    with d1:
        st.download_button(
            "Descargar pronósticos CSV",
            data=df_to_csv_bytes(round_numeric_df(df_export, decimals=3), index=False),
            file_name="pronostico_subestacional.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "Descargar serie mensual CSV",
            data=df_to_csv_bytes(round_numeric_df(df_resamp, decimals=3), index=False),
            file_name="serie_mensual_subestacional.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Descargar análogos CSV",
            data=df_to_csv_bytes(round_numeric_df(analogos, decimals=4), index=False),
            file_name="analogos_subestacional.csv",
            mime="text/csv",
        )

def render_plan_pydrodelta() -> None:
    st.subheader("Plan pydrodelta")
    st.caption(
        "Genera los archivos YAML de plan para correr las rutinas de pydrodelta "
        "(https://github.com/jbianchi81/pydrodelta) con lo que se fue configurando en los "
        "otros dos tabs. Se generan dos planes separados: operativo (paso horario) y "
        "subestacional (paso mensual)."
    )

    caso = st.text_input(
        "Caso (nombre de la carpeta de salidas)",
        value=st.session_state.get("plan_caso", "pilcomayo-mlp"),
        key="plan_caso",
        help="Las salidas del plan se escriben en ./data/<caso>/, relativas al directorio "
             "desde donde se corra pydrodelta. El YAML se guarda en resultados/planes/.",
    )
    st.caption(f"Salidas del plan: `{plan_builder.DATA_DIR_TPL.format(caso=caso)}/`")

    # Plan operativo
    st.markdown("### Plan operativo (paso 1 hora)")

    if st.session_state.df_union is None or st.session_state.lags_df is None:
        st.info("Ejecutá los pasos 1 y 2 del pronóstico operativo para poder generar el plan.")
    else:
        obs_col = st.session_state.obs_col
        estaciones = {
            r["Estacion"]: int(r["serie_id"])
            for r in st.session_state.stations
            if r.get("Estacion") and pd.notna(r.get("serie_id"))
        }

        upstream_default = [
            e for e in st.session_state.get("upstream_sel_unificado", [])
            if e in estaciones and e != obs_col
        ]
        if not upstream_default:
            upstream_default = [e for e in estaciones if e != obs_col][:2]

        upstream_plan = st.multiselect(
            "Estaciones aguas arriba a incluir (un LinearFit por estación)",
            options=[e for e in estaciones if e != obs_col],
            default=upstream_default,
            key="plan_upstream_sel",
        )

        # Ventanas: se toman de lo elegido en los pasos anteriores, editables
        d_from = st.session_state.get("download_from")
        d_to = st.session_state.get("download_to")
        dias_atras_def = (d_to - d_from).days if d_from and d_to else 90

        f_from = st.session_state.get("fit_start_unif")
        f_to = st.session_state.get("fit_end_unif")
        tail_steps_def = int((f_to - f_from).days * 24) if f_from and f_to else 0

        lags_df = st.session_state.lags_df
        lags_sel = [int(get_lag_for_station(lags_df, e, default=0)) for e in upstream_plan]
        horas_adelante_def = max(lags_sel) if lags_sel else 24

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            plan_id_op = st.number_input(
                "id del plan (placeholder)",
                value=int(st.session_state.get("plan_id_op", plan_builder.PLAN_ID_OPERATIVO)),
                step=1,
                key="plan_id_op",
                help="Placeholder hasta que haya un calibrado asignado en la API destino.",
            )
        with c2:
            dias_atras = st.number_input(
                "Días hacia atrás (timestart)",
                value=int(dias_atras_def),
                min_value=1,
                step=1,
                key="plan_dias_atras",
            )
        with c3:
            horas_adelante = st.number_input(
                "Horas hacia adelante (timeend)",
                value=int(horas_adelante_def),
                min_value=1,
                step=1,
                key="plan_horas_adelante",
            )

        c4, c5 = st.columns([2, 1], gap="large")
        with c4:
            nombre_op = st.text_input(
                "Nombre del plan",
                value=f"Pilcomayo - pronóstico operativo {obs_col}",
                key="plan_nombre_op",
            )
        with c5:
            tail_steps = st.number_input(
                "tail_steps (pasos de calibración)",
                value=int(tail_steps_def),
                min_value=0,
                step=1,
                key="plan_tail_steps",
                help="Ventana de ajuste del Paso 3 expresada en pasos de 1 hora. "
                     "0 = usar toda la serie. pydrodelta recalibra en cada corrida.",
            )

        st.caption(
            "El lag adoptado va como `x_offset` de cada serie aguas arriba y el `y_offset` "
            "queda en 0: el corrimiento vertical lo absorbe el intercepto que recalibra pydrodelta."
        )

        if st.button("Generar plan operativo", type="primary", key="plan_gen_op"):
            try:
                refs = plan_builder.refs_desde_resumen(
                    st.session_state.df_resumen,
                    estaciones,
                    params_por_estacion={e: get_params_limpieza(e) for e in estaciones},
                )
                obs_ref = refs[obs_col]
                upstream_refs = []
                for est in upstream_plan:
                    ref = refs[est]
                    ref.x_offset_horas = int(get_lag_for_station(lags_df, est, default=0))
                    ref.comment = f"lag por correlación cruzada: {ref.x_offset_horas} h"
                    upstream_refs.append(ref)

                if not upstream_refs:
                    st.error("Elegí al menos una estación aguas arriba.")
                else:
                    plan = plan_builder.build_plan_operativo(
                        obs_ref=obs_ref,
                        upstream_refs=upstream_refs,
                        caso=caso,
                        plan_id=int(plan_id_op),
                        nombre=nombre_op,
                        dias_atras=int(dias_atras),
                        horas_adelante=int(horas_adelante),
                        tail_steps=int(tail_steps) or None,
                    )
                    avisos = plan_builder.advertencias_plan(plan, [obs_ref] + upstream_refs)
                    texto = plan_builder.plan_to_yaml(
                        plan,
                        comentarios=[
                            f"generado por app_Prono_MLP el {datetime.now():%Y-%m-%d %H:%M}",
                            "id de plan provisorio (placeholder): reemplazar por el calibrado real",
                        ],
                    )
                    ruta = plan_builder.guardar_plan(texto, f"plan_operativo_{caso}.yml")
                    st.session_state.plan_yaml_op = texto
                    st.session_state.plan_yaml_op_nombre = ruta.name
                    st.session_state.plan_avisos_op = avisos
                    st.success(f"Plan operativo generado y guardado en `{ruta}`.")
            except Exception as e:
                st.exception(e)

        if st.session_state.get("plan_yaml_op"):
            for aviso in st.session_state.get("plan_avisos_op", []):
                st.warning(aviso)
            st.download_button(
                "Descargar plan operativo (YAML)",
                data=st.session_state.plan_yaml_op.encode("utf-8"),
                file_name=st.session_state.plan_yaml_op_nombre,
                mime="text/yaml",
                key="plan_dl_op",
            )
            with st.expander("Ver YAML del plan operativo", expanded=False):
                st.code(st.session_state.plan_yaml_op, language="yaml")

    # Plan subestacional
    st.markdown("### Plan subestacional (paso mensual)")

    station_def = StationConfig()
    cleaning_def = CleaningConfig()

    s1, s2, s3 = st.columns(3, gap="large")
    with s1:
        plan_id_sub = st.number_input(
            "id del plan (placeholder)",
            value=int(st.session_state.get("plan_id_sub", plan_builder.PLAN_ID_SUBESTACIONAL)),
            step=1,
            key="plan_id_sub",
        )
    with s2:
        serie_actual = st.number_input(
            "serie_id actual",
            value=int(station_def.id_serie),
            step=1,
            key="plan_serie_sub",
        )
    with s3:
        serie_hist = st.number_input(
            "serie_id histórica",
            value=int(station_def.id_serie_hist),
            step=1,
            key="plan_serie_hist_sub",
        )

    nombre_sub = st.text_input(
        "Nombre del plan",
        value=f"Pilcomayo - pronóstico subestacional {station_def.nombre}",
        key="plan_nombre_sub",
    )

    st.caption(
        "Parámetros tomados del tab subestacional: "
        f"búsqueda {st.session_state.get('sub_long_busqueda', 6)} meses, "
        f"pronóstico {st.session_state.get('sub_long_prono', 4)} meses, "
        f"{st.session_state.get('sub_cant_analogos', 5)} análogos, "
        f"orden por {st.session_state.get('sub_orden_analogos', 'RMSE')}."
    )

    if st.button("Generar plan subestacional", type="primary", key="plan_gen_sub"):
        if not a5_token:
            st.error("Falta A5_TOKEN: se necesita para leer estacion.id y var.id desde A5.")
        else:
            try:
                client = Crud(A5_URL_FIJO, token=a5_token)
                with st.spinner("Leyendo metadata de las series desde A5..."):
                    meta_act = plan_builder.leer_metadata_serie(client, int(serie_actual))
                    meta_hist = plan_builder.leer_metadata_serie(client, int(serie_hist))

                ref_actual = plan_builder.SerieRef(
                    estacion=meta_act.get("estacion_nombre") or station_def.nombre,
                    serie_id=int(serie_actual),
                    estacion_id=meta_act.get("estacion_id"),
                    var_id=meta_act.get("var_id"),
                    tipo=meta_act.get("tipo", "puntual"),
                    lim_outliers=tuple(cleaning_def.limite_outliers),
                )
                ref_hist = plan_builder.SerieRef(
                    estacion=meta_hist.get("estacion_nombre") or station_def.nombre,
                    serie_id=int(serie_hist),
                    estacion_id=meta_hist.get("estacion_id"),
                    var_id=meta_hist.get("var_id"),
                    tipo=meta_hist.get("tipo", "puntual"),
                    lim_outliers=tuple(cleaning_def.limite_outliers),
                )

                orden = str(st.session_state.get("sub_orden_analogos", "RMSE"))
                plan = plan_builder.build_plan_subestacional(
                    ref_actual=ref_actual,
                    ref_historica=ref_hist,
                    caso=caso,
                    plan_id=int(plan_id_sub),
                    nombre=nombre_sub,
                    timestart=pd.Timestamp(station_def.fecha_desde).strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z"
                    ),
                    long_busqueda=int(st.session_state.get("sub_long_busqueda", 6)),
                    long_prono=int(st.session_state.get("sub_long_prono", 4)),
                    cantidad_analogos=int(st.session_state.get("sub_cant_analogos", 5)),
                    orden_analogos=orden,
                    orden_ascending=orden in {"RMSE", "ErrVol"},
                )
                avisos = plan_builder.advertencias_plan(plan, [ref_actual, ref_hist])
                texto = plan_builder.plan_to_yaml(
                    plan,
                    comentarios=[
                        f"generado por app_Prono_MLP el {datetime.now():%Y-%m-%d %H:%M}",
                        "id de plan provisorio (placeholder): reemplazar por el calibrado real",
                    ],
                )
                ruta = plan_builder.guardar_plan(texto, f"plan_subestacional_{caso}.yml")
                st.session_state.plan_yaml_sub = texto
                st.session_state.plan_yaml_sub_nombre = ruta.name
                st.session_state.plan_avisos_sub = avisos
                st.success(f"Plan subestacional generado y guardado en `{ruta}`.")
            except Exception as e:
                st.exception(e)

    if st.session_state.get("plan_yaml_sub"):
        for aviso in st.session_state.get("plan_avisos_sub", []):
            st.warning(aviso)
        st.download_button(
            "Descargar plan subestacional (YAML)",
            data=st.session_state.plan_yaml_sub.encode("utf-8"),
            file_name=st.session_state.plan_yaml_sub_nombre,
            mime="text/yaml",
            key="plan_dl_sub",
        )
        with st.expander("Ver YAML del plan subestacional", expanded=False):
            st.code(st.session_state.plan_yaml_sub, language="yaml")


tab_operativo, tab_subestacional, tab_plan = st.tabs([
    "Pronóstico operativo",
    "Pronóstico subestacional",
    "Plan pydrodelta",
])

with tab_operativo:
    render_pronostico_operativo()

with tab_subestacional:
    render_pronostico_subestacional()

with tab_plan:
    render_plan_pydrodelta()

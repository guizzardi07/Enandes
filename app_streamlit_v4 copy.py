from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Imports del proyecto
def _import_project_functions():
    """
    Importa funciones/clases del paquete `modulos/`.
    Requiere ejecutar Streamlit desde la raíz del proyecto.
    """
    try:
        from modulos.series import construir_series_union
        from modulos.hindcast import evaluar_estaciones_individuales, ajustar_estacion_con_lag
        from a5client import Crud
        return construir_series_union, evaluar_estaciones_individuales, ajustar_estacion_con_lag, Crud
    except Exception as e:
        raise ImportError(
            "No se pudo importar desde `modulos/`. Verificá que exista `modulos/series.py`, "
            "`modulos/hindcast.py` y `a5client.py`, y que estés corriendo Streamlit "
            "desde la raíz del proyecto."
        ) from e

construir_series_union, evaluar_estaciones_individuales, ajustar_estacion_con_lag, Crud = _import_project_functions()

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

def _default_plot_window_from_index(idx: pd.DatetimeIndex, days: int = 90) -> Tuple[date, date]:
    if idx is None or len(idx) == 0:
        today = _today_local()
        return today - timedelta(days=days), today
    idx_min = idx.min()
    idx_max = idx.max()
    if pd.isna(idx_min) or pd.isna(idx_max):
        today = _today_local()
        return today - timedelta(days=days), today
    start = max(idx_min.date(), (idx_max - pd.Timedelta(days=days)).date())
    return start, idx_max.date()

def _plot_timeseries_daily_grid(df: pd.DataFrame, ylabel: str = "Nivel", title: str | None = None) -> plt.Figure:
    """
    Plot de series temporales con:
    - ticks mayores mensuales (labels)
    - grilla menor diaria en X (líneas verticales)
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col))

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Fecha")
    if title:
        ax.set_title(title)
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
    return fig

def get_lag_for_station(
    df_lags: pd.DataFrame,
    station_name: str,
    default: int = 0
) -> int:
    """
    Devuelve lag_manual para una estación, independientemente de si el DF usa
    'estacion', 'Estacion' o el índice. Hace matching robusto (exacto y suave).
    """
    if df_lags is None or df_lags.empty:
        return int(default)

    # detectar columna de nombre
    name_col = None
    if "Estacion" in df_lags.columns:
        name_col = "Estacion"
    elif "estacion" in df_lags.columns:
        name_col = "estacion"

    # lag_manual debe existir
    if "lag_manual" not in df_lags.columns:
        return int(default)

    # --- caso con columna ---
    if name_col is not None:
        # match exacto
        m = df_lags[name_col].astype(str) == str(station_name)
        if m.any():
            return int(df_lags.loc[m, "lag_manual"].iloc[0])

        # match suave
        key = str(station_name).strip().lower()
        s_map = {str(v).strip().lower(): i for i, v in df_lags[name_col].items()}
        if key in s_map:
            return int(df_lags.loc[s_map[key], "lag_manual"])

    # --- fallback: índice ---
    if station_name in df_lags.index:
        return int(df_lags.loc[station_name, "lag_manual"])

    idx_map = {str(i).strip().lower(): i for i in df_lags.index}
    key = str(station_name).strip().lower()
    if key in idx_map:
        return int(df_lags.loc[idx_map[key], "lag_manual"])

    return int(default)

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
    """
    Scatter compacto Obs vs Fit, pensado para dashboard (fig chica).
    """
    # Alinear índices y limpiar NaNs
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

def forecast_from_upstream(
    df: pd.DataFrame,
    est: str,
    obs_col: str,
    lag: int,
    modelo,
    freq: str = "1h",
) -> pd.Series:
    """
    Genera pronóstico de obs_col a partir de upstream con lag:
      y_hat(t) = const + beta * up(t - lag*step)

    Devuelve una Serie indexada por t (tiempo objetivo en obs_col).
    """
    step = pd.to_timedelta(freq)

    # Queremos y_hat en tiempos t donde exista upstream en (t - lag*step)
    up = df[est].copy()
    idx_t = df.index

    t_up = idx_t - lag * step
    up_at = up.reindex(t_up).to_numpy()

    # params
    const = float(modelo.params.get("const", 0.0))
    beta = float(modelo.params.get("up_lag", np.nan))

    y_hat = const + beta * up_at
    y_hat = pd.Series(y_hat, index=idx_t, name=f"y_hat_{est}_lag{lag}")

    return y_hat

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

    #st.caption("El paso temporal es fijo (1H). La carpeta de figuras se configura en el Paso 1.")

# Session state init
if "df_union" not in st.session_state:
    st.session_state.df_union = None
if "df_resumen" not in st.session_state:
    st.session_state.df_resumen = None
if "lags_df" not in st.session_state:
    st.session_state.lags_df = None

# downstream fijo
if "obs_col" not in st.session_state:
    st.session_state.obs_col = "Misión La Paz"
# estaciones editables
if "stations" not in st.session_state:
    st.session_state.stations = [
        {"Estacion": "Misión La Paz", "serie_id": 42293},
        {"Estacion": "Villa Montes", "serie_id": 42291},
        {"Estacion": "Puente Aruma", "serie_id": 42294},
    ]
# step fijo, visible
if "step_adopt" not in st.session_state:
    st.session_state.step_adopt = "h"  # 1H fijo
# carpeta figuras (editable, pero la ubicamos en Paso 1)
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

# nombres para descargas (se muestran encima de botones de descarga)
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
        width="stretch",
        column_config={
            "Estacion": st.column_config.TextColumn(required=True),
            "serie_id": st.column_config.NumberColumn(required=True, step=1),
        },
        key="stations_editor",
    )
    st.session_state.stations = stations_df.to_dict("records")

with colB:
    st.info("Estación objetivo : **Misión La Paz** (fija).")

# Paso 1: Descargar + limpiar (y guardar a resultados/)
st.subheader("Paso 1 — Descargar y limpiar series")

# Paso temporal + carpeta de figuras (arriba del botón, como pediste)
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

today = _today_local()
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
            step_td = _parse_step(step_adopt)

            estaciones_dict: Dict[str, int] = {
                r["Estacion"]: int(r["serie_id"])
                for r in st.session_state.stations
                if r.get("Estacion") and pd.notna(r.get("serie_id"))
            }

            client = Crud(A5_URL_FIJO, token=a5_token)

            timestart = _to_dt_start(d_start)
            timeend = _to_dt_end(d_end)

            # carpetas de salida
            out_dir = Path("resultados")
            out_dir.mkdir(parents=True, exist_ok=True)
            Path(st.session_state.carpeta_figuras).mkdir(parents=True, exist_ok=True)

            # nombres de salida para guardado (fijos y consistentes)
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

            # Guardado adicional (por si construir_series_union no lo hace o si querés el xlsx directo)
            try:
                df_union.to_csv(out_csv_path, index=True)
                df_union.to_excel(out_xlsx_path, sheet_name="series_limpias")
            except Exception:
                pass

            # inicializar ventanas por defecto de plot
            pstart, pend = _default_plot_window_from_index(df_union.index, days=90)
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

    # Nombre archivo (CSV) y Nombre resumen (Excel) justo arriba de botones de descarga
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

    dcol1, dcol2, dcol3 = st.columns([1, 1, 2], gap="large")
    with dcol1:
        st.download_button(
            "Descargar CSV (series limpias)",
            data=df_union.to_csv().encode("utf-8"),
            file_name=Path(st.session_state.archivo_descarga_csv).name,
            mime="text/csv",
        )
    with dcol2:
        excel_bytes = _df_to_excel_bytes(df_union, sheet_name="series_limpias")
        st.download_button(
            "Descargar Excel (series limpias)",
            data=excel_bytes,
            file_name=Path(st.session_state.archivo_descarga_xlsx).name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dcol3:
        st.caption("Se guardan automáticamente en `resultados/` cuando presionás el botón de descarga+limpieza.")

    st.markdown("### Series descargadas (sin lag)")
    if st.session_state.plot_download_start is None or st.session_state.plot_download_end is None:
        pstart, pend = _default_plot_window_from_index(df_union.index, days=90)
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
        fig = _plot_timeseries_daily_grid(df_plot, ylabel="Nivel")
        st.pyplot(fig, width="stretch")

# Paso 2: Estimar lag óptimo (ventana) + gráfico con lags aplicados a TODAS las upstream
st.subheader("Paso 2 — Lag óptimo por estación + gráfico con lag aplicado")

if st.session_state.df_union is None:
    st.info("Primero ejecutá el Paso 1 para tener df_union.")
else:
    df_union = st.session_state.df_union
    step_td = _parse_step(st.session_state.step_adopt)

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
            pstart, pend = _default_plot_window_from_index(df_union.index, days=90)
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
                aligned = _apply_lag_align(df_plot[col], step=step_td, lag_steps=lag)
                aligned_dict[f"{col} (aligned +{lag})"] = aligned

            aligned_df = pd.DataFrame(aligned_dict)
            fig = _plot_timeseries_daily_grid(
                aligned_df,
                ylabel="Nivel",
                title="Series alineadas por lag_manual"
            )
            st.pyplot(fig, width="stretch")

            st.caption("Todas las upstream se desplazan según su `lag_manual` (en pasos de 1H).")

# Paso 3 — Ajuste por estación (lag fijo) + evaluar en otra ventana + pronóstico (up+lag+modelo)
st.subheader("Paso 3 — Ajuste y pronóstico (modelo + lag)")

if (st.session_state.df_union is None) or (st.session_state.lags_df is None):
    st.info("Ejecutá Paso 1 y Paso 2 para habilitar el ajuste (df_union + lags_df).")
else:
    df_union = st.session_state.df_union
    df_lags_edit = st.session_state.lags_df
    obs_col = st.session_state.obs_col

    # Upstreams disponibles
    upstream_cols_all = [c for c in df_union.columns if c != obs_col]
    est_sel = st.selectbox("Elegí estación upstream", upstream_cols_all, key="fit_station_select")

    # Lag (robusto): tomado de lags_df (Paso 2), sin depender de mayúsculas/espacios
    lag_sel = get_lag_for_station(st.session_state.lags_df, est_sel, default=0)
    st.caption(f"Lag usado (lag_manual): **{lag_sel} h**")

    # Ventanas por defecto
    pstart, pend = _default_plot_window_from_index(df_union.index, days=90)

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

    # Recortes
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

    # Ajuste (calibra en df_fit)
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

    # Métricas del ajuste (sobre ventana de ajuste)
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

    st.markdown("### Ajuste temporal (ventana de AJUSTE)")
    df_fit_plot = pd.DataFrame(
        {
            f"{obs_col} (obs)": y_obs_fit,
            f"Fit (desde {est_sel}, lag {lag_sel}h)": y_fit_fit,
        }
    )
    fig_fit = _plot_timeseries_daily_grid(
        df_fit_plot,
        ylabel="Nivel",
        title=f"Ajuste {est_sel} → {obs_col} (ventana de ajuste)",
    )
    st.pyplot(fig_fit, width="stretch")

    st.markdown("### Scatter (AJUSTE) — Obs vs Fit")
    fig_sc_fit = plot_scatter_obs_fit(
        y_obs_fit,
        y_fit_fit,
        title="Obs vs Fit (ajuste)",
        figsize=(2.75, 2.75),
        s=10,
        fontsize=9,
        ticksize=8,
    )
    st.pyplot(fig_sc_fit, width="content")

    # ----------------------------------------------------------------------------------
    # Pronóstico / estimación usando upstream + lag + modelo calibrado, aplicado a df_plot
    # ----------------------------------------------------------------------------------
    st.markdown("### Pronóstico aplicado (ventana de GRÁFICO / evaluación)")

    if df_plot.empty:
        st.warning("La ventana de GRÁFICO no tiene datos.")
        st.stop()

    # Serie pronosticada en el eje temporal de df_plot
    y_hat_plot = forecast_from_upstream(
        df=df_plot,
        est=est_sel,
        obs_col=obs_col,
        lag=lag_sel,
        modelo=modelo,
        freq="1h",  # paso fijo
    )

    y_obs_plot = df_plot[obs_col]

    df_pred_plot = pd.DataFrame(
        {
            f"{obs_col} (obs)": y_obs_plot,
            f"Pronóstico (desde {est_sel}, lag {lag_sel}h)": y_hat_plot,
        }
    )

    # Gráfico temporal (evaluación)
    fig_pred = _plot_timeseries_daily_grid(
        df_pred_plot,
        ylabel="Nivel",
        title="Observado vs Pronóstico (ventana de gráfico)",
    )
    st.pyplot(fig_pred, width="stretch")

    # Scatter evaluación (ventana de gráfico)
    st.markdown("### Scatter (EVALUACIÓN) — Obs vs Pronóstico")
    fig_sc_eval = plot_scatter_obs_fit(
        y_obs_plot,
        y_hat_plot,
        title="Obs vs Pronóstico (evaluación)",
        figsize=(2.75, 2.75),
        s=10,
        fontsize=9,
        ticksize=8,
    )
    st.pyplot(fig_sc_eval, width="content")

    # Descarga del pronóstico (ventana de gráfico)
    df_out = pd.DataFrame(
        {
            "Fecha": y_hat_plot.index,
            "obs": y_obs_plot.values,
            "y_hat": y_hat_plot.values,
        }
    )
    st.download_button(
        "Descargar pronóstico (CSV)",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"pronostico_{est_sel}_lag{lag_sel}_plotwindow.csv",
        mime="text/csv",
    )

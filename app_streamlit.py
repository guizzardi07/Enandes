
"""
Streamlit dashboard - primera versión simple

Objetivos:
- Descargar y construir series unificadas (df_union)
- Evaluar estaciones individuales (R² + lag óptimo)
- Ejecutar hindcast diario
- Visualizar resultados (spaghetti, obs + pronos, métricas por lead)
- Descargar tablas (CSV/Excel)

Este app intenta funcionar en dos layouts típicos de tu proyecto:
1) Paquete "modulos/" (como en run_hindcast.py)
2) Archivos sueltos en el mismo directorio (hindcast.py, series.py, resultados.py, etc.)

Uso:
  streamlit run app_streamlit.py
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Imports robustos (paquete vs archivos sueltos)
def _import_project_functions():
    # 1) "modulos" como paquete
    try:
        from modulos import construir_series_union, evaluar_estaciones_individuales, hindcast_diario  # type: ignore
        from modulos.resultados import (  # type: ignore
            cargar_serie_observada,
            cargar_hindcast,
            merge_prono_obs,
            calcular_metricas_por_lead,
            plot_spaghetti,
            plot_obs_y_resumen_pronos,
            plot_metricas_por_lead,
        )
        return (construir_series_union, evaluar_estaciones_individuales, hindcast_diario,
                cargar_serie_observada, cargar_hindcast, merge_prono_obs, calcular_metricas_por_lead,
                plot_spaghetti, plot_obs_y_resumen_pronos, plot_metricas_por_lead)
    except Exception:
        pass

    # 2) archivos sueltos
    from series import construir_series_union  # type: ignore
    from hindcast import evaluar_estaciones_individuales, hindcast_diario  # type: ignore
    from resultados import (  # type: ignore
        cargar_serie_observada,
        cargar_hindcast,
        merge_prono_obs,
        calcular_metricas_por_lead,
        plot_spaghetti,
        plot_obs_y_resumen_pronos,
        plot_metricas_por_lead,
    )

    return (construir_series_union, evaluar_estaciones_individuales, hindcast_diario,
            cargar_serie_observada, cargar_hindcast, merge_prono_obs, calcular_metricas_por_lead,
            plot_spaghetti, plot_obs_y_resumen_pronos, plot_metricas_por_lead)

(
    construir_series_union,
    evaluar_estaciones_individuales,
    hindcast_diario,
    cargar_serie_observada,
    cargar_hindcast,
    merge_prono_obs,
    calcular_metricas_por_lead,
    plot_spaghetti,
    plot_obs_y_resumen_pronos,
    plot_metricas_por_lead,
) = _import_project_functions()

# A5 client
try:
    from dotenv import load_dotenv
    from a5client import Crud  # type: ignore
except Exception:
    Crud = None  # type: ignore
    load_dotenv = None  # type: ignore

# Utils
def _default_station_map() -> Dict[str, int]:
    # valores de tu run_hindcast.py
    return {
        "Misión La Paz": 42293,
        "Villa Montes": 42291,
        "Puente Aruma": 42294,
    }

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _df_to_download_bytes(df: pd.DataFrame, kind: str) -> Tuple[bytes, str, str]:
    """
    kind: 'csv' or 'xlsx'
    returns: (bytes, mime, ext)
    """
    if kind == "csv":
        b = df.to_csv(index=True).encode("utf-8")
        return b, "text/csv", "csv"
    if kind == "xlsx":
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=True, sheet_name="data")
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
    raise ValueError("kind debe ser 'csv' o 'xlsx'")

@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[0], index_col=0)

@st.cache_data(show_spinner=False)
def _read_hindcast_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Fecha", "Fecha_emitido"])

# UI
st.set_page_config(page_title="Pilcomayo | Tablero de control", layout="wide")

st.title("Tablero de control (Streamlit) – versión 1")
st.caption("Ejecutar el flujo (series → evaluación → hindcast) + ver y descargar resultados.")

with st.sidebar:
    st.header("Proyecto / paths")

    carpeta_resultados = st.text_input("Carpeta resultados", value="resultados")
    carpeta_figuras = st.text_input("Carpeta figuras (series)", value=str(Path(carpeta_resultados) / "figuras_series_h"))

    archivo_union = st.text_input("CSV df_union", value=str(Path(carpeta_resultados) / "series_nivel_union_h.csv"))
    archivo_resumen = st.text_input("XLSX resumen series", value=str(Path(carpeta_resultados) / "resumen_series_niveles_h.xlsx"))
    archivo_hindcast = st.text_input("CSV hindcast", value=str(Path(carpeta_resultados) / "hindcast_diario.csv"))

    st.divider()
    st.header("A5 (descarga)")
    use_a5 = st.checkbox("Usar API A5 para descargar series", value=True)

    if use_a5:
        if load_dotenv is not None:
            load_dotenv()

        a5_url = st.text_input("A5_URL (env o manual)", value=os.getenv("A5_URL", ""))
        a5_token = st.text_input("A5_TOKEN (env o manual)", value=os.getenv("A5_TOKEN", ""), type="password")
    else:
        a5_url, a5_token = "", ""

    st.divider()
    st.header("Estaciones")
    st.write("Editá la tabla si querés agregar/quitar estaciones.")

    st.session_state.setdefault("stations_df", pd.DataFrame(
        [{"estacion": k, "serie_id": v} for k, v in _default_station_map().items()]
    ))
    stations_df = st.data_editor(
        st.session_state["stations_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="stations_editor",
    )

    st.divider()
    st.header("Parámetros series")
    timestart = st.text_input("timestart", value="2005-11-01")
    timeend = st.text_input("timeend", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    step_adopt = st.selectbox("step_adopt", options=["h", "3h", "6h", "D"], index=0)
    plot_series = st.checkbox("Mostrar plots al descargar", value=False)

    st.divider()
    st.header("Parámetros evaluación/hindcast")
    obs_col = st.text_input("Columna observada (downstream)", value="Misión La Paz")
    estaciones_up_default = ["Villa Montes", "Puente Aruma"]
    estaciones_up = st.multiselect("Estaciones upstream a evaluar", options=stations_df["estacion"].tolist(),
                                   default=[e for e in estaciones_up_default if e in stations_df["estacion"].tolist()])
    max_lag = st.number_input("max_lag (horas)", min_value=1, max_value=240, value=72, step=1)
    ini_lag = st.number_input("ini_lag (horas)", min_value=0, max_value=240, value=2, step=1)

    st.write("---")
    fecha_ini_eval = st.date_input("Hindcast: fecha inicio", value=pd.to_datetime("2022-01-03").date())
    fecha_fin_eval = st.date_input("Hindcast: fecha fin", value=pd.to_datetime("2024-12-31").date())
    hora_emision = st.number_input("Hora emisión diaria", min_value=0, max_value=23, value=9, step=1)
    freq = st.selectbox("freq (paso base)", options=["1h", "3h", "6h"], index=0)

    ventana_lag_years = st.number_input("Ventana lag (años)", min_value=0, max_value=10, value=1, step=1)
    ventana_lag_months = st.number_input("Ventana lag (meses)", min_value=0, max_value=24, value=3, step=1)
    ventana_ML_years = st.number_input("Ventana ML (años)", min_value=0, max_value=10, value=1, step=1)
    ventana_ML_months = st.number_input("Ventana ML (meses)", min_value=0, max_value=24, value=1, step=1)

    horizonte_horas = st.number_input("horizonte_horas (si el código lo fuerza al lag, se ignora)", min_value=1, max_value=240, value=42, step=1)
    min_muestras = st.number_input("min_muestras", min_value=10, max_value=50000, value=200, step=10)

# Convertir estaciones
Estaciones: Dict[str, int] = {
    str(r["estacion"]): int(r["serie_id"]) for _, r in stations_df.dropna().iterrows()
    if str(r["estacion"]).strip() and pd.notna(r["serie_id"])
}

# paths
res_dir = _ensure_dir(carpeta_resultados)
fig_dir = _ensure_dir(carpeta_figuras)

tab1, tab2, tab3 = st.tabs(["1) Series", "2) Hindcast", "3) Resultados"])

# -----------------------------
# TAB 1 - Series
with tab1:
    st.subheader("1) Series unificadas (df_union)")
    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        st.write("**Acciones**")
        recompute_union = st.checkbox("Recalcular df_union (si existe, lo pisa)", value=False)
        if st.button("Construir / cargar df_union", type="primary"):
            if (not use_a5) and (not Path(archivo_union).exists()):
                st.error("No estás usando A5 y no existe el archivo_union. Activá A5 o cargá un CSV existente.")
            else:
                with st.spinner("Procesando series…"):
                    if recompute_union and Path(archivo_union).exists():
                        Path(archivo_union).unlink()

                    if not Path(archivo_union).exists():
                        if Crud is None:
                            st.error("No se pudo importar a5client. Instalalo en tu env o desactivá A5.")
                        else:
                            client = Crud(url=a5_url, token=a5_token)
                            df_union, df_resumen = construir_series_union(
                                Estaciones=Estaciones,
                                timestart=timestart,
                                timeend=pd.to_datetime(timeend),
                                step_adopt=step_adopt,
                                client=client,
                                carpeta_figuras=fig_dir,
                                archivo_salida=archivo_union,
                                archivo_salida_resumen=archivo_resumen,
                            )
                            st.success(f"df_union generado: {archivo_union}")
                            st.success(f"resumen generado: {archivo_resumen}")
                            st.session_state["df_union"] = df_union
                            st.session_state["df_resumen"] = df_resumen
                    else:
                        df_union = _read_csv_cached(archivo_union)
                        st.session_state["df_union"] = df_union
                        st.success(f"df_union cargado desde: {archivo_union}")

    with colB:
        st.write("**Carga manual (opcional)**")
        up_union = st.file_uploader("Subir un CSV de df_union", type=["csv"], key="upload_union")
        if up_union is not None:
            df_union = pd.read_csv(up_union, parse_dates=[0], index_col=0)
            st.session_state["df_union"] = df_union
            st.success("df_union cargado desde archivo subido.")

    if "df_union" in st.session_state:
        df_union = st.session_state["df_union"]
        st.write("**Vista previa df_union**")
        st.dataframe(df_union.tail(50), use_container_width=True)

        b, mime, ext = _df_to_download_bytes(df_union, "csv")
        st.download_button("Descargar df_union (CSV)", data=b, file_name=f"df_union.{ext}", mime=mime)

        # Evaluación estaciones
        st.divider()
        st.subheader("Evaluar estaciones individuales (R² + lag óptimo)")

        if st.button("Calcular ranking", type="secondary"):
            with st.spinner("Evaluando estaciones…"):
                orden, df_eval = evaluar_estaciones_individuales(
                    df_union,
                    estaciones=tuple(estaciones_up) if estaciones_up else tuple(),
                    obs_col=obs_col,
                    max_lag=int(max_lag),
                    ini_lag=int(ini_lag),
                )
                st.session_state["df_eval"] = df_eval
                st.session_state["orden"] = orden

        if "df_eval" in st.session_state:
            st.write("**Ranking**")
            st.dataframe(st.session_state["df_eval"], use_container_width=True)
            st.write("Orden:", st.session_state["orden"])

# -----------------------------
# TAB 2 - Hindcast
# -----------------------------
with tab2:
    st.subheader("2) Hindcast diario")

    if "df_union" not in st.session_state:
        st.info("Primero cargá o construí df_union en el Tab 1.")
    else:
        df_union = st.session_state["df_union"]

        if "orden" in st.session_state and len(st.session_state["orden"]) >= 1:
            primary_default = st.session_state["orden"][0]
            secondary_default = st.session_state["orden"][1] if len(st.session_state["orden"]) > 1 else None
        else:
            primary_default = estaciones_up[0] if estaciones_up else "Villa Montes"
            secondary_default = estaciones_up[1] if len(estaciones_up) > 1 else None

        col1, col2, col3 = st.columns([1, 1, 1], gap="large")
        with col1:
            primary = st.selectbox("Primary", options=df_union.columns.tolist(), index=df_union.columns.tolist().index(primary_default) if primary_default in df_union.columns else 0)
        with col2:
            secondary = st.selectbox("Secondary (fallback)", options=[None] + df_union.columns.tolist(), index=0 if secondary_default is None else ([None] + df_union.columns.tolist()).index(secondary_default))
        with col3:
            st.write("**Salida**")
            overwrite_hind = st.checkbox("Sobrescribir hindcast existente", value=False)

        if st.button("Ejecutar hindcast", type="primary"):
            with st.spinner("Corriendo hindcast diario…"):
                if overwrite_hind and Path(archivo_hindcast).exists():
                    Path(archivo_hindcast).unlink()

                # date -> str
                ini = pd.to_datetime(fecha_ini_eval).strftime("%Y-%m-%d")
                fin = pd.to_datetime(fecha_fin_eval).strftime("%Y-%m-%d")

                df_hind = hindcast_diario(
                    df_union=df_union,
                    primary=primary,
                    secondary=secondary if secondary is not None else primary,
                    obs_col=obs_col,
                    fecha_ini_eval=ini,
                    fecha_fin_eval=fin,
                    freq=freq,
                    ventana_lag=pd.DateOffset(years=int(ventana_lag_years), months=int(ventana_lag_months)),
                    ventana_ML=pd.DateOffset(years=int(ventana_ML_years), months=int(ventana_ML_months)),
                    horizonte_horas=int(horizonte_horas),
                    min_muestras=int(min_muestras),
                )

                st.session_state["df_hind"] = df_hind
                _ensure_dir(Path(archivo_hindcast).parent)
                df_hind.to_csv(archivo_hindcast, index=False)
                st.success(f"Hindcast guardado en: {archivo_hindcast}")

        st.write("**Cargar hindcast existente**")
        colL, colU = st.columns([1, 1])
        with colL:
            if st.button("Cargar hindcast desde path"):
                if Path(archivo_hindcast).exists():
                    st.session_state["df_hind"] = _read_hindcast_cached(archivo_hindcast)
                    st.success("Hindcast cargado.")
                else:
                    st.error("No existe el archivo indicado.")
        with colU:
            up_h = st.file_uploader("Subir hindcast CSV", type=["csv"], key="upload_hindcast")
            if up_h is not None:
                df_hind = pd.read_csv(up_h, parse_dates=["Fecha", "Fecha_emitido"])
                st.session_state["df_hind"] = df_hind
                st.success("Hindcast cargado desde archivo subido.")

        if "df_hind" in st.session_state:
            df_hind = st.session_state["df_hind"]
            st.write("**Vista previa hindcast**")
            st.dataframe(df_hind.head(50), use_container_width=True)

            st.download_button(
                "Descargar hindcast (CSV)",
                data=df_hind.to_csv(index=False).encode("utf-8"),
                file_name="hindcast.csv",
                mime="text/csv",
            )

# -----------------------------
# TAB 3 - Resultados / plots
# -----------------------------
with tab3:
    st.subheader("3) Resultados y gráficos")

    if "df_hind" not in st.session_state:
        st.info("Necesitás un hindcast (Tab 2) para ver resultados.")
    else:
        df_hind = st.session_state["df_hind"]

        colP1, colP2 = st.columns([1, 1], gap="large")

        with colP1:
            st.write("### Spaghetti")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_spaghetti(df_hind, ax=ax, alpha=0.15)
            st.pyplot(fig, clear_figure=True)

        with colP2:
            st.write("### Observado + pronósticos (media)")
            # Observado: desde df_union si está; si no, desde archivo_union
            if "df_union" in st.session_state:
                obs = st.session_state["df_union"][obs_col].rename("obs")
            else:
                obs = cargar_serie_observada(archivo_union, nombre_columna=obs_col)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            plot_obs_y_resumen_pronos(df_hind, obs, ax=ax2)
            st.pyplot(fig2, clear_figure=True)

        st.write("---")
        st.write("### Métricas por lead")
        if "df_union" in st.session_state:
            obs = st.session_state["df_union"][obs_col].rename("obs")
        else:
            obs = cargar_serie_observada(archivo_union, nombre_columna=obs_col)

        df_merged = merge_prono_obs(df_hind, obs)
        df_metrics = calcular_metricas_por_lead(df_merged)

        st.dataframe(df_metrics, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(9, 4))
        plot_metricas_por_lead(df_metrics, ax=ax3, incluir_nse=True)
        st.pyplot(fig3, clear_figure=True)

        st.download_button(
            "Descargar métricas (CSV)",
            data=df_metrics.to_csv(index=True).encode("utf-8"),
            file_name="metricas_por_lead.csv",
            mime="text/csv",
        )

st.caption("Siguiente paso típico: encapsular 'run_hindcast.py' en funciones puras y separar IO/plots para acelerar el dashboard.")

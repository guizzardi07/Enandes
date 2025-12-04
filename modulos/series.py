from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from a5client import Crud, observacionesListToDataFrame

from .limpieza_series import (
    eliminaVentana,
    corrimiento_vertical,
    removeOutliers,
    DetectaSaltos_v1,
    inferir_frecuencia,
    graficar_serie_niveles,
    _slugify,
    get_params_limpieza,
)

logger = logging.getLogger(__name__)

def leer_serie_nivel_estacion(
    estacion: str,
    serie_id: int,
    timestart: str,
    timeend: datetime,
    client: Crud,
    fig_dir: str | Path | None = None,
    plot_serie: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Descarga la serie de nivel de una estación desde la API A5 y deja
    una serie con una sola columna 'valor'.

    - Descarga la serie.
    - Identifica la columna de nivel.
    - Normaliza nombres de columnas.
    - Devuelve df con índice datetime y columna 'valor'.
    - Devuelve también un dict con metadatos básicos.
    """
    logger.info("Descargando serie para estación %s (id=%s)", estacion, serie_id)

    resp = client.readSerie(serie_id, timestart, timeend)
    df = observacionesListToDataFrame(resp["observaciones"])

    # Guardamos fecha explícitamente y normalizamos nombres
    col_fecha = "fecha"
    df[col_fecha] = df.index
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    col_fecha = "fecha"

    # --- Detección de columna de nivel ---
    candidatas_h = [c for c in df.columns if c != col_fecha]
    prioridad = ["valor", "h (m)", "h(m)", "h", "nivel", "altura"]
    col_h: str | None = None

    for p in prioridad:
        for c in candidatas_h:
            if p in c:
                col_h = c
                break
        if col_h is not None:
            break

    if col_h is None:
        num_cols = df[candidatas_h].select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col_h = num_cols[0]

    if col_h is None:
        raise ValueError(
            f"No se encontró columna de nivel en {estacion}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Nos quedamos solo con fecha + nivel
    df = df[[col_fecha, col_h]].dropna(subset=[col_h])
    if df.empty:
        logger.warning("La serie de %s quedó vacía luego de dropna", estacion)

    # Index = fecha, columna única "valor"
    df = df.set_index(col_fecha)
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={col_h: "valor"})

    # Plot rápido
    if plot_serie and not df.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["valor"], linewidth=2, label=estacion)
        plt.xlabel("Fecha")
        plt.ylabel("Nivel [m]")
        plt.grid(linestyle="-.", linewidth=0.5)
        plt.tight_layout()
        plt.legend()
        plt.show()

    meta = {
        "Estacion": estacion,
        "Fecha Inicio": df.index.min().date() if not df.empty else None,
        "Fecha Fin": df.index.max().date() if not df.empty else None,
        "Cantidad de registros": len(df),
    }

    # Figura de serie bruta (opcional, versión PNG)
    if fig_dir is not None and not df.empty:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        ruta_figura = fig_dir / f"Serie_Altura_{_slugify(estacion)}.png"
        graficar_serie_niveles(df, "valor", estacion, ruta_figura)

    return df, meta

def analizar_frecuencia_y_faltantes(
    df: pd.DataFrame,
    estacion: str,
    step_adopt: str,
    fig_dir: str | Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Infere la frecuencia principal de la serie y calcula el porcentaje
    de timestamps faltantes respecto a un índice regular con frecuencia
    `step_adopt`.

    - Usa `inferir_frecuencia` para estimar el paso dominante.
    - Calcula % de timestamps faltantes en un índice completo.
    - Genera figura de distribución de pasos temporales (opcional).

    Devuelve
    --------
    stats : dict
        Diccionario con:
          - Frecuencia (horas)
          - pct_faltantes
          - n_timestamps_esperados
          - n_timestamps_faltantes
    df_resample_base : DataFrame
        Serie re-muestreada a `step_adopt` con mean (sin rellenar).
    """
    if df.empty:
        return {
            "Frecuencia": None,
            "pct faltantes": None,
            "n_timestamps_esperados": 0,
            "n_timestamps_faltantes": 0,
        }, df

    fecha_inicio_ts = df.index.min().floor("h")
    fecha_fin_ts = df.index.max().floor("h")

    # --- Inferir frecuencia principal ---
    step, top_hours = inferir_frecuencia(df.index)
    if step is not None:
        paso_mas_comun_horas = step.total_seconds() / 3600.0
    else:
        paso_mas_comun_horas = None

    # Figura de pasos temporales
    if top_hours is not None and fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        top_hours.sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_xlabel("Paso temporal (horas)")
        ax.set_ylabel("Frecuencia relativa")
        ax.set_xticklabels(
            [f"{h:.2f}" for h in top_hours.sort_values(ascending=False).index],
            rotation=0,
        )
        plt.tight_layout()
        ruta_figura = fig_dir / f"Frec_PasoTemp_{_slugify(estacion)}.png"
        plt.savefig(ruta_figura, dpi=150)
        plt.close()

    # --- Porcentaje de timestamps faltantes vs step_adopt ---
    if fecha_inicio_ts is not None and fecha_fin_ts is not None:
        idx_completo = pd.date_range(
            start=fecha_inicio_ts, end=fecha_fin_ts, freq=step_adopt
        )
        n_timestamps_esperados = len(idx_completo)

        df_resamp_base = df.resample(step_adopt).mean().dropna()
        faltantes = idx_completo.difference(df_resamp_base.index)
        n_timestamps_faltantes = len(faltantes)

        pct_timestamps_faltantes = (
            100 * n_timestamps_faltantes / n_timestamps_esperados
            if n_timestamps_esperados > 0
            else None
        )
    else:
        n_timestamps_esperados = 0
        n_timestamps_faltantes = 0
        pct_timestamps_faltantes = None
        df_resamp_base = df.copy()

    stats = {
        "Frecuencia": paso_mas_comun_horas,
        "pct faltantes": pct_timestamps_faltantes,
        "n_timestamps_esperados": n_timestamps_esperados,
        "n_timestamps_faltantes": n_timestamps_faltantes,
    }

    return stats, df_resamp_base

def limpiar_y_rellenar_serie(
    df: pd.DataFrame,
    estacion: str,
    step_adopt: str,
    fig_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica limpieza y relleno sobre una serie de nivel:

      - Aplica limpieza (ventanas, corrimientos, outliers, saltos) según
        parámetros específicos de la estación (get_params_limpieza).
      - Re-muestrea a `step_adopt` y completa con NaN los timestamps faltantes.
      - Interpola huecos menores a 24 pasos (24 horas si step_adopt = 'h').

    Parámetros
    ----------
    df : DataFrame
        Serie original con índice datetime y columna 'valor'.
    estacion : str
        Nombre de la estación (clave para parámetros de limpieza).
    step_adopt : str
        Frecuencia objetivo, por ejemplo "h".
    fig_dir : str | Path | None
        Carpeta para guardar figuras de la serie limpia.

    Devuelve
    --------
    df_interp : DataFrame
        Serie limpia, re-muestreada y con huecos cortos interpolados.
    stats : dict
        Diccionario con H min y H max.
    """
    if df.empty:
        return df.copy(), {"H min": None, "H max": None}

    # --- Parámetros específicos de limpieza ---
    params = get_params_limpieza(estacion)

    # Ventanas a eliminar
    if params.get("ventanas"):
        df = eliminaVentana(df, params["ventanas"], nom_col="valor")

    # Corrimientos verticales
    if params.get("corrimientos"):
        df = corrimiento_vertical(df, params["corrimientos"], nom_col="valor")

    # Outliers manuales
    if params.get("outliers"):
        _, df = removeOutliers(df, params["outliers"], column="valor")

    # Saltos
    if params.get("saltos"):
        sp = params["saltos"]
        df = DetectaSaltos_v1(
            df, "valor", sp["ventana"], sp["umbral"], plot=False
        )
        if "valor_filt" in df.columns:
            df["valor"] = df["valor_filt"]

    # --- Resample + índice completo + interpolación ---
    df_resamp = df.resample(step_adopt).mean().asfreq(step_adopt)
    df_resamp.index = pd.to_datetime(df_resamp.index)

    # Asegurar índice sin tz (por si la API trae tz-aware)
    if isinstance(df_resamp.index, pd.DatetimeIndex) and df.index.tz is not None:
        df_resamp.index = df_resamp.index.tz_localize(None)

    # Interpolar huecos de hasta 24 pasos
    df_interp = df_resamp.interpolate(
        method="time",
        limit=24,
        limit_direction="both",
    )

    # Estadísticas básicas
    serie_h = df_interp["valor"]
    h_min = serie_h.min(skipna=True)
    h_max = serie_h.max(skipna=True)

    # Figura de serie limpia
    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        ruta_figura = fig_dir / f"Serie_Altura_{_slugify(estacion)}_clean.png"
        graficar_serie_niveles(df_interp, "valor", estacion, ruta_figura)

    stats = {
        "H min": h_min,
        "H max": h_max,
    }
    return df_interp, stats

def analiza_series_nivel(
    Estaciones: dict[str, int],
    timestart: str,
    timeend: datetime,
    step_adopt: str,
    client: Crud,
    fig_dir: str | Path | None = None,
    plot_serie: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Para todas las estaciones:
      1) Descargar y preparar serie de nivel.
      2) Analizar frecuencia principal y % de timestamps faltantes.
      3) Limpiar, re-muestrear e interpolar huecos cortos.

    Devuelve:
    df_resumen : DataFrame
        Resumen por estación (una fila por estación).
    series_clean : dict[str, DataFrame]
        Diccionario {estacion: df_clean}, con df_clean índice datetime
        y columna 'valor'.
    """
    filas_resumen: list[dict] = []
    series_clean: dict[str, pd.DataFrame] = {}

    for estacion, serie_id in Estaciones.items():
        logger.info("=== Estación: %s ===", estacion)

        # Descargar y preparar
        df_raw, meta = leer_serie_nivel_estacion(
            estacion=estacion,
            serie_id=serie_id,
            timestart=timestart,
            timeend=timeend,
            client=client,
            fig_dir=fig_dir,
            plot_serie=plot_serie,
        )

        if df_raw.empty:
            meta.update(
                {
                    "Frecuencia": None,
                    "pct faltantes": None,
                    "H min": None,
                    "H max": None,
                }
            )
            filas_resumen.append(meta)
            continue

        # Analizar frecuencia y % faltantes
        stats_freq, _ = analizar_frecuencia_y_faltantes(
            df=df_raw,
            estacion=estacion,
            step_adopt=step_adopt,
            fig_dir=fig_dir,
        )

        # 3Limpiar y rellenar
        df_clean, stats_clean = limpiar_y_rellenar_serie(
            df=df_raw,
            estacion=estacion,
            step_adopt=step_adopt,
            fig_dir=fig_dir,
        )

        series_clean[estacion] = df_clean

        # Armar fila resumen combinando meta + stats
        resumen = {
            **meta,
            **{
                "Frecuencia": stats_freq["Frecuencia"],
                "pct faltantes": stats_freq["pct faltantes"],
                "H min": stats_clean["H min"],
                "H max": stats_clean["H max"],
            },
        }
        filas_resumen.append(resumen)

    df_resumen = pd.DataFrame(filas_resumen)
    return df_resumen, series_clean

def construir_series_union(
    Estaciones: Dict[str, int],
    timestart: str,
    timeend: datetime,
    step_adopt: str,
    client: Crud,
    carpeta_figuras: str | Path,
    archivo_salida: str = "series_nivel_union_h.csv",
    archivo_salida_resumen: str = "resumen_series_niveles_h.xlsx",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el flujo de análisis/limpieza y arma un dataframe unificado
    con una columna por estación.

    Devuelve:
    df_union : DataFrame
        Series de nivel limpias, índice datetime, columnas por estación.
    df_resumen : DataFrame
        Resumen estadístico por estación.
    """
    df_resumen, series_dic = analiza_series_nivel(
        Estaciones,
        timestart,
        timeend,
        step_adopt,
        client,
        carpeta_figuras,
        plot_serie=True,
    )

    df_resumen.to_excel(archivo_salida_resumen, index=False)

    renombradas: Dict[str, pd.DataFrame] = {}
    for est, df in series_dic.items():
        df = df[~df.index.duplicated(keep="first")]
        renombradas[est] = df.rename(columns={"valor": est})

    df_union = pd.concat(renombradas.values(), axis=1)
    df_union = df_union[~df_union.index.duplicated(keep="first")]

    df_union.to_csv(archivo_salida)
    logger.info("Series unificadas guardadas en: %s", archivo_salida)

    return df_union, df_resumen

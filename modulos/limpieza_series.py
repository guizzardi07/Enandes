"""
Módulo de utilidades para limpieza y análisis básico de series de nivel.

Incluye:
- Funciones de limpieza (ventanas, corrimientos, outliers, saltos)
- Funciones auxiliares (inferir frecuencia, graficar)
- Diccionario PARAMS_LIMPIEZA con parámetros por estación
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funciones de limpieza
def eliminaVentana(df: pd.DataFrame,
                   dic_ElimVent: Dict,
                   nom_col: str = "valor") -> pd.DataFrame:
    """
    Reemplaza por NaN los valores de `nom_col` dentro de ventanas de tiempo
    definidas en `dic_ElimVent`.
    """
    df0 = df.copy()
    for vent_i, cfg in dic_ElimVent.items():
        inicio_ventana = dt.datetime.strptime(cfg["desde"], "%d/%m/%y %H:%M:%S")
        fin_ventana = dt.datetime.strptime(cfg["hasta"], "%d/%m/%y %H:%M:%S")
        mask = (df0["fecha"] >= inicio_ventana) & (df0["fecha"] <= fin_ventana)
        df0.loc[mask, nom_col] = np.nan
    return df0

def corrimiento_vertical(df: pd.DataFrame,
                         dic_corrim: Dict,
                         nom_col: str = "valor",
                         plot: bool = False) -> pd.DataFrame:
    """
    Aplica corrimientos verticales (suma de un delta) a `nom_col`
    en ventanas de tiempo definidas en `dic_corrim`.
    """
    df_corr = df.copy()
    for corrim_i, cfg in dic_corrim.items():
        fecha_inicio = dt.datetime.strptime(cfg["desde"], "%d/%m/%y %H:%M:%S")
        fecha_fin = dt.datetime.strptime(cfg["hasta"], "%d/%m/%y %H:%M:%S")
        delta = cfg["delta"]
        mask = (df_corr.fecha >= fecha_inicio) & (df_corr.fecha <= fecha_fin)
        df_corr.loc[mask, nom_col] = df_corr.loc[mask, nom_col] + delta
    return df_corr

def removeOutliers(df: pd.DataFrame,
                   limite_outliers: Tuple[float, float],
                   column: str = "valor",
                   plot: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Marca como NaN los valores fuera del rango [minv, maxv] y devuelve:
      - outliers_df: filas que fueron consideradas outliers
      - df_filtrado: dataframe con outliers reemplazados por NaN
    """
    df_filt = df.copy()
    minv, maxv = limite_outliers
    mask_out = (df[column] > maxv) | (df[column] < minv)
    df_filt[column] = np.where(mask_out, np.nan, df[column])
    outliers_df = df[mask_out]
    return outliers_df, df_filt

def DetectaSaltos_v1(df: pd.DataFrame,
                     nomCol: str,
                     ventana_largo: int,
                     umbral_corte: float = 3,
                     plot: bool = False) -> pd.DataFrame:
    """
    Detección simple de saltos mediante comparación de la serie con
    una media móvil adelante/atrás.
    """
    cols_originales = df.columns.tolist()

    df_f = df.copy()
    half = (ventana_largo - 1) // 2

    df_f["media_anterior"] = (
        df_f[nomCol].shift(1).rolling(window=half, min_periods=1).mean()
    )
    df_f["media_posterior"] = (
        df_f[nomCol].shift(-half - 1).rolling(window=half, min_periods=1).mean()
    )
    df_f["media_movil"] = (
        df_f["media_anterior"] * half + df_f["media_posterior"] * half
    ) / (2 * half)
    df_f["residuo"] = (df_f[nomCol] - df_f["media_movil"]).abs()
    df_f["es_outlier"] = df_f["residuo"].abs() > umbral_corte

    col_filt = f"{nomCol}_filt"
    df_f[col_filt] = df_f[nomCol]
    df_f.loc[df_f["es_outlier"], col_filt] = np.nan

    return df_f[cols_originales]


# Funciones auxiliares de análisis / gráficos
def inferir_frecuencia(index: pd.DatetimeIndex):
    """
    Intenta inferir la frecuencia de una serie temporal.

    Devuelve:
      - step: timedelta del paso más común (o None)
      - top_hours: serie con los pasos (en horas) que explican al menos el 10%
                   de los datos (o los 5 más frecuentes si no se cumple eso).
    """
    if len(index) < 2:
        return None, None

    index = index.sort_values()

    diffs = index.to_series().diff().dropna()
    if len(diffs) == 0:
        return None, None

    # Paso temporal principal
    step = diffs.mode().iloc[0]

    # Frecuencias relativas de cada paso
    resume_diffs = diffs.value_counts(normalize=True)

    # Filtrar pasos que tengan al menos el 10% de los datos
    top = resume_diffs[resume_diffs >= 0.10]

    if len(top) == 0:
        top = resume_diffs.head(5)
    elif len(top) > 5:
        top = top.head(5)

    top_hours = top.copy()
    top_hours.index = top_hours.index.total_seconds() / 3600.0  # a horas

    return step, top_hours

def _slugify(nombre: str) -> str:
    """Devuelve un nombre limpio para usar en archivos."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in nombre.strip())

def graficar_serie_niveles(df: pd.DataFrame,
                           col_h: str,
                           estacion: str,
                           ruta_figura: Path | str) -> str:
    """
    Genera y guarda una figura de la serie de niveles (h vs tiempo).
    """
    ruta_figura = Path(ruta_figura)

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df[col_h], linewidth=0.8)
    plt.title(f"Serie de niveles - {estacion}")
    plt.xlabel("Fecha")
    plt.ylabel("Nivel h (m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta_figura.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_figura, dpi=150)
    plt.close()

    return str(ruta_figura)

# Parámetros de limpieza por estación
PARAMS_LIMPIEZA: Dict[str, Dict] = {
    # Ejemplo completo
    "Ejemplo": {
        "ventanas": {
            # "elim1": {"desde": "10/03/21 00:00:00", "hasta": "12/03/21 00:00:00"}
        },
        "corrimientos": {
            # "ajuste1": {
            #     "desde": "01/01/20 00:00:00",
            #     "hasta": "15/01/20 00:00:00",
            #     "delta": -0.05,
            # },
        },
        "outliers": (0.10, 10.0),  # (min, max)
        "saltos": {"ventana": 7, "umbral": 3},
    },
    "Misión La Paz": {
        "outliers": (2.205, 8.0),
        "saltos": {"ventana": 7, "umbral": 2},
    },
    "Villa Montes": {
        "outliers": (0.25, 8.0),
        "saltos": {"ventana": 7, "umbral": 2},
    },
    "Puente Aruma": {
        "outliers": (2.4, 9.0),
        "saltos": {"ventana": 7, "umbral": 2},
    },
    "Palca Grande": {
        "outliers": (0, 9.0),
    },
    "San Josecito": {
        "outliers": (0, 9.0),
    },
    "Viña Quemada": {
        "outliers": (0, 9.0),
    },
    "Talula": {
        "outliers": (0, 9.0),
    },
    "Tarapaya": {
        "outliers": (0, 9.0),
    },
    
}

def get_params_limpieza(estacion: str) -> Dict:
    """
    Devuelve el diccionario de parámetros de limpieza para una estación.

    Si no hay parámetros definidos, devuelve {} y loguea un mensaje informativo.
    """
    if estacion in PARAMS_LIMPIEZA:
        return PARAMS_LIMPIEZA[estacion]
    print(f"[INFO] No hay parámetros de limpieza definidos para '{estacion}'.")
    return {}

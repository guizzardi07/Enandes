from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Funciones para:
- Cargar series observadas y pronósticos (hindcast).
- Calcular métricas de desempeño por lead de pronóstico.
- Generar gráficos tipo:
    * "spaghetti" de todos los pronósticos
    * Observado + media/P10/P90 de pronósticos
    * RMSE / MAE por lead

Formato esperado del hindcast:
    - Fecha         (datetime): instante pronosticado
    - Valor         (float): valor pronosticado
    - Fecha_emitido (datetime): instante de emisión del pronóstico

Formato esperado de la serie observada:
    - DataFrame o Serie con índice datetime y una columna (o nombre) de nivel.
"""

# 1) Funciones de carga
def cargar_serie_observada(
    archivo_union: str | Path,
    nombre_columna: str = "Misión La Paz",
) -> pd.Series:
    """
    Carga una serie observada desde el CSV de df_union.

    Parámetros
    ----------
    archivo_union : str | Path
        Ruta al archivo 'series_nivel_union_h.csv'.
    nombre_columna : str
        Nombre de la columna que contiene la serie observada.

    Devuelve
    --------
    obs : Series
        Serie observada con índice datetime y nombre 'obs'.
    """
    archivo_union = Path(archivo_union)
    df_union = pd.read_csv(archivo_union, parse_dates=[0], index_col=0)
    if nombre_columna not in df_union.columns:
        raise ValueError(
            f"La columna '{nombre_columna}' no está en {archivo_union}. "
            f"Columnas disponibles: {list(df_union.columns)}"
        )
    obs = df_union[nombre_columna].rename("obs")
    return obs

def cargar_hindcast(
    archivo_hindcast: str | Path,
) -> pd.DataFrame:
    """
    Carga el hindcast desde un CSV.

    El archivo debe contener las columnas:
      - 'Fecha'
      - 'Valor'
      - 'Fecha_emitido'

    Devuelve
    --------
    df_h : DataFrame
        Hindcast con columnas ['Fecha', 'Valor', 'Fecha_emitido'].
    """
    archivo_hindcast = Path(archivo_hindcast)
    df_h = pd.read_csv(
        archivo_hindcast,
        parse_dates=["Fecha", "Fecha_emitido"],
    )
    cols_req = {"Fecha", "Valor", "Fecha_emitido"}
    if not cols_req.issubset(df_h.columns):
        raise ValueError(
            f"El archivo {archivo_hindcast} debe contener las columnas "
            f"{cols_req}, y solo tiene {df_h.columns.tolist()}"
        )
    return df_h

# 2) Utilidades: agregar lead y merge con observado
def agregar_lead_horas(df_h: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega una columna 'lead_h' al hindcast en horas:
        lead_h = (Fecha - Fecha_emitido) en horas.

    Devuelve una copia del DataFrame original.
    """
    df = df_h.copy()
    # Diferencia en segundos / 3600 -> horas
    delta = (df["Fecha"] - df["Fecha_emitido"]).dt.total_seconds() / 3600.0
    df["lead_h"] = delta.astype(int)
    return df

def merge_prono_obs(
    df_h: pd.DataFrame,
    serie_obs: pd.Series,
) -> pd.DataFrame:
    """
    Une pronósticos y observados en un solo DataFrame.

    Parámetros
    ----------
    df_h : DataFrame
        Hindcast con columnas ['Fecha', 'Valor', 'Fecha_emitido', ...].
    serie_obs : Series
        Serie observada con índice datetime y nombre, p.ej. 'obs'.

    Devuelve
    --------
    df_merged : DataFrame
        Tiene como mínimo las columnas:
          - Fecha
          - Fecha_emitido
          - Valor       (simulado)
          - obs         (observado)
          - lead_h      (horas)
    """
    # Asegurar lead_h
    if "lead_h" not in df_h.columns:
        df_h = agregar_lead_horas(df_h)

    df_merged = df_h.merge(
        serie_obs,
        left_on="Fecha",
        right_index=True,
        how="inner",
    )
    return df_merged

# 3) Cálculo de métricas
def calcular_metricas_por_lead(
    df_merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula métricas por lead de pronóstico (en horas).

    Requiere columnas:
      - 'Valor' (simulado)
      - 'obs'   (observado)
      - 'lead_h'

    Devuelve
    --------
    df_metrics : DataFrame
        Índice: lead_h (horas)
        Columnas:
          - N          : número de casos
          - RMSE
          - MAE
          - BIAS       : sesgo medio (sim - obs)
          - NSE        : Nash-Sutcliffe
    """
    if "Valor" not in df_merged or "obs" not in df_merged or "lead_h" not in df_merged:
        raise ValueError("df_merged debe contener columnas: Valor, obs, lead_h")

    df = df_merged.copy().dropna(subset=["Valor", "obs", "lead_h"])

    if df.empty:
        raise ValueError("df_merged está vacío luego de dropna en Valor/obs/lead_h")

    df["error"] = df["Valor"] - df["obs"]
    df["sq_error"] = df["error"] ** 2
    df["abs_error"] = df["error"].abs()

    # Varianza global de la observación (para NSE)
    var_obs = df["obs"].var()
    if var_obs == 0 or np.isnan(var_obs):
        var_obs = np.nan  # evitar división por cero

    # Agrupar por lead
    grp = df.groupby("lead_h")

    # Métricas
    rmse = grp["sq_error"].mean().apply(lambda x: np.sqrt(x))
    mae = grp["abs_error"].mean()
    bias = grp["error"].mean()

    # NSE = 1 - MSE / var(obs)
    mse = grp["sq_error"].mean()
    nse = 1 - mse / var_obs
    
    # Armar dataframe final
    df_metrics = pd.DataFrame({
        "N": grp.size(),
        "RMSE": rmse,
        "MAE": mae,
        "BIAS": bias,
        "NSE": nse,
    })

    return df_metrics

# 4) Gráficos de pronósticos
def plot_spaghetti(
    df_h: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.3,
) -> plt.Axes:
    """
    Grafica todos los pronósticos superpuestos ("spaghetti").

    Parámetros
    ----------
    df_h : DataFrame
        Hindcast con columnas 'Fecha', 'Valor', 'Fecha_emitido'.
    ax : Axes, optional
        Eje de matplotlib donde dibujar. Si None, se crea uno nuevo.
    alpha : float
        Transparencia de las curvas individuales.

    Devuelve
    --------
    ax : Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Asegurar orden temporal
    df_h = df_h.sort_values(["Fecha_emitido", "Fecha"])

    for fecha_em in df_h["Fecha_emitido"].drop_duplicates():
        sub = df_h[df_h["Fecha_emitido"] == fecha_em]
        ax.plot(
            sub["Fecha"],
            sub["Valor"],
            color="tab:blue",
            alpha=alpha,
        )

    ax.set_title("Pronósticos")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nivel pronosticado [m]")
    ax.grid(alpha=0.3)
    return ax

def plot_obs_y_resumen_pronos(
    df_h: pd.DataFrame,
    serie_obs: pd.Series,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Grafica:
      - Serie observada
      - Media diaria (o horaria) de pronósticos
      - Banda P10–P90 de pronósticos

    Parámetros
    ----------
    df_h : DataFrame
        Hindcast con columnas 'Fecha' y 'Valor'.
    serie_obs : Series
        Serie observada, índice datetime.
    ax : Axes, optional
        Eje de matplotlib. Si None, se crea uno nuevo.

    Devuelve
    --------
    ax : Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Serie observada
    serie_obs = serie_obs.sort_index()
    ax.plot(serie_obs.index, serie_obs.values, label="Observado", color="black", linewidth=1.5)

    # Resumen de pronósticos por instante pronosticado (Fecha)
    df_h = df_h.copy()
    df_h = df_h.sort_values("Fecha")

    df_media = df_h.groupby("Fecha")["Valor"].mean()
    # df_p10 = df_h.groupby("Fecha")["Valor"].quantile(0.10)
    # df_p90 = df_h.groupby("Fecha")["Valor"].quantile(0.90)

    ax.plot(df_media.index, df_media.values,
            label="Pronósticos",# "Media pronósticos"
            color="tab:blue")
    # ax.fill_between(df_p10.index, df_p10, df_p90,
    #                 alpha=0.2,
    #                 color="tab:blue",
    #                 label="P10 - P90")

    #ax.set_title("Serie observada + resumen de pronósticos")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nivel [m]")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax

# 5) Gráficos de métricas
def plot_metricas_por_lead(
    df_metrics: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    incluir_nse: bool = True,
) -> Tuple[plt.Axes, Optional[plt.Axes]]:
    """
    Grafica RMSE y MAE por lead (en un mismo eje) y opcionalmente NSE
    en un eje secundario.

    Parámetros
    ----------
    df_metrics : DataFrame
        Resultado de `calcular_metricas_por_lead`.
    ax : Axes, optional
        Eje principal de matplotlib. Si None, se crea uno nuevo.
    incluir_nse : bool
        Si True, agrega NSE en eje secundario.

    Devuelve
    --------
    ax : Axes
        Eje de RMSE / MAE.
    ax2 : Axes or None
        Eje secundario de NSE (si incluir_nse=True).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    dfm = df_metrics.sort_index()

    ax.plot(dfm.index, dfm["RMSE"], marker="o", label="RMSE")
    ax.plot(dfm.index, dfm["MAE"], marker="s", label="MAE")
    ax.set_xlabel("Lead (horas)")
    ax.set_ylabel("Error")
    ax.set_title("Error por lead de pronóstico")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax2 = None
    if incluir_nse and "NSE" in dfm.columns:
        ax2 = ax.twinx()
        ax2.plot(dfm.index, dfm["NSE"], marker="^", linestyle="--", label="NSE")
        ax2.set_ylabel("NSE")
        ax2.axhline(0.0, color="grey", linestyle=":")
        # Manejar leyenda combinada
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    return ax, ax2

# Ejemplo de uso como script
if __name__ == "__main__":
    # Ejemplo mínimo de uso “standalone”.

    archivo_union = "series_nivel_union_h.csv"
    archivo_hindcast = "hindcast_diario_72h_1a3m_single_stations.csv"

    obs = cargar_serie_observada(archivo_union, nombre_columna="Misión La Paz")
    df_h = cargar_hindcast(archivo_hindcast)

    # 1) Spaghetti
    plot_spaghetti(df_h)
    plt.tight_layout()
    plt.show()

    # 2) Observado + resumen pronósticos
    plot_obs_y_resumen_pronos(df_h, obs)
    plt.tight_layout()
    plt.show()

    # 3) Métricas por lead
    df_merged = merge_prono_obs(df_h, obs)
    df_metrics = calcular_metricas_por_lead(df_merged)
    print(df_metrics)

    plot_metricas_por_lead(df_metrics)
    plt.tight_layout()
    plt.show()

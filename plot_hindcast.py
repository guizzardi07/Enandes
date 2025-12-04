# plot_hindcast.py
from __future__ import annotations
import matplotlib.pyplot as plt

from modulos import (
    cargar_serie_observada,
    cargar_hindcast,
    merge_prono_obs,
    calcular_metricas_por_lead,
    plot_spaghetti,
    plot_obs_y_resumen_pronos,
    plot_metricas_por_lead,
)

def main() -> None:
    archivo_union = "resultados/series_nivel_union_h.csv"
    archivo_hindcast = "resultados/hindcast_diario_72h_1a3m_single_stations.csv"

    obs = cargar_serie_observada(archivo_union, nombre_columna="Misión La Paz")
    df_h = cargar_hindcast(archivo_hindcast)

    # Spaghetti
    plot_spaghetti(df_h)
    plt.tight_layout()
    plt.show()

    # Observado + resumen pronósticos
    plot_obs_y_resumen_pronos(df_h, obs)
    plt.tight_layout()
    plt.show()

    # Métricas por lead
    df_merged = merge_prono_obs(df_h, obs)
    df_metrics = calcular_metricas_por_lead(df_merged)
    print(df_metrics)

    plot_metricas_por_lead(df_metrics)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# run_hindcast.py
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from a5client import Crud

from modulos import (
    construir_series_union,
    evaluar_estaciones_individuales,
    hindcast_diario,
)

# Configuración de logging SOLO a archivo

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "hindcast_logs.txt"

# Eliminar handlers
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
    force=True,
)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     force=True,
# )
logger = logging.getLogger(__name__)

def main() -> None:
    load_dotenv()
    client = Crud(url=os.getenv("A5_URL"), token=os.getenv("A5_TOKEN"))

    # Estaciones consideradas en el modelo
    Estaciones = {
        "Misión La Paz": 42293,
        "Villa Montes": 42291,
        "Puente Aruma": 42294,
    }
    
    carpeta_resultados = Path("resultados")
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    archivo_union = Path("resultados/series_nivel_union_h.csv")
    archivo_resumen = Path("resultados/resumen_series_niveles_h.xlsx")

    # Construir df_union (o leer si ya existe)
    if not archivo_union.exists():
        df_union, df_resumen = construir_series_union(
            Estaciones=Estaciones,
            timestart="2005-11-01",
            timeend=datetime.now(),
            step_adopt="h",
            client=client,
            carpeta_figuras="resultados/figuras_series_h",
            archivo_salida=str(archivo_union),
            archivo_salida_resumen= str(archivo_resumen),
        )
    else:
        logger.info("Cargando series unificadas desde %s", archivo_union)
        df_union = pd.read_csv(archivo_union, parse_dates=[0], index_col=0)
    
    # Evaluar estaciones individuales
    estaciones_up = ("Villa Montes", "Puente Aruma")
    orden, df_eval = evaluar_estaciones_individuales(
        df_union,
        estaciones=estaciones_up,
        obs_col="Misión La Paz",
        max_lag=72,
        ini_lag=2,
    )
    logger.info("Ranking estaciones:\n%s", df_eval)

    primary = orden[0]
    secondary = orden[1] if len(orden) > 1 else None
    logger.info("Usando primary=%s, secondary=%s", primary, secondary)
    
    # Hindcast diario
    df_hind = hindcast_diario(
        df_union=df_union,
        primary=primary,
        secondary=secondary,
        obs_col="Misión La Paz",
        fecha_ini_eval="2022-01-03",
        fecha_fin_eval="2024-12-31",
        freq="1h",
        ventana_lag=pd.DateOffset(years=1, months=3),
        ventana_ML=pd.DateOffset(years=1, months=1),
        horizonte_horas=42,
        min_muestras=200,
    )

    salida_csv = Path("resultados/hindcast_diario_72h_1a3m_single_stations.csv")
    df_hind.to_csv(salida_csv, index=False)
    logger.info("Hindcast diario guardado en %s", salida_csv)


if __name__ == "__main__":
    import pandas as pd
    main()

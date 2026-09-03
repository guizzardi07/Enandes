"""
modulos/plan_builder.py

Construcción de planes de `pydrodelta` (https://github.com/jbianchi81/pydrodelta)
a partir de lo que el usuario va configurando en la app.

Se generan dos planes independientes:

- **Operativo** (paso 1 hora): un procedimiento `LinearFit` por estación aguas
  arriba, con el lag estimado por correlación cruzada aplicado como `x_offset`
  de la serie de entrada.
- **Subestacional** (paso mensual): procedimientos `Persistence` y `Analogy`
  sobre la serie mensual de la estación objetivo.

Convenciones adoptadas
----------------------
- `id` del plan: placeholder editable (todavía no hay calibrados asignados en la
  API destino), por eso tampoco se escribe `cal_id`.
- node id = `estacion.id` y var id = `var.id` devueltos por la API A5.
- El shift vertical va en cero: el corrimiento lo absorbe el intercepto que
  `pydrodelta` recalibra en cada corrida (`extra_pars.tail_steps`).
- La limpieza no portable (ventanas de eliminación, corrimientos por tramo,
  detección de saltos con ventana) queda fuera del plan. Sí se traducen
  `lim_outliers` y `lim_jump`.
- Topology inline (como `EjemploPlan/confl-lag-and-route.yml`).
- Salidas locales a CSV, relativas a `./data/<caso>/` (relativas al directorio
  desde donde se corre pydrodelta). El YAML se guarda en `resultados/planes/`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

SCHEMA_URL = "https://alerta.ina.gob.ar/ina/pydrodelta/schema/plan.json"

# Placeholders: reemplazar por el id de calibrado real cuando exista en la API destino
PLAN_ID_OPERATIVO = 999901
PLAN_ID_SUBESTACIONAL = 999902

# Placeholders de node id / var id cuando no se pudo leer el metadata de A5
NODE_ID_PLACEHOLDER = 900000
VAR_ID_PLACEHOLDER = 2  # altura hidrométrica en A5

PLANES_DIR = Path("resultados") / "planes"
DATA_DIR_TPL = "./data/{caso}"


# Utilidades

def slugify(texto: str) -> str:
    """Normaliza un nombre para usarlo en ids, nombres de archivo y carpetas."""
    txt = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^\w\s-]", "", txt).strip().lower()
    txt = re.sub(r"[\s_]+", "-", txt)
    return re.sub(r"-{2,}", "-", txt) or "sin-nombre"


def _limpio(d: Dict[str, Any]) -> Dict[str, Any]:
    """Saca las claves con valor None, para no ensuciar el YAML."""
    return {k: v for k, v in d.items() if v is not None}


def _plain(obj: Any) -> Any:
    """Convierte tipos numpy/pandas/tuplas a tipos nativos serializables a YAML."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


# Referencias a series

@dataclass
class SerieRef:
    """Una serie de A5 con todo lo necesario para escribirla en la topología."""

    estacion: str
    serie_id: int
    estacion_id: Optional[int] = None
    var_id: Optional[int] = None
    tipo: str = "puntual"
    lim_outliers: Optional[Tuple[float, float]] = None
    lim_jump: Optional[float] = None
    x_offset_horas: int = 0
    y_offset: float = 0.0
    comment: Optional[str] = None

    @property
    def node_id(self) -> int:
        return int(self.estacion_id) if self.estacion_id is not None else NODE_ID_PLACEHOLDER

    @property
    def variable_id(self) -> int:
        return int(self.var_id) if self.var_id is not None else VAR_ID_PLACEHOLDER

    @property
    def node_variable(self) -> List[int]:
        return [self.node_id, self.variable_id]

    @property
    def completo(self) -> bool:
        """True si tiene node id y var id reales (no placeholders)."""
        return self.estacion_id is not None and self.var_id is not None

    def to_serie_dict(self) -> Dict[str, Any]:
        """Dict `nodeserie` para la topología."""
        d: Dict[str, Any] = {
            "series_id": int(self.serie_id),
            "tipo": self.tipo or "puntual",
        }
        if self.lim_outliers is not None:
            d["lim_outliers"] = [float(self.lim_outliers[0]), float(self.lim_outliers[1])]
        if self.lim_jump is not None:
            d["lim_jump"] = float(self.lim_jump)
        if self.x_offset_horas:
            d["x_offset"] = {"hours": int(self.x_offset_horas)}
        # y_offset siempre explícito: el ajuste vertical lo hace la calibración
        d["y_offset"] = float(self.y_offset)
        if self.comment:
            d["comment"] = self.comment
        return d


def params_limpieza_a_serie(ref: SerieRef, params: Optional[Dict[str, Any]]) -> SerieRef:
    """Vuelca los parámetros de limpieza portables de la app sobre la SerieRef."""
    if not params:
        return ref
    outliers = params.get("outliers")
    if outliers is not None and len(outliers) >= 2:
        ref.lim_outliers = (float(outliers[0]), float(outliers[1]))
    saltos = params.get("saltos")
    if saltos and saltos.get("umbral") is not None:
        ref.lim_jump = float(saltos["umbral"])
    return ref


def refs_desde_resumen(
    df_resumen: Optional[pd.DataFrame],
    estaciones: Dict[str, int],
    params_por_estacion: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, SerieRef]:
    """
    Arma un SerieRef por estación combinando la tabla de estaciones de la app
    con los metadatos de A5 que quedaron en `df_resumen` (estacion_id, var_id).
    """
    meta_por_est: Dict[str, Dict[str, Any]] = {}
    if df_resumen is not None and not df_resumen.empty and "Estacion" in df_resumen.columns:
        for _, fila in df_resumen.iterrows():
            meta_por_est[str(fila["Estacion"])] = fila.to_dict()

    refs: Dict[str, SerieRef] = {}
    for estacion, serie_id in estaciones.items():
        meta = meta_por_est.get(estacion, {})
        ref = SerieRef(
            estacion=estacion,
            serie_id=int(serie_id),
            estacion_id=_int_o_none(meta.get("estacion_id")),
            var_id=_int_o_none(meta.get("var_id")),
            tipo=str(meta.get("tipo") or "puntual"),
        )
        params = (params_por_estacion or {}).get(estacion)
        refs[estacion] = params_limpieza_a_serie(ref, params)
    return refs


def _int_o_none(valor: Any) -> Optional[int]:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def leer_metadata_serie(client, serie_id: int) -> Dict[str, Any]:
    """
    Consulta A5 para obtener estacion.id / var.id de una serie.

    Se usa para el plan subestacional, donde las series no pasan por el flujo
    del tab operativo. Pide una ventana chica para no bajar observaciones de más.
    """
    ahora = datetime.now()
    resp = client.readSerie(int(serie_id), ahora - timedelta(days=1), ahora)
    estacion = resp.get("estacion") or {}
    var = resp.get("var") or {}
    return {
        "serie_id": int(serie_id),
        "estacion_id": _int_o_none(estacion.get("id")),
        "estacion_nombre": estacion.get("nombre"),
        "var_id": _int_o_none(var.get("id")),
        "tipo": resp.get("tipo", "puntual"),
    }


# Nodos

def _node_dict(
    node_id: int,
    nombre: str,
    time_interval: Dict[str, int],
    var_id: int,
    series: List[Dict[str, Any]],
    interpolation_limit: Optional[Dict[str, int]] = None,
    extrapolate: Optional[bool] = None,
) -> Dict[str, Any]:
    variable = _limpio(
        {
            "id": int(var_id),
            "interpolation_limit": interpolation_limit,
            "extrapolate": extrapolate,
            "series": series,
        }
    )
    return {
        "id": int(node_id),
        "name": nombre,
        "time_interval": time_interval,
        "node_type": "station",
        "variables": [variable],
    }


# Plan operativo

def build_plan_operativo(
    obs_ref: SerieRef,
    upstream_refs: Sequence[SerieRef],
    caso: str,
    plan_id: int = PLAN_ID_OPERATIVO,
    nombre: str = "Pilcomayo - pronóstico operativo",
    dias_atras: int = 90,
    horas_adelante: int = 48,
    tail_steps: Optional[int] = None,
    interpolation_limit_horas: int = 24,
) -> Dict[str, Any]:
    """
    Plan de paso horario: un `LinearFit` por estación aguas arriba.

    Parámetros
    ----------
    obs_ref : SerieRef
        Estación objetivo (Misión La Paz).
    upstream_refs : Sequence[SerieRef]
        Estaciones aguas arriba seleccionadas para el ajuste, con `x_offset_horas`
        ya cargado con el lag adoptado.
    dias_atras / horas_adelante :
        Ventana de la topología, relativa al `forecast_date` (fecha de corrida).
    tail_steps :
        Cantidad de pasos finales usados para calibrar (equivale a la ventana de
        ajuste elegida en el Paso 3). Si es None, pydrodelta usa toda la serie.
    """
    data_dir = DATA_DIR_TPL.format(caso=caso)
    time_interval = {"hours": 1}
    interp = {"hours": int(interpolation_limit_horas)}

    nodos = [
        _node_dict(
            node_id=obs_ref.node_id,
            nombre=obs_ref.estacion,
            time_interval=time_interval,
            var_id=obs_ref.variable_id,
            series=[obs_ref.to_serie_dict()],
            interpolation_limit=interp,
        )
    ]
    for ref in upstream_refs:
        nodos.append(
            _node_dict(
                node_id=ref.node_id,
                nombre=ref.estacion,
                time_interval=time_interval,
                var_id=ref.variable_id,
                series=[ref.to_serie_dict()],
                interpolation_limit=interp,
                extrapolate=True,
            )
        )

    procedimientos = []
    for ref in upstream_refs:
        pid = f"linearfit-{slugify(ref.estacion)}-{slugify(obs_ref.estacion)}"
        extra_pars: Dict[str, Any] = {"warmup_steps": 0, "drop_warmup": False}
        if tail_steps:
            extra_pars["tail_steps"] = int(tail_steps)
        procedimientos.append(
            {
                "id": pid,
                "type": "LinearFit",
                "boundaries": [{"name": "input_1", "node_variable": ref.node_variable}],
                "outputs": [{"name": "output", "node_variable": obs_ref.node_variable}],
                "parameters": {},
                "extra_pars": extra_pars,
                "save_results": f"{data_dir}/{pid}.csv",
                "save_dict": f"{data_dir}/{pid}.json",
            }
        )

    return {
        "id": int(plan_id),
        "name": nombre,
        "forecast_date": {"hours": 0},
        "time_interval": time_interval,
        "output_sim_csv": f"{data_dir}/{caso}_sim.csv",
        "output_stats": f"{data_dir}/{caso}_stats.json",
        "topology": {
            "timestart": {"days": -int(dias_atras)},
            "timeend": {"hours": int(horas_adelante)},
            "interpolation_limit": interp,
            "save_variable": [
                {
                    "var_id": obs_ref.variable_id,
                    "output": f"{data_dir}/{caso}_{obs_ref.variable_id}.csv",
                    "format": "csv",
                    "pivot": True,
                }
            ],
            "report_file": f"{data_dir}/{caso}_report.json",
            "nodes": nodos,
        },
        "procedures": procedimientos,
    }


# Plan subestacional

def build_plan_subestacional(
    ref_actual: SerieRef,
    ref_historica: Optional[SerieRef],
    caso: str,
    plan_id: int = PLAN_ID_SUBESTACIONAL,
    nombre: str = "Pilcomayo - pronóstico subestacional",
    timestart: str = "1980-01-01T00:00:00.000Z",
    long_busqueda: int = 6,
    long_prono: int = 4,
    cantidad_analogos: int = 5,
    orden_analogos: str = "RMSE",
    orden_ascending: bool = True,
    time_window: str = "month",
) -> Dict[str, Any]:
    """
    Plan de paso mensual: `Persistence` + `Analogy` sobre la serie mensual.

    La serie actual va primera y la histórica segunda: en pydrodelta el orden de
    `series` es orden de prioridad, igual que la combinación que hace la app.
    """
    data_dir = DATA_DIR_TPL.format(caso=caso)
    time_interval = {"months": 1}

    series = [ref_actual.to_serie_dict()]
    series[0]["agg_func"] = "mean"
    if ref_historica is not None:
        serie_hist = ref_historica.to_serie_dict()
        serie_hist["agg_func"] = "mean"
        serie_hist["comment"] = "serie histórica"
        series.append(serie_hist)

    nodo = _node_dict(
        node_id=ref_actual.node_id,
        nombre=ref_actual.estacion,
        time_interval=time_interval,
        var_id=ref_actual.variable_id,
        series=series,
        interpolation_limit={"months": 1},
    )

    node_variable = ref_actual.node_variable

    persistencia = {
        "id": "persistencia-cuantiles",
        "type": "Persistence",
        "boundaries": [{"name": "input", "node_variable": node_variable}],
        "outputs": [{"name": "output", "node_variable": node_variable}],
        "parameters": {
            "search_length": int(long_busqueda),
            "forecast_length": int(long_prono),
            "time_window": time_window,
        },
        "save_results": f"{data_dir}/persistencia.csv",
        "save_dict": f"{data_dir}/persistencia.json",
    }

    analogia = {
        "id": "analogias-historicas",
        "type": "Analogy",
        "boundaries": [{"name": "input", "node_variable": node_variable}],
        "outputs": [{"name": "output", "node_variable": node_variable}],
        "parameters": {
            "search_length": int(long_busqueda),
            "forecast_length": int(long_prono),
            "number_of_analogs": int(cantidad_analogos),
            "order_by": str(orden_analogos),
            "ascending": bool(orden_ascending),
            "time_window": time_window,
        },
        "save_results": f"{data_dir}/analogias.csv",
        "save_dict": f"{data_dir}/analogias.json",
    }

    return {
        "id": int(plan_id),
        "name": nombre,
        "forecast_date": {"days": 0},
        "time_interval": time_interval,
        "output_sim_csv": f"{data_dir}/{caso}_sim.csv",
        "output_stats": f"{data_dir}/{caso}_stats.json",
        "topology": {
            "timestart": timestart,
            "timeend": {"months": int(long_prono)},
            "nodes": [nodo],
            "save_variable": [
                {
                    "var_id": ref_actual.variable_id,
                    "output": f"{data_dir}/{caso}_{ref_actual.variable_id}.csv",
                    "format": "csv",
                    "pivot": True,
                }
            ],
            "report_file": f"{data_dir}/{caso}_report.json",
        },
        "procedures": [persistencia, analogia],
    }


# Serialización

def plan_to_yaml(plan: Dict[str, Any], comentarios: Optional[Iterable[str]] = None) -> str:
    """Serializa el plan a YAML, con la línea de schema arriba (como el ejemplo)."""
    encabezado = [f"# yaml-language-server: $schema={SCHEMA_URL}"]
    encabezado += [f"# {linea}" for linea in (comentarios or [])]
    cuerpo = yaml.safe_dump(
        _plain(plan),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )
    return "\n".join(encabezado) + "\n" + cuerpo


def guardar_plan(texto_yaml: str, nombre_archivo: str, carpeta: Path | str = PLANES_DIR) -> Path:
    """Guarda el YAML en disco y devuelve la ruta."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre_archivo
    ruta.write_text(texto_yaml, encoding="utf-8")
    return ruta


def advertencias_plan(plan: Dict[str, Any], refs: Sequence[SerieRef]) -> List[str]:
    """Chequeos mínimos antes de entregar el archivo."""
    avisos: List[str] = []
    incompletas = [r.estacion for r in refs if not r.completo]
    if incompletas:
        avisos.append(
            "Sin metadata de A5 (se usaron placeholders de node id / var id): "
            + ", ".join(incompletas)
        )
    if len(plan.get("procedures", [])) > 1:
        salidas = [tuple(o["node_variable"]) for p in plan["procedures"] for o in p["outputs"]]
        if len(set(salidas)) < len(salidas):
            avisos.append(
                "Hay más de un procedimiento escribiendo sobre la misma variable de nodo. "
                "Con overwrite=false (default) el segundo solo completa los huecos que dejó "
                "el primero; cada modelo queda igual en su propio save_results."
            )
    return avisos

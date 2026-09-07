"""
modulos/plan_builder.py

Construcción de planes de `pydrodelta` (https://github.com/jbianchi81/pydrodelta)
a partir de lo que el usuario va configurando en la app.

Se generan planes independientes:

- **Operativo** (paso 1 hora): *un plan por estación aguas arriba*, cada uno con
  un único `LinearFit` y el lag estimado por correlación cruzada aplicado como
  `x_offset` de la serie de entrada. Van separados porque en pydrodelta dos
  procedimientos que escriben sobre la misma `node_variable` no se combinan: con
  `overwrite=false` (default) el segundo solo rellena los huecos que dejó el
  primero.
- **Subestacional** (paso mensual): *un plan por método*, `Persistence` y
  `Analogy` por separado, sobre la serie mensual de la estación objetivo. Se
  separan por el mismo motivo que los operativos.

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
- `series_sim` en la variable de salida: es donde pydrodelta deposita la
  simulación. Sin esa clave el plan corre igual, pero `output_sim_csv` sale
  vacío (la salida se arma recorriendo `series_sim`).
- `qualifiers: [superior, inferior]`: `LinearFit`, `Persistence` y `Analogy`
  calculan siempre la banda de error; sin declararla acá se descarta al escribir
  la salida.
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
PLAN_ID_OPERATIVO = 999901      # los operativos toman ids consecutivos desde acá
PLAN_ID_SUBESTACIONAL = 999911

# Placeholders de node id / var id cuando no se pudo leer el metadata de A5
NODE_ID_PLACEHOLDER = 900000
VAR_ID_PLACEHOLDER = 2  # altura hidrométrica en A5

PLANES_DIR = Path("resultados") / "planes"
DATA_DIR_TPL = "./data/{caso}"

# Miembros de la banda de error que LinearFit / Persistence / Analogy generan
# siempre y que hay que pedir explícitamente para que lleguen a la salida.
QUALIFIERS_DEFAULT: Tuple[str, ...] = ("superior", "inferior")


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
    series_sim: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    variable = _limpio(
        {
            "id": int(var_id),
            "interpolation_limit": interpolation_limit,
            "extrapolate": extrapolate,
            "series": series,
            "series_sim": series_sim,
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

@dataclass
class PlanOperativo:
    """Un plan operativo ya armado, con el nombre de archivo que le corresponde."""

    estacion: str
    caso: str
    nombre_archivo: str
    plan: Dict[str, Any]


def build_plan_operativo(
    obs_ref: SerieRef,
    upstream_ref: SerieRef,
    caso: str,
    plan_id: int = PLAN_ID_OPERATIVO,
    nombre: str = "Pilcomayo - pronóstico operativo",
    dias_atras: int = 90,
    horas_adelante: int = 48,
    tail_steps: Optional[int] = None,
    interpolation_limit_horas: int = 24,
    series_sim_id: Optional[int] = None,
    qualifiers: Optional[Sequence[str]] = QUALIFIERS_DEFAULT,
    plot_pdf: bool = True,
) -> Dict[str, Any]:
    """
    Plan de paso horario con un único `LinearFit`: una estación aguas arriba
    contra la estación objetivo.

    Parámetros
    ----------
    obs_ref : SerieRef
        Estación objetivo (Misión La Paz).
    upstream_ref : SerieRef
        Estación aguas arriba del ajuste, con `x_offset_horas` ya cargado con el
        lag adoptado.
    dias_atras / horas_adelante :
        Ventana de la topología, relativa al `forecast_date` (fecha de corrida).
    tail_steps :
        Cantidad de pasos finales usados para calibrar (equivale a la ventana de
        ajuste elegida en el Paso 3). Si es None, pydrodelta usa toda la serie.
    series_sim_id :
        `series_id` de A5 donde pydrodelta deposita la simulación. Sin esto el
        plan corre pero `output_sim_csv` sale vacío.
    qualifiers :
        Miembros de la banda de error a incluir en la salida. `LinearFit` calcula
        `inferior`/`superior` (95% de los residuos del ajuste) en toda corrida,
        pero se descartan si no se piden acá.
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
            series_sim=[{"series_id": int(series_sim_id)}] if series_sim_id else None,
        ),
        _node_dict(
            node_id=upstream_ref.node_id,
            nombre=upstream_ref.estacion,
            time_interval=time_interval,
            var_id=upstream_ref.variable_id,
            series=[upstream_ref.to_serie_dict()],
            interpolation_limit=interp,
            extrapolate=True,
        ),
    ]

    pid = f"linearfit-{slugify(upstream_ref.estacion)}-{slugify(obs_ref.estacion)}"
    extra_pars: Dict[str, Any] = {"warmup_steps": 0, "drop_warmup": False}
    if tail_steps:
        extra_pars["tail_steps"] = int(tail_steps)
    procedimiento = {
        "id": pid,
        "type": "LinearFit",
        "boundaries": [{"name": "input_1", "node_variable": upstream_ref.node_variable}],
        "outputs": [{"name": "output", "node_variable": obs_ref.node_variable}],
        "parameters": {},
        "extra_pars": extra_pars,
        "save_results": f"{data_dir}/{pid}.csv",
        "save_dict": f"{data_dir}/{pid}.json",
    }

    topology: Dict[str, Any] = {
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
    }
    if plot_pdf:
        topology["plot_variable"] = [
            {
                "var_id": obs_ref.variable_id,
                "output": f"{data_dir}/{caso}_{obs_ref.variable_id}.pdf",
                "timestart": {"days": -int(dias_atras)},
                "extra_sim_columns": True,
            }
        ]
    topology["report_file"] = f"{data_dir}/{caso}_report.json"
    topology["nodes"] = nodos

    plan: Dict[str, Any] = {
        "id": int(plan_id),
        "name": nombre,
        "forecast_date": {"hours": 0},
        "time_interval": time_interval,
        "output_sim_csv": f"{data_dir}/{caso}_sim.csv",
        "output_stats": f"{data_dir}/{caso}_stats.json",
    }
    if qualifiers:
        plan["qualifiers"] = [str(q) for q in qualifiers]
    plan["topology"] = topology
    plan["procedures"] = [procedimiento]
    return plan


def build_planes_operativos(
    obs_ref: SerieRef,
    upstream_refs: Sequence[SerieRef],
    caso: str,
    plan_id_base: int = PLAN_ID_OPERATIVO,
    nombre_base: str = "Pilcomayo - pronóstico operativo",
    dias_atras: int = 90,
    horas_adelante: int = 48,
    tail_steps: Optional[int] = None,
    interpolation_limit_horas: int = 24,
    series_sim_por_estacion: Optional[Dict[str, int]] = None,
    qualifiers: Optional[Sequence[str]] = QUALIFIERS_DEFAULT,
    plot_pdf: bool = True,
) -> List[PlanOperativo]:
    """
    Un plan por estación aguas arriba.

    No se juntan varias regresiones en un mismo plan: en pydrodelta dos
    procedimientos que apuntan a la misma `node_variable` de salida no se
    promedian ni se comparan, con `overwrite=false` (default) el segundo solo
    rellena los huecos que dejó el primero. Cada plan lleva su propio `id`
    (consecutivo desde `plan_id_base`) y su propia carpeta de salidas
    `./data/<caso>-<estación>/`, así no se pisan entre sí.
    """
    series_sim_por_estacion = series_sim_por_estacion or {}
    planes: List[PlanOperativo] = []
    for i, ref in enumerate(upstream_refs):
        caso_plan = f"{caso}-{slugify(ref.estacion)}"
        plan = build_plan_operativo(
            obs_ref=obs_ref,
            upstream_ref=ref,
            caso=caso_plan,
            plan_id=int(plan_id_base) + i,
            nombre=f"{nombre_base} desde {ref.estacion}",
            dias_atras=dias_atras,
            horas_adelante=horas_adelante,
            tail_steps=tail_steps,
            interpolation_limit_horas=interpolation_limit_horas,
            series_sim_id=series_sim_por_estacion.get(ref.estacion),
            qualifiers=qualifiers,
            plot_pdf=plot_pdf,
        )
        planes.append(
            PlanOperativo(
                estacion=ref.estacion,
                caso=caso_plan,
                nombre_archivo=f"plan_operativo_{caso_plan}.yml",
                plan=plan,
            )
        )
    return planes


# Plan subestacional

# Métodos del módulo subestacional: clave -> nombre para el título del plan.
METODOS_SUBESTACIONAL: Dict[str, str] = {
    "persistencia": "persistencia de cuantiles",
    "analogia": "analogías históricas",
}


@dataclass
class PlanSubestacional:
    """Un plan subestacional ya armado, con el nombre de archivo que le corresponde."""

    metodo: str
    caso: str
    nombre_archivo: str
    plan: Dict[str, Any]


def _procedimiento_subestacional(
    metodo: str,
    node_variable: List[int],
    data_dir: str,
    long_busqueda: int,
    long_prono: int,
    cantidad_analogos: int,
    orden_analogos: str,
    orden_ascending: bool,
    time_window: str,
) -> Dict[str, Any]:
    """Arma el `Persistence` o el `Analogy` del plan, según el método pedido."""
    if metodo == "persistencia":
        pid = "persistencia-cuantiles"
        tipo = "Persistence"
        parameters: Dict[str, Any] = {
            "search_length": int(long_busqueda),
            "forecast_length": int(long_prono),
            "time_window": time_window,
        }
    elif metodo == "analogia":
        pid = "analogias-historicas"
        tipo = "Analogy"
        parameters = {
            "search_length": int(long_busqueda),
            "forecast_length": int(long_prono),
            "number_of_analogs": int(cantidad_analogos),
            "order_by": str(orden_analogos),
            "ascending": bool(orden_ascending),
            "time_window": time_window,
        }
    else:
        raise ValueError(
            f"Método subestacional desconocido: {metodo!r}. "
            f"Esperaba uno de {sorted(METODOS_SUBESTACIONAL)}."
        )

    return {
        "id": pid,
        "type": tipo,
        "boundaries": [{"name": "input", "node_variable": node_variable}],
        "outputs": [{"name": "output", "node_variable": node_variable}],
        "parameters": parameters,
        "save_results": f"{data_dir}/{pid}.csv",
        "save_dict": f"{data_dir}/{pid}.json",
    }


def build_plan_subestacional(
    ref_actual: SerieRef,
    ref_historica: Optional[SerieRef],
    caso: str,
    metodo: str = "persistencia",
    plan_id: int = PLAN_ID_SUBESTACIONAL,
    nombre: str = "Pilcomayo - pronóstico subestacional",
    timestart: str = "1980-01-01T00:00:00.000Z",
    long_busqueda: int = 6,
    long_prono: int = 4,
    cantidad_analogos: int = 5,
    orden_analogos: str = "RMSE",
    orden_ascending: bool = True,
    time_window: str = "month",
    series_sim_id: Optional[int] = None,
    qualifiers: Optional[Sequence[str]] = QUALIFIERS_DEFAULT,
    plot_pdf: bool = True,
) -> Dict[str, Any]:
    """
    Plan de paso mensual con un único procedimiento: `Persistence` o `Analogy`.

    Los dos métodos van en planes separados por la misma razón que las regresiones
    del plan operativo: apuntan a la misma `node_variable` de salida y pydrodelta
    no los combina, con `overwrite=false` (default) el segundo solo rellena los
    huecos que dejó el primero.

    La serie actual va primera y la histórica segunda: en pydrodelta el orden de
    `series` es orden de prioridad, igual que la combinación que hace la app.
    """
    if metodo not in METODOS_SUBESTACIONAL:
        raise ValueError(
            f"Método subestacional desconocido: {metodo!r}. "
            f"Esperaba uno de {sorted(METODOS_SUBESTACIONAL)}."
        )

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
        series_sim=[{"series_id": int(series_sim_id)}] if series_sim_id else None,
    )

    procedimiento = _procedimiento_subestacional(
        metodo=metodo,
        node_variable=ref_actual.node_variable,
        data_dir=data_dir,
        long_busqueda=long_busqueda,
        long_prono=long_prono,
        cantidad_analogos=cantidad_analogos,
        orden_analogos=orden_analogos,
        orden_ascending=orden_ascending,
        time_window=time_window,
    )

    topology: Dict[str, Any] = {
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
    }
    if plot_pdf:
        topology["plot_variable"] = [
            {
                "var_id": ref_actual.variable_id,
                "output": f"{data_dir}/{caso}_{ref_actual.variable_id}.pdf",
                "timestart": timestart,
                "extra_sim_columns": True,
            }
        ]
    topology["report_file"] = f"{data_dir}/{caso}_report.json"

    plan: Dict[str, Any] = {
        "id": int(plan_id),
        "name": nombre,
        "forecast_date": {"days": 0},
        "time_interval": time_interval,
        "output_sim_csv": f"{data_dir}/{caso}_sim.csv",
        "output_stats": f"{data_dir}/{caso}_stats.json",
    }
    if qualifiers:
        plan["qualifiers"] = [str(q) for q in qualifiers]
    plan["topology"] = topology
    plan["procedures"] = [procedimiento]
    return plan


def build_planes_subestacionales(
    ref_actual: SerieRef,
    ref_historica: Optional[SerieRef],
    caso: str,
    metodos: Sequence[str] = ("persistencia", "analogia"),
    plan_id_base: int = PLAN_ID_SUBESTACIONAL,
    nombre_base: str = "Pilcomayo - pronóstico subestacional",
    timestart: str = "1980-01-01T00:00:00.000Z",
    long_busqueda: int = 6,
    long_prono: int = 4,
    cantidad_analogos: int = 5,
    orden_analogos: str = "RMSE",
    orden_ascending: bool = True,
    time_window: str = "month",
    series_sim_por_metodo: Optional[Dict[str, int]] = None,
    qualifiers: Optional[Sequence[str]] = QUALIFIERS_DEFAULT,
    plot_pdf: bool = True,
) -> List[PlanSubestacional]:
    """
    Un plan por método subestacional (persistencia y analogías).

    Mismo criterio que `build_planes_operativos`: cada método lleva su propio
    `id` (consecutivo desde `plan_id_base`) y su propia carpeta de salidas
    `./data/<caso>-<metodo>/`, para que los dos pronósticos queden completos y
    comparables en lugar de pisarse sobre la misma variable de nodo.
    """
    series_sim_por_metodo = series_sim_por_metodo or {}
    planes: List[PlanSubestacional] = []
    for i, metodo in enumerate(metodos):
        caso_plan = f"{caso}-{slugify(metodo)}"
        plan = build_plan_subestacional(
            ref_actual=ref_actual,
            ref_historica=ref_historica,
            caso=caso_plan,
            metodo=metodo,
            plan_id=int(plan_id_base) + i,
            nombre=f"{nombre_base} por {METODOS_SUBESTACIONAL[metodo]}",
            timestart=timestart,
            long_busqueda=long_busqueda,
            long_prono=long_prono,
            cantidad_analogos=cantidad_analogos,
            orden_analogos=orden_analogos,
            orden_ascending=orden_ascending,
            time_window=time_window,
            series_sim_id=series_sim_por_metodo.get(metodo),
            qualifiers=qualifiers,
            plot_pdf=plot_pdf,
        )
        planes.append(
            PlanSubestacional(
                metodo=metodo,
                caso=caso_plan,
                nombre_archivo=f"plan_subestacional_{caso_plan}.yml",
                plan=plan,
            )
        )
    return planes


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
    salidas = [
        tuple(o["node_variable"])
        for p in plan.get("procedures", [])
        for o in p.get("outputs", [])
    ]
    if len(set(salidas)) < len(salidas):
        avisos.append(
            "Hay más de un procedimiento escribiendo sobre la misma variable de nodo. "
            "Con overwrite=false (default) el segundo solo completa los huecos que dejó "
            "el primero; cada modelo queda igual en su propio save_results."
        )

    con_sim = {
        (nodo["id"], var["id"])
        for nodo in plan.get("topology", {}).get("nodes", [])
        for var in nodo.get("variables", [])
        if var.get("series_sim")
    }
    sin_sim = sorted(set(salidas) - con_sim)
    if sin_sim:
        avisos.append(
            "Falta `series_sim` en la variable de salida "
            + ", ".join(f"[{n}, {v}]" for n, v in sin_sim)
            + ": el plan corre igual, pero `output_sim_csv` sale vacío porque pydrodelta "
            "arma la salida recorriendo `series_sim`. Cargá el series_id de la serie de "
            "simulación en A5."
        )
    return avisos

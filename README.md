# Tablero de Control – Pilcomayo

Aplicación desarrollada en **Python** y **Streamlit** para la descarga, limpieza, análisis y pronóstico hidrológico de niveles para la estación **Misión La Paz**, en el río Pilcomayo.

El tablero integra dos componentes principales:

1. **Pronóstico operativo de corto plazo** , basado en estaciones aguas arriba.
2. **Pronóstico hidrológico subestacional mensual**, basado en persistencia de cuantiles y analogías históricas.

La aplicación utiliza series provenientes de la API A5 del INA y organiza el flujo de trabajo en una interfaz interactiva reproducible.

---
## Funcionalidades principales

### Pronóstico operativo

Permite construir un pronóstico horario de niveles para **Misión La Paz** a partir de estaciones aguas arriba, principalmente:

- **Villa Montes**
- **Puente Aruma**

El flujo incluye:

- descarga de series horarias desde A5;
- limpieza automática de series;
- remuestreo y regularización a paso horario;
- estimación de lags entre estaciones;
- ajuste de modelos lineales nivel–nivel;
- diagnóstico del ajuste;
- pronóstico operativo de corto plazo;
- ajuste vertical manual del modelo;
- visualización estática e interactiva;
- descarga de resultados en CSV o Excel.

### Pronóstico subestacional

El módulo subestacional genera un pronóstico mensual mediante dos métodos complementarios:

1. **Persistencia de cuantiles**
2. **Analogías históricas**

El flujo subestacional incluye:

- descarga y combinación de serie histórica y serie actual;
- limpieza de outliers;
- regularización diaria;
- corrección por mínimo hidráulico anual;
- agregado mensual;
- inferencia automática del mes de emisión;
- cálculo de pronóstico por persistencia;
- búsqueda de años análogos;
- cálculo de métricas de similitud;
- pronóstico ponderado por analogías;
- visualización de resultados y descarga de tablas.

---

## Estructura general de la aplicación

La app está organizada en dos pestañas principales:

```text
Pronóstico operativo | Pronóstico subestacional
```

### Pestaña 1 — Pronóstico operativo

Esta pestaña mantiene el flujo de trabajo de corto plazo dividido en tres pasos:

1. **Descargar y limpiar series**
2. **Estimar lag por estación**
3. **Ajustar modelo y generar pronóstico operativo**

### Pestaña 2 — Pronóstico subestacional

Esta pestaña permite configurar y ejecutar el pronóstico mensual subestacional. El usuario puede definir:

- cantidad de meses usados para la búsqueda de analogías;
- cantidad de meses a pronosticar;
- cantidad de años análogos seleccionados;
- criterio de ordenamiento de analogías;
- origen del pronóstico, automático o manual.

---

## Instalación

### 0. Instalar Python

La aplicación requiere **Python 3.10 o superior**.

Descargar Python desde:

```text
https://www.python.org/downloads/
```

Durante la instalación en Windows, marcar:

- **Add Python to PATH**
- instalación estándar recomendada

Verificar la instalación:

```bash
python --version
```

---

### 1. Clonar el repositorio

```bash
git clone https://github.com/guizzardi07/Enandes.git
cd Enandes
```

---

### 2. Crear entorno virtual

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / Mac

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

---

### 4. Instalar dependencias

```bash
pip install pandas numpy matplotlib plotly streamlit statsmodels openpyxl python-dotenv pytz a5-client
```

---

## Configuración de credenciales A5

El sistema utiliza **python-dotenv** para leer credenciales desde un archivo `.env`.

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```env
A5_URL="https://alerta.ina.gob.ar/a6"
A5_TOKEN="tu-token"
```

La URL del servicio A5 está fijada en la aplicación como:

```text
https://alerta.ina.gob.ar/a6
```

El token puede cargarse desde el archivo `.env` o ingresarse manualmente en la barra lateral de la app.

---

## Ejecución de la aplicación

### Windows

Desde la carpeta del proyecto, ejecutar:

```text
iniciar_tablero.bat
```

El script activa el entorno virtual, ejecuta Streamlit y abre el tablero en el navegador.

También puede ejecutarse manualmente con:

```bash
streamlit run app_Prono_MLP.py
```

---

### Linux / Mac

Dar permisos de ejecución al script:

```bash
chmod +x iniciar_tablero.sh
```

Ejecutar:

```bash
./iniciar_tablero.sh
```

O manualmente:

```bash
streamlit run app_Prono_MLP.py
```

---

# Guía de uso

## Configuración inicial

En la barra lateral:

1. Verificar la URL fija del servicio A5.
2. Ingresar o confirmar el **A5_TOKEN**.

---

# Pronóstico operativo

## Paso 1 — Descargar y limpiar series

En este paso se define el período de descarga y se ejecuta la construcción de las series limpias.

La aplicación:

- descarga las series desde A5;
- identifica la columna de nivel;
- aplica parámetros de limpieza por estación;
- elimina outliers;
- detecta saltos;
- interpola huecos cortos;
- regulariza las series a paso horario;
- construye un `df_union` con una columna por estación.

### Salidas

Se generan automáticamente:

```text
resultados/series_nivel_union_1H.csv
resultados/series_nivel_union_1H.xlsx
resultados/resumen_series_niveles_1H.xlsx
resultados/figuras/
```

Además, desde la interfaz se puede descargar la serie limpia en:

- CSV
- Excel

---

## Paso 2 — Estimación de lag

En este paso se estima el tiempo de respuesta entre cada estación aguas arriba y la estación objetivo **Misión La Paz**.

El usuario define:

- ventana temporal para estimar el lag;
- lag mínimo;
- lag máximo.

La aplicación devuelve una tabla con:

- estación;
- lag óptimo;
- correlación máxima;
- cantidad de datos utilizados;
- lag manual editable.

El `lag_manual` permite corregir o fijar manualmente el tiempo de traslado adoptado.

También se genera un gráfico con las series alineadas según el lag adoptado.

---

## Paso 3 — Ajuste, diagnóstico y pronóstico operativo

En este paso se ajusta un modelo lineal entre la estación objetivo y una o dos estaciones aguas arriba.

El usuario selecciona:

- estaciones aguas arriba a utilizar;
- ventana de calibración;
- shift vertical opcional para cada estación.

### Diagnóstico opcional

El panel de diagnóstico permite revisar:

- R² del ajuste;
- cantidad de muestras;
- intercepto;
- pendiente;
- gráfico temporal observado vs ajustado;
- scatter observado vs ajustado.

### Pronóstico operativo

La app genera una curva compuesta por:

- observado de la última semana;
- ajuste reciente;
- pronóstico futuro;
- línea vertical indicando el momento de emisión.

También se genera una versión interactiva del gráfico con Plotly.

### Descarga de resultados

Los resultados finales pueden descargarse en CSV o Excel.

---

# Pronóstico subestacional

El módulo subestacional está orientado a generar un pronóstico mensual a partir de la información histórica y reciente de la estación **Puerto Pilcomayo**.

## Preparación de datos

El flujo de preparación incluye:

1. descarga de serie actual e histórica desde A5;
2. combinación de ambas series, priorizando la serie actual;
3. eliminación de valores fuera de rango;
4. regularización diaria;
5. interpolación de huecos cortos;
6. corrección por mínimo hidráulico anual;
7. agregado mensual.

La corrección por mínimo hidráulico utiliza el percentil bajo anual por año hidrológico y una tendencia suavizada para ajustar desplazamientos verticales de la serie.

---

## Parámetros configurables

Desde la interfaz se pueden definir:

- **Meses de búsqueda**: cantidad de meses previos usados para comparar la situación actual con años históricos.
- **Meses a pronosticar**: horizonte mensual del pronóstico.
- **Cantidad de análogos**: número de años similares seleccionados.
- **Ordenar analogías por**: criterio usado para seleccionar los años análogos.

Criterios disponibles:

- `RMSE`
- `Score`
- `CoefC`
- `Nash`
- `ErrVol`

---

## Método 1 — Persistencia de cuantiles

El método de persistencia calcula el cuantil correspondiente al valor mensual del mes de origen respecto de los valores históricos del mismo mes.

Luego aplica ese cuantil a los meses futuros del horizonte de pronóstico.

Por ejemplo, si el mes actual se ubica en un cuantil alto respecto de su climatología mensual, el pronóstico mantiene esa condición relativa para los meses siguientes.

La salida incluye:

- fecha pronosticada;
- horizonte;
- valor pronosticado;
- cuantil base utilizado.

---

## Método 2 — Analogías históricas

El método de analogías busca años históricos cuya evolución previa sea similar a la situación actual.

Para comparar trazas históricas se aplica una transformación logarítmica y una estandarización por mes. Esto permite comparar anomalías relativas entre meses con distinta distribución hidrológica.

Para cada año candidato se calculan métricas de similitud:

- **RMSE**: error cuadrático medio;
- **CoefC**: coeficiente de correlación lineal;
- **Nash**: eficiencia de Nash-Sutcliffe;
- **ErrVol**: error porcentual de volumen acumulado;
- **Score**: indicador combinado de similitud.

Luego se seleccionan los mejores años análogos y se construye un pronóstico ponderado. Los pesos se asignan en función inversa al RMSE.

---

## Resultados subestacionales

La pestaña subestacional muestra:

- resumen del origen del pronóstico;
- período cubierto por la serie mensual;
- cantidad de meses sin datos;
- parámetros usados;
- tabla de pronóstico por persistencia;
- tabla de pronóstico por analogías;
- años análogos seleccionados;
- serie mensual usada por el modelo.

---

## Gráficos subestacionales

El módulo incluye los siguientes gráficos:

### Pronóstico mensual interactivo

Muestra en una misma figura:

- observaciones recientes;
- pronóstico por persistencia;
- pronóstico por analogía.

### Boxplot histórico + persistencia

Compara el pronóstico por persistencia contra la distribución histórica mensual.

### Trazas análogas seleccionadas

Muestra:

- la traza objetivo;
- los años análogos seleccionados;
- el pronóstico final por analogía;
- los meses previos usados para la búsqueda;
- los meses futuros pronosticados.

### Comparación mensual

Compara observaciones recientes, persistencia y analogía en una misma figura.

---

## Descargas subestacionales

Desde la interfaz se pueden descargar:

- `pronostico_subestacional.csv`
- `serie_mensual_subestacional.csv`
- `analogos_subestacional.csv`

---

# Módulos principales

## `app_Prono_MLP.py`

Aplicación principal de Streamlit.

Contiene:

- configuración general de la interfaz;
- barra lateral con credenciales;
- pestaña de pronóstico operativo;
- pestaña de pronóstico subestacional;
- gráficos interactivos con Plotly;
- gestión del estado de sesión.

---

## `series.py`

Contiene el flujo de descarga, limpieza y unificación de series.

Funciones principales:

- `leer_serie_nivel_estacion`
- `analizar_frecuencia_y_faltantes`
- `limpiar_y_rellenar_serie`
- `analiza_series_nivel`
- `construir_series_union`

---

## `hindcast.py`

Contiene funciones para estimación de lags, ajuste de modelos y pronóstico operativo.

Funciones principales:

- `estimar_lags_por_estacion`
- `get_lag_for_station`
- `ajustar_estacion_con_lag`
- `forecast_from_upstream`
- `forecast_horizon_from_upstream_last`
- `hindcast_diario`

---

## `subestacional.py`

Contiene el flujo completo del pronóstico mensual subestacional.

Incluye:

- clases de configuración con `dataclass`;
- descarga de serie actual e histórica;
- limpieza y regularización;
- corrección por mínimo hidráulico;
- agregado mensual;
- persistencia de cuantiles;
- analogías históricas;
- métricas de similitud;
- gráficos de diagnóstico y pronóstico.

Funciones principales:

- `prepare_regular_series`
- `resample_series`
- `forecast_persistence`
- `forecast_analogy`
- `plot_forecast_boxplot`
- `plot_analogy_traces`
- `plot_forecasts_comparison`
- `get_last_valid_forecast_origin`

---

## `limpieza_series.py`

Incluye utilidades para limpieza de series:

- eliminación de ventanas;
- corrimientos verticales;
- remoción de outliers;
- detección de saltos;
- inferencia de frecuencia;
- gráficos de series de nivel;
- parámetros de limpieza por estación.

---

## `plotting.py`

Incluye funciones de visualización para la app operativa:

- series temporales con grilla adaptativa;
- scatter observado vs ajustado;
- colores fijos por estación.

Convención de colores:

- **Misión La Paz**: rojo
- **Villa Montes**: verde
- **Puente Aruma**: violeta

---

## `io_utils.py`

Utilidades para exportación:

- conversión de DataFrame a CSV bytes;
- conversión de DataFrame a Excel bytes;
- sanitización de nombres de archivo.

---

## `utils_time.py`

Utilidades para fechas y pasos temporales:

- fecha local actual;
- conversión de fechas de inicio y fin;
- parseo del paso temporal;
- ventanas por defecto para gráficos.

---

## `utils_series.py`

Utilidades para alinear series por lag:

- desplazamiento temporal de estaciones aguas arriba;
- armado de DataFrames para gráficos con lag aplicado.

---

# Archivos generados

La aplicación genera archivos automáticamente en la carpeta:

```text
resultados/
```

Archivos principales:

```text
series_nivel_union_1H.csv
series_nivel_union_1H.xlsx
resumen_series_niveles_1H.xlsx
```

Figuras:

```text
resultados/figuras/
```

Las descargas realizadas desde botones de Streamlit se guardan en la carpeta de descargas del navegador.

---

# Notas operativas

- El paso temporal del pronóstico operativo es fijo: **1 hora**.
- Los lags se interpretan en horas.
- La estación objetivo operativa es **Misión La Paz**.
- La URL de A5 está fijada en la app.
- El token puede leerse desde `.env` o ingresarse manualmente.
- La calibración operativa utiliza la ventana definida por el usuario.
- El shift vertical es un ajuste manual posterior al modelo lineal.
- El origen subestacional puede inferirse automáticamente a partir del último mes mensual válido.

---

# Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Statsmodels
- OpenPyXL
- python-dotenv
- pytz
- a5-client

---

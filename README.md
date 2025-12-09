Sistema de hindcast hidrométrico (Pilcomayo – Misión La Paz)

Este repositorio implementa un flujo completo para:

Descargar series de nivel desde la API A5.
Limpiar y homogeneizar las series.
Ajustar modelos lineales con retardo (lag) entre estaciones aguas arriba y Misión La Paz, seleccionar la mejor estación predictora y generar hindcasts diarios.
Evaluar el desempeño de los pronósticos y generar gráficos de diagnóstico.

Estructura principal del proyecto

Los scripts más importantes son:

run_hindcast.py:
Descarga y limpia las series de nivel desde A5.
Construye el archivo unificado resultados/series_nivel_union_h.csv y un resumen en resultados/resumen_series_niveles_h.xlsx.
Evalúa cada estación aguas arriba de forma individual para elegir la mejor y la secundaria.
Corre el hindcast diario y guarda los pronósticos en la carpeta resultados.

plot_hindcast.py:
Carga la serie observada de Misión La Paz desde series_nivel_union_h.csv.
Carga el archivo de hindcast generado por run_hindcast.py.
Genera gráficos y métricas por lead de pronóstico.

modulos/limpieza_series.py: funciones para limpieza de series (eliminación de ventanas, corrimientos, outliers, saltos, gráficos rápidos) y diccionario de parámetros por estación.
modulos/series.py (en el repo puede llamarse series.py/resultados.py según cómo lo organizaste): flujo de descarga, análisis de frecuencia, limpieza y unificación de las series en un único DataFrame.
modulos/hindcast.py: evaluación de estaciones individuales, calibración ventana a ventana y generación del hindcast diario.
modulos/resultados.py (o similar): funciones para cargar observados e hindcasts, calcular métricas (RMSE, MAE, BIAS, NSE) y graficar resultados.

Requisitos
Python: 3.10 o superior.
Librerías principales de Python:
pandas
numpy
matplotlib
statsmodels
python-dotenv
Cliente a5client (para acceder a la API A5).

Instalación
Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

Crear y activar un entorno virtual (recomendado)
python -m venv .venv
# En Linux/Mac
source .venv/bin/activate
# En Windows
.venv\Scripts\activate

Instalar dependencias
pip install pandas numpy matplotlib statsmodels python-dotenv a5client openpyxl

Configuración de credenciales A5
El script principal lee las credenciales desde variables de entorno usando python-dotenv.

Crear un archivo .env en la raíz del proyecto con el siguiente contenido:

A5_URL = "https://alerta.ina.gob.ar/a6"
A5_TOKEN=tu-token-de-a5

Cómo correr el flujo principal (hindcast)
El script run_hindcast.py se encarga de todo el pipeline:

Carga las variables de entorno (A5_URL, A5_TOKEN).
Define el diccionario de estaciones que participan del modelo (Misión La Paz, Villa Montes, Puente Aruma) con sus IDs de serie en A5.
Descarga, limpia y unifica las series en resultados/series_nivel_union_h.csv.
Evalúa cada estación aguas arriba de forma individual con un modelo lineal con lag y arma un ranking por R².
Selecciona la estación principal (primary) y secundaria (secondary).
Corre el hindcast diario entre las fechas definidas y guarda los resultados en la carpeta resultados.

Para ejecutarlo:
python run_hindcast.py

Después de correrlo deberías tener:
logs/hindcast_logs.txt con toda la traza de logging.
resultados/series_nivel_union_h.csv
resultados/resumen_series_niveles_h.xlsx
resultados/hindcast_diario_72h_1a3m_single_stations.csv

Cómo generar los gráficos de evaluación
Una vez generado el archivo de hindcast, podés correr:
python plot_hindcast.py

Este script:
Carga la serie observada de Misión La Paz desde resultados/series_nivel_union_h.csv.
Carga el hindcast desde resultados/hindcast_diario_72h_1a3m_single_stations.csv.
Genera todos los graficos.

Notas sobre limpieza y parámetros por estación
La lógica de limpieza está centralizada en modulos/limpieza_series.py y se controla mediante el diccionario PARAMS_LIMPIEZA, donde se pueden definir, para cada estación: ventanas a eliminar, corrimientos verticales, rangos de outliers y parámetros para detección de saltos.

Para agregar o ajustar parámetros de una estación:
Editar PARAMS_LIMPIEZA y ajustar:
ventanas: períodos a eliminar.
corrimientos: ajustes de nivel por ventanas.
outliers: rango mínimo y máximo aceptable.
saltos: ventana y umbral para detectar saltos.

Logging
Todo el logging del flujo principal se envía solo a archivo, a logs/hindcast_logs.txt.
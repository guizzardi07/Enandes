# Pronóstico hidrológico del río Pilcomayo en Misión la Paz

Automatización e implementación de procedimientos operativos

Este repositorio implementa un flujo completo para:

* Descargar series de nivel desde la API A5.
* Limpiar y homogeneizar las series.
* Ajustar modelos lineales con retardo (lag) entre estaciones aguas arriba y Misión La Paz, seleccionar la mejor estación predictora y generar hindcasts diarios.
* Evaluar el desempeño de los pronósticos y generar gráficos de diagnóstico.

## Estructura principal del proyecto

Los scripts más importantes son:

### **`run_hindcast.py`**
Descarga y limpia las series de nivel desde A5.
Construye el archivo unificado resultados/series_nivel_union_h.csv y un resumen en resultados/resumen_series_niveles_h.xlsx.
Evalúa cada estación aguas arriba de forma individual para elegir la mejor y la secundaria.
Corre el hindcast diario y guarda los pronósticos en la carpeta resultados.

### **`plot_hindcast.py`**
Carga la serie observada de Misión La Paz desde series_nivel_union_h.csv.
Carga el archivo de hindcast generado por run_hindcast.py.
Genera gráficos y métricas por lead de pronóstico.

### **`modulos/limpieza_series.py`** 
Funciones para limpieza de series (eliminación de ventanas, corrimientos, outliers, saltos, gráficos rápidos) y diccionario de parámetros por estación.

### **`modulos/series.py`**
Flujo de descarga, análisis de frecuencia, limpieza y unificación de las series en un único DataFrame.

### **`modulos/hindcast.py`**
Evaluación de estaciones individuales, calibración ventana a ventana y generación del hindcast diario.

### **`modulos/resultados.py`**
funciones para cargar observados e hindcasts, calcular métricas (RMSE, MAE, BIAS, NSE) y graficar resultados.

## Requisitos
* Python **3.10+**
* Librerías principales:

  ```
  pandas
  numpy
  matplotlib
  statsmodels
  python-dotenv
  a5client
  openpyxl
  ```

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

### 2. Crear un entorno virtual (opcional pero recomendado)

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install pandas numpy matplotlib statsmodels python-dotenv a5client openpyxl
```


## Configurar credenciales A5
El sistema utiliza `python-dotenv` para leer parámetros desde un archivo `.env`.

Crear `.env` en la raíz del proyecto:

```env
A5_URL="https://alerta.ina.gob.ar/a6"
A5_TOKEN="tu-token"
```

## Cómo ejecutar el hindcast

El pipeline completo se ejecuta con:

```bash
python run_hindcast.py
```

El script:

1. Descarga series desde A5.
2. Limpia y unifica las series en
   `resultados/series_nivel_union_h.csv`
3. Evalúa estaciones predictoras.
4. Corre el hindcast diario para ventanas móviles anual + 3 meses.
5. Guarda resultados en:
   `resultados`
6. Guarda el log completo en:
   `logs/hindcast_logs.txt`

---

## Cómo generar gráficos de evaluación

Después de ejecutar el hindcast:

```bash
python plot_hindcast.py
```

Genera todos los graficos.


## Limpieza de series

Los parámetros de limpieza se controlan desde un diccionario por estación:

* Ventanas a eliminar
* Corrimientos verticales
* Rango de outliers
* Parámetros para detección de saltos
* Gráficos rápidos para inspección

Cada estación se limpia de acuerdo a su sección en `PARAMS_LIMPIEZA`.

## Logging
Todo el logging del flujo principal se envía solo a archivo, a logs/hindcast_logs.txt.
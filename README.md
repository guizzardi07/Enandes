# Tablero de Control – Pilcomayo

## ¿Qué hace esta aplicación?

Este tablero permite construir un **pronóstico operativo de niveles** para la estación
**Misión La Paz**, a partir de estaciones aguas arriba, mediante un flujo reproducible que incluye:

- Descarga de series horarias desde la API A5 (INA)
- Limpieza automática de series (outliers, saltos, huecos cortos)
- Unificación de series a paso **horario (1H)**
- Estimación del **lag temporal** entre estaciones
- Ajuste de modelos lineales nivel–nivel con lag
- Pronóstico operativo mostrando:
  - última semana observada
  - ajuste reciente
  - pronóstico futuro

## Instalación

### 0. Instalar Python

La aplicación requiere **Python 3.10 o superior**.

1. Descargar Python desde el sitio oficial:
   👉 [https://www.python.org/downloads/](https://www.python.org/downloads/)

2. Durante la instalación:

   * ✔️ Marcar **“Add Python to PATH”**
   * ✔️ Usar la instalación estándar

3. Verificar la instalación:

```bash
python --version
```

Debería devolver algo como:

```
Python 3.10.x
```

---

### 1. Clonar el repositorio

```bash
git clone https://github.com/guizzardi07/Enandes.git
cd Enandes
```

---

### 2. Crear un entorno virtual (opcional pero recomendado)

```bash
python -m venv .venv
```

Activar el entorno:

```bash
# Linux / Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Una vez activado, el prompt debería indicar que estás dentro del entorno virtual.

---

### 3. Instalar dependencias

Primero, actualizar `pip`:

```bash
python -m pip install --upgrade pip
```

Luego instalar las dependencias necesarias:

```bash
pip install pandas numpy matplotlib statsmodels python-dotenv a5-client openpyxl streamlit
```

---

## Configurar credenciales A5

El sistema utiliza **python-dotenv** para leer parámetros desde un archivo `.env`.

1. Crear un archivo `.env` en la raíz del proyecto.
2. Agregar el siguiente contenido:

```env
A5_URL="https://alerta.ina.gob.ar/a6"
A5_TOKEN="tu-token"
```
---

## Ejecutar la aplicación

### Windows

1. Ir a la carpeta del proyecto.
2. Hacer doble clic en el archivo:

```
iniciar_tablero.bat
```

El script:

* Activa el entorno virtual
* Ejecuta la aplicación
* Abre el tablero automáticamente en el navegador

---

### Linux / Mac

1. Abrir una terminal en la carpeta del proyecto.
2. Ejecutar:

```bash
./iniciar_tablero.sh
```

Si es la primera vez y aparece un error de permisos:

```bash
chmod +x iniciar_tablero.sh
```

Luego volver a ejecutar el script.

---

# Guía de uso

## Estructura general de la app

La app se usa en **3 pasos secuenciales**:

1. Descarga y limpieza de series
2. Estimación de lags
3. Ajuste, diagnóstico y pronóstico operativo

---

## Configuración inicial

En la barra lateral:

1. Ingresar el **A5_TOKEN**
2. La URL del servicio A5 es fija

---

## Paso 1 — Descargar y limpiar series

1. Seleccionar el período **Desde / Hasta**
2. Presionar **“Descargar + limpiar (construir df_union)”**

La aplicación:

* Descarga las series desde A5
* Aplica limpieza automática
* Remuestrea a paso horario (1H)

### Resultados

* Vista previa de las series
* Gráfico temporal sin aplicar lag
* Archivos guardados automáticamente en la carpeta:

  ```
  resultados/
  ```

### Descargas

* **Descargar CSV (series limpias)**
  Guarda el archivo en la **carpeta Descargas del navegador**

---

## Paso 2 — Estimar lag por estación

1. Definir la ventana temporal para estimar el lag
2. Ajustar:
   * `max_lag`: lag máximo a evaluar (en horas)
   * `ini_lag`: lag mínimo
3. Presionar **“Estimar lag óptimo”**

### Resultados

* Tabla con el lag estimado por estación
* Posibilidad de editar manualmente el `lag_manual`
* Gráfico con las series alineadas según el lag

---

## Paso 3 — Ajuste, diagnóstico y pronóstico

### Selección de estaciones

* Elegir hasta **2 estaciones upstream**
* Se muestra el **lag adoptado** para cada una

---

### Ventana de ajuste (calibración)

* Definir el período que se usará para **ajustar los modelos**
* Esta ventana se utiliza tanto para:

  * el diagnóstico
  * el pronóstico operativo

---

### Diagnóstico

Permite:

* Ver métricas del ajuste (R², n, coeficientes)
* Gráfico temporal de ajuste
* Scatter Observado vs Ajustado

---

### Operativo — Última semana + pronóstico

La app:

* Ajusta los modelos usando la ventana de calibración
* Muestra:

  * Observado de la última semana
  * Ajuste reciente
  * Pronóstico futuro
* Marca el instante de **emisión del pronóstico**

### Resultados

* Tabla **Resumen modelos**
* Gráfico final operativo
* Botón **Descargar CSV** con la serie final

---

## Notas importantes

* El paso temporal es **horario (1H)** y no es configurable
* Los lags se interpretan siempre en **horas**
* Los botones de descarga guardan archivos en la carpeta **Descargas**
* Los archivos generados automáticamente se guardan en `resultados/`

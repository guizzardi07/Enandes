#!/usr/bin/env bash

# chmod +x iniciar_tablero.sh

# Cortar si hay error
set -e

# Ir a la carpeta del script (muy importante)
cd "$(dirname "$0")"

# Activar entorno virtual si existe
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Verificar que streamlit esté disponible
if ! command -v streamlit &> /dev/null; then
    echo "ERROR: streamlit no está instalado o no está en el PATH"
    read -p "Presione ENTER para salir"
    exit 1
fi

# Lanzar Streamlit
streamlit run app_streamlit_v4.py

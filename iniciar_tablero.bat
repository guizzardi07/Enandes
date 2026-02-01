@echo off
cd /d "%~dp0"

REM Activar entorno virtual (si existe)
IF EXIST ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Lanzar Streamlit
streamlit run app_streamlit_v1.py

pause

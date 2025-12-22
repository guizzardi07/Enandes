# modulos/io_utils.py
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import pandas as pd

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    """Convierte DF a bytes Excel (para st.download_button)."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name=sheet_name)
    return buffer.getvalue()


def df_to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    """Convierte DF a bytes CSV UTF-8 (para st.download_button)."""
    return df.to_csv(index=index).encode("utf-8")


def safe_filename(name: str, default: str) -> str:
    """Asegura nombre de archivo (sin path), con fallback."""
    try:
        n = Path(name).name.strip()
        return n if n else default
    except Exception:
        return default

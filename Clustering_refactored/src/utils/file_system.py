import os


def ensure_directory_exists(directory_path: str) -> None:
    """
    Comprueba si un directorio existe en la ruta especificada. Si no existe,
    crea toda la jerarquía de carpetas necesaria.

    Args:
        directory_path (str): Ruta (relativa o absoluta) del directorio a crear.
    """
    os.makedirs(directory_path, exist_ok=True)
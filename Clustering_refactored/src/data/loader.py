import pandas as pd
from typing import List


def load_selected_columns_from_csv(file_path: str, column_names: List[str]) -> pd.DataFrame:
    """
    Lee un archivo CSV desde la ruta indicada y extrae únicamente las columnas solicitadas.

    Args:
        file_path (str): Ruta relativa o absoluta al archivo CSV.
        column_names (List[str]): Lista con los nombres exactos de las columnas a extraer.

    Returns:
        pd.DataFrame: Un nuevo DataFrame aislado solo con las columnas especificadas.
    """
    return pd.read_csv(file_path)[column_names].copy()


def clean_and_cast_numeric_column(dataframe: pd.DataFrame, numeric_column_name: str) -> pd.DataFrame:
    """
    Limpia una columna que debería ser numérica pero contiene texto o datos corruptos.
    Fuerza la conversión a número, elimina las filas que no se pudieron convertir y
    finalmente castea la columna a enteros.

    Args:
        dataframe (pd.DataFrame): El DataFrame original.
        numeric_column_name (str): El nombre de la columna a limpiar (ej. 'score').

    Returns:
        pd.DataFrame: Un DataFrame limpio y sin valores nulos en la columna objetivo.
    """
    # 1. Forzamos la conversión. Lo que sea texto puro (ej. comas mal escapadas) se vuelve NaN (Not a Number)
    dataframe[numeric_column_name] = pd.to_numeric(dataframe[numeric_column_name], errors='coerce')

    # 2. Eliminamos las filas que resultaron en NaN para dejar solo datos válidos
    cleaned_dataframe = dataframe.dropna(subset=[numeric_column_name]).copy()

    # 3. Aseguramos que el tipo de dato final sea entero (int)
    cleaned_dataframe[numeric_column_name] = cleaned_dataframe[numeric_column_name].astype(int)

    return cleaned_dataframe


def drop_nulls_and_cast_to_string(dataframe: pd.DataFrame, text_column_name: str) -> pd.DataFrame:
    """
    Elimina las filas donde la columna de texto está vacía y asegura que
    los datos restantes sean tratados estrictamente como cadenas de texto (strings).

    Args:
        dataframe (pd.DataFrame): El DataFrame original.
        text_column_name (str): El nombre de la columna de texto (ej. 'review').

    Returns:
        pd.DataFrame: Un DataFrame sin textos nulos y con la columna formateada a string.
    """
    # 1. Eliminamos las filas donde no hay texto
    cleaned_dataframe = dataframe.dropna(subset=[text_column_name]).copy()

    # 2. Forzamos que todo el contenido sea de tipo string
    cleaned_dataframe[text_column_name] = cleaned_dataframe[text_column_name].astype(str)

    return cleaned_dataframe
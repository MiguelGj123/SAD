import re
import pandas as pd


def classify_sentiment_by_score(score: int, min_positive_score: int, exact_neutral_score: int) -> str:
    """
    Clasifica una puntuación numérica en una categoría de sentimiento ('positivo', 'neutro', 'negativo').
    Los umbrales se pasan por parámetro para evitar el hardcoding.

    Args:
        score (int): La puntuación de la reseña a evaluar.
        min_positive_score (int): Puntuación mínima para considerar la reseña como positiva (ej. 4).
        exact_neutral_score (int): Puntuación exacta para considerar la reseña como neutra (ej. 3).

    Returns:
        str: El sentimiento clasificado ('positivo', 'neutro' o 'negativo').
    """
    if score >= min_positive_score:
        return 'positivo'
    elif score == exact_neutral_score:
        return 'neutro'
    else:
        return 'negativo'


def apply_basic_text_cleaning(raw_text: str) -> str:
    """
    Aplica una limpieza estándar multilingüe a una cadena de texto.
    Convierte a minúsculas, elimina URLs, puntuación, números y espacios redundantes.

    Args:
        raw_text (str): El texto original de la reseña.

    Returns:
        str: El texto limpio y normalizado.
    """
    text = raw_text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Eliminar URLs
    text = re.sub(r'[^\w\s]', ' ', text)  # Eliminar signos de puntuación
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios múltiples
    return text


def filter_dataframe_by_column_value(dataframe: pd.DataFrame, column_name: str, target_value: str) -> pd.DataFrame:
    """
    Filtra un DataFrame para quedarse únicamente con las filas que coincidan
    con un valor específico en una columna dada, y resetea el índice.
    Es una función genérica (ya no está atada a la palabra "sentimiento").

    Args:
        dataframe (pd.DataFrame): El DataFrame a filtrar.
        column_name (str): La columna sobre la que se aplicará el filtro.
        target_value (str): El valor exacto que deben tener las filas para conservarse.

    Returns:
        pd.DataFrame: Un nuevo DataFrame filtrado y con el índice reseteado.
    """
    filtered_df = dataframe[dataframe[column_name] == target_value].copy()
    return filtered_df.reset_index(drop=True)
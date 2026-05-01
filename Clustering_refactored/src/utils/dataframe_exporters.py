import pandas as pd
import numpy as np
import csv
from typing import List, Dict


def add_top_words_label(df: pd.DataFrame, cluster_col: str, label_col: str, words_dict: Dict[int, List[str]],
                        top_n: int = 5, separator: str = ' | ') -> pd.DataFrame:
    """
    Crea una nueva columna en el DataFrame uniendo las palabras más representativas
    del cluster al que pertenece cada fila. Muy útil para los tooltips en Tableau.

    Args:
        df (pd.DataFrame): DataFrame original.
        cluster_col (str): Nombre de la columna que contiene el ID del cluster/topic.
        label_col (str): Nombre de la nueva columna a crear.
        words_dict (Dict[int, List[str]]): Diccionario con las palabras clave por cluster.
        top_n (int): Número máximo de palabras a mostrar en la etiqueta.
        separator (str): Cadena de texto usada para unir las palabras.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con la columna de etiquetas añadida.
    """
    df_result = df.copy()

    def generate_label(row):
        cluster_id = row[cluster_col]
        return separator.join(words_dict[cluster_id][:top_n])

    df_result[label_col] = df_result.apply(generate_label, axis=1)
    return df_result


def add_topic_probabilities(df: pd.DataFrame, doc_topic_matrix: np.ndarray, n_topics: int,
                            col_prefix: str = 'prob_topic_', decimals: int = 4) -> pd.DataFrame:
    """
    Añade las probabilidades del Soft Clustering (LDA) como columnas individuales en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.
        doc_topic_matrix (np.ndarray): Matriz de distribución de probabilidad (documentos x topics).
        n_topics (int): Número total de topics.
        col_prefix (str): Prefijo para los nombres de las nuevas columnas.
        decimals (int): Número de decimales para redondear las probabilidades.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con una columna extra por cada topic.
    """
    df_result = df.copy()
    for t in range(n_topics):
        df_result[f'{col_prefix}{t}'] = doc_topic_matrix[:, t].round(decimals)
    return df_result


def prepare_neutral_reviews(df: pd.DataFrame, target_cluster_col: str, target_label_col: str,
                            neutral_cluster_id: int = -1, neutral_label: str = 'neutro') -> pd.DataFrame:
    """
    Formatea las reseñas neutras para que tengan la misma estructura de columnas
    que las positivas y negativas, asignándoles un ID y etiqueta por defecto.

    Args:
        df (pd.DataFrame): DataFrame con reseñas exclusivamente neutras.
        target_cluster_col (str): Nombre de la columna del ID de cluster/topic.
        target_label_col (str): Nombre de la columna de las palabras clave.
        neutral_cluster_id (int): ID numérico que se le dará a los neutros (ej. -1).
        neutral_label (str): Etiqueta de texto que se les dará (ej. 'neutro').

    Returns:
        pd.DataFrame: DataFrame formateado listo para concatenar.
    """
    df_result = df.copy()
    df_result[target_cluster_col] = neutral_cluster_id
    df_result[target_label_col] = neutral_label
    return df_result


def concat_and_export_csv(dataframes: List[pd.DataFrame], filepath: str, csv_separator: str = ',',
                          fill_na_prefix: str = None, fill_value: float = 0.0) -> None:
    """
    Concatena una lista de DataFrames y exporta el resultado a un archivo CSV.
    Permite rellenar valores nulos en un subconjunto de columnas específico.

    Args:
        dataframes (List[pd.DataFrame]): Lista de DataFrames a unir.
        filepath (str): Ruta completa donde se guardará el CSV.
        csv_separator (str): Carácter separador para el CSV (por defecto '~').
        fill_na_prefix (str): Si se especifica, rellenará los NaN de las columnas que empiecen por este prefijo.
        fill_value (float): Valor para reemplazar los NaN en las columnas del prefijo.
    """
    df_resultado = pd.concat(dataframes, ignore_index=True)

    if fill_na_prefix:
        # Busca las columnas que empiecen por el prefijo (útil para que los neutros tengan prob_topic_X = 0)
        target_cols = [c for c in df_resultado.columns if c.startswith(fill_na_prefix)]
        df_resultado[target_cols] = df_resultado[target_cols].fillna(fill_value)

    df_resultado.to_csv(filepath, sep= csv_separator ,  index=False,
                        quoting=csv.QUOTE_NONNUMERIC,
                        quotechar='"',
                        escapechar='\\')
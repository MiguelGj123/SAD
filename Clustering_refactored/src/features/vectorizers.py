from typing import List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


def build_tfidf_vectorizer(max_features: int, min_df: int, max_df: float,
                           custom_stopwords: List[str],
                           ngram_range: Tuple[int, int] = (1, 1)) -> TfidfVectorizer:
    """
    Construye y configura un modelo TF-IDF (Term Frequency - Inverse Document Frequency).
    Ideal para Hard Clustering (K-Means) ya que penaliza las palabras muy frecuentes.

    Args:
        max_features (int): Límite del tamaño del vocabulario.
        min_df (int): Frecuencia mínima de aparición de una palabra (en documentos).
        max_df (float): Frecuencia máxima de aparición de una palabra (porcentaje de documentos).
        custom_stopwords (List[str]): Lista de palabras vacías a ignorar.

    Returns:
        TfidfVectorizer: La instancia del vectorizador configurada y lista para usarse.
    """
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        stop_words=custom_stopwords,
        ngram_range=ngram_range
    )


def build_count_vectorizer(max_features: int, min_df: int, max_df: float,
                           custom_stopwords: List[str],
                           ngram_range: Tuple[int, int] = (1, 1)) -> CountVectorizer:
    """
    Construye y configura un modelo Bag of Words (frecuencias absolutas).
    Requisito estricto para Soft Clustering (LDA), ya que este algoritmo se basa en conteos.

    Args:
        max_features (int): Límite del tamaño del vocabulario.
        min_df (int): Frecuencia mínima de aparición de una palabra (en documentos).
        max_df (float): Frecuencia máxima de aparición de una palabra (porcentaje de documentos).
        custom_stopwords (List[str]): Lista de palabras vacías a ignorar.

    Returns:
        CountVectorizer: La instancia del vectorizador configurada y lista para usarse.
    """
    return CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=custom_stopwords,
        ngram_range=ngram_range
    )


def fit_and_transform_texts(vectorizer: Any, text_corpus: List[str]) -> Tuple[Any, Any, np.ndarray]:
    """
    Entrena el vectorizador con los textos proporcionados y los transforma en una matriz matemática.
    Extrae además el vocabulario generado para su uso posterior en la interpretación de clusters/topics.

    Args:
        vectorizer (Any): Instancia de un vectorizador (TfidfVectorizer o CountVectorizer).
        text_corpus (List[str]): Lista de textos limpios a procesar.

    Returns:
        Tuple[Any, Any, np.ndarray]:
            - matrix: La matriz matemática generada (dispersa).
            - vectorizer: El modelo vectorizador ya entrenado.
            - vocab: Un array con las palabras (nombres de las características) correspondientes a las columnas.
    """
    matrix = vectorizer.fit_transform(text_corpus)
    vocab = vectorizer.get_feature_names_out()

    return matrix, vectorizer, vocab
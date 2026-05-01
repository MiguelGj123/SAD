import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Tuple, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')


def evaluate_lda_coherence_gensim(X: Any, tokenized_texts: List[List[str]], vocab: np.ndarray, k_range: List[int],
                                  random_state: Optional[int], processes: int, max_iter: int = 15,
                                  top_words_for_eval: int = 20) -> List[float]:
    """
    Calcula la métrica de coherencia c_v utilizando la librería gensim. Esta métrica es la
    recomendada en la literatura académica por ser más robusta y alineada con la interpretación
    humana que la log-verosimilitud (log-likelihood).

    Args:
        X (Any): Matriz documento-término vectorizada sobre la que se ajustará el modelo LDA.
        tokenized_texts (List[List[str]]): Lista de documentos, donde cada documento es una lista de tokens (palabras).
        vocab (np.ndarray): Un array de NumPy que contiene el vocabulario completo asociado a las columnas de X.
        k_range (List[int]): Lista de enteros que representan el número de tópicos (K) a evaluar.
        random_state (Optional[int]): Semilla aleatoria para garantizar la reproducibilidad.
        processes (int): Número de procesos (workers) a utilizar para paralelizar el cálculo en gensim.
        max_iter (int, opcional): Número máximo de iteraciones para el algoritmo LDA. Por defecto 15.
        top_words_for_eval (int, opcional): Número de palabras principales por tópico a considerar para evaluar la coherencia. Por defecto 20.

    Returns:
        List[float]: Una lista con las puntuaciones de coherencia c_v calculadas para cada valor de K.
    """
    # 1. Importamos internamente las clases necesarias de gensim para evitar dependencias globales estrictas si no se usa esta función
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary

    # 2. Creamos un diccionario de gensim a partir de los textos tokenizados. Esto mapea cada palabra única a un ID.
    diccionario = Dictionary(tokenized_texts)

    # 3. Inicializamos una lista vacía para ir almacenando la puntuación de coherencia de cada K
    scores = []

    # 4. Iteramos sobre cada número de tópicos (K) proporcionado en el rango
    for k in k_range:
        # 5. Instanciamos el modelo LDA de scikit-learn indicando número de componentes, semilla y límite de iteraciones
        lda = LatentDirichletAllocation(n_components=k, random_state=random_state, max_iter=max_iter)

        # 6. Ajustamos el modelo a nuestra matriz de características X
        lda.fit(X)

        # 7. Convertimos los tópicos generados por sklearn a un formato compatible con gensim (una lista de listas de palabras)
        topics_words = []
        for topic_idx in range(k):
            # argsort() ordena los índices por su peso; [::-1] los invierte (de mayor a menor); [:top_words_for_eval] recorta los top N
            top_idx = lda.components_[topic_idx].argsort()[::-1][:top_words_for_eval]
            all_words = [vocab[i] for i in top_idx]
            valid_words = [w for w in all_words if w in diccionario.token2id]

            if not valid_words:
                return None

                # Mapeamos esos índices al vocabulario real y añadimos la lista de palabras del tópico a nuestra lista principal
            topics_words.append([vocab[i] for i in top_idx])

        # 8. Instanciamos el modelo de coherencia de gensim pasándole los tópicos, los textos reales, el diccionario y la métrica deseada ('c_v')
        cm_model = CoherenceModel(
            topics=topics_words, texts=tokenized_texts,
            dictionary=diccionario, coherence='c_v', processes=processes
        )

        # 9. Calculamos la coherencia global del modelo actual y la añadimos a la lista de resultados
        scores.append(cm_model.get_coherence())

    return scores


def evaluate_lda_loglikelihood(X: Any, k_range: List[int], random_state: Optional[int], max_iter: int = 20) -> List[
    float]:
    """
    Método de respaldo (fallback) que calcula la log-verosimilitud (log-likelihood) utilizando
    scikit-learn. Se utiliza generalmente si gensim no está instalado. En esta métrica,
    un valor mayor (menos negativo) indica un mejor ajuste.

    Args:
        X (Any): Matriz documento-término vectorizada.
        k_range (List[int]): Lista de enteros que representan los valores de K a evaluar.
        random_state (Optional[int]): Semilla aleatoria para la reproducibilidad.
        max_iter (int, opcional): Número máximo de iteraciones. Por defecto 20.

    Returns:
        List[float]: Una lista con las puntuaciones de log-verosimilitud para cada K.
    """
    # 1. Inicializamos la lista de puntuaciones vacía
    scores = []

    # 2. Iteramos sobre cada número de tópicos (K) a evaluar
    for k in k_range:
        # 3. Instanciamos el modelo LDA configurando n_jobs=-1 para usar todos los núcleos disponibles del procesador
        lda = LatentDirichletAllocation(n_components=k, random_state=random_state, max_iter=max_iter, n_jobs=-1)

        # 4. Ajustamos el modelo a los datos X
        lda.fit(X)

        # 5. Calculamos la puntuación (log-likelihood aproximado de los datos dados el modelo) y la guardamos
        scores.append(lda.score(X))

    return scores


def get_optimal_topics_from_scores(scores: List[float], k_range: List[int]) -> int:
    """
    Determina el número óptimo de tópicos (K) identificando la puntuación máxima
    en la lista de métricas de evaluación calculadas previamente.

    Args:
        scores (List[float]): Lista con las puntuaciones de las métricas (coherencia o log-likelihood).
        k_range (List[int]): Lista con los valores de K evaluados correspondientes a las puntuaciones.

    Returns:
        int: El número de tópicos considerado como óptimo.
    """
    # 1. Utilizamos np.argmax para encontrar el índice del valor más alto en la lista de puntuaciones,
    # y devolvemos el valor correspondiente en la misma posición dentro de la lista k_range
    return k_range[np.argmax(scores)]


def fit_lda_model(X: Any, k: int, random_state: Optional[int], max_iter: int = 50) -> Tuple[
    LatentDirichletAllocation, np.ndarray, np.ndarray]:
    """
    Entrena el modelo Latent Dirichlet Allocation (LDA) definitivo utilizando el número
    óptimo de tópicos y devuelve el modelo junto con las matrices resultantes para su análisis.

    Args:
        X (Any): Matriz documento-término vectorizada sobre la que se entrenará el modelo final.
        k (int): Número de tópicos óptimo.
        random_state (Optional[int]): Semilla aleatoria para la reproducibilidad.
        max_iter (int, opcional): Número máximo de iteraciones del algoritmo. Por defecto 50.

    Returns:
        Tuple[LatentDirichletAllocation, np.ndarray, np.ndarray]: Una tupla que contiene:
        - El objeto del modelo LDA ya entrenado.
        - La matriz documento-tópico (probabilidad de que un documento pertenezca a un tópico).
        - Un array unidimensional con el índice del tópico dominante para cada documento.
    """
    # 1. Instanciamos el modelo LDA con los parámetros definitivos.
    # Utilizamos learning_method='online' que suele ser más rápido para conjuntos de datos grandes.
    lda_final = LatentDirichletAllocation(
        n_components=k, random_state=random_state, max_iter=max_iter, learning_method='online'
    )

    # 2. Ajustamos el modelo a los datos y simultáneamente obtenemos la matriz documento-tópico,
    # la cual indica el peso/probabilidad de cada tópico dentro de cada documento.
    doc_topic_matrix = lda_final.fit_transform(X)

    # 3. Determinamos el tópico dominante (el de mayor probabilidad) para cada documento
    # evaluando a lo largo del eje de los tópicos (axis=1) con argmax.
    topic_dominante = doc_topic_matrix.argmax(axis=1)

    return lda_final, doc_topic_matrix, topic_dominante


def filter_subngrams(words_list: List[str]) -> List[str]:
    # Primero eliminamos duplicados manteniendo el orden
    seen = []
    for w in words_list:
        if w not in seen:
            seen.append(w)

    # Luego filtramos sub-ngramas
    filtered = []
    for word in seen:
        is_redundant = any(
            word != other and word in other
            for other in seen
        )
        if not is_redundant:
            filtered.append(word)
    return filtered


def get_top_words_per_topic(model, vocab, k_final, n_top_words):
    palabras_por_topic = {}

    for i in range(k_final):
        top_idx = model.components_[i].argsort()[::-1][:n_top_words]
        top_words = [vocab[j] for j in top_idx]

        print(f"  Tópico {i} antes de filtrar: {top_words}")  # ← debug temporal
        palabras_por_topic[i] = filter_subngrams(top_words)
        print(f"  Tópico {i} después de filtrar: {palabras_por_topic[i]}")  # ← debug temporal

    return palabras_por_topic
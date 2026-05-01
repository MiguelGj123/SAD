import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Tuple, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')


# --- Modificación en evaluate_lda_coherence_gensim ---
def evaluate_lda_coherence_gensim(X: Any, tokenized_texts: List[List[str]], vocab: np.ndarray, k_range: List[int],
                                  random_state: Optional[int], processes: int, max_iter: int = 15) -> List[float]:
    """
    Calcula la métrica de coherencia c_v utilizando gensim.
    Versión robusta: reconstruye los textos desde la matriz X para soportar n-gramas
    generados por CountVectorizer y evitar listas vacías.
    """
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary

    # 1. Reconstruir los textos reales desde la matriz X
    # Esto asegura que si usas bigramas/trigramas, Gensim los procese correctamente.
    real_texts = []
    X_csr = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)

    for i in range(X_csr.shape[0]):
        row_indices = X_csr.indices[X_csr.indptr[i]:X_csr.indptr[i + 1]]
        doc_words = [vocab[idx] for idx in row_indices]
        real_texts.append(doc_words)

    # 2. Creamos el diccionario de gensim con estos textos alineados
    diccionario = Dictionary(real_texts)

    coherence_scores = []

    for k in k_range:
        # Usamos n_jobs=-1 para acelerar el ajuste
        lda = LatentDirichletAllocation(n_components=k, random_state=random_state, max_iter=max_iter, n_jobs=-1)
        lda.fit(X)

        # Extraer top words para enviarlas a Gensim
        topics_words = []
        for topic_idx in range(k):
            top_idx = lda.components_[topic_idx].argsort()[::-1][:20]
            words = [vocab[i] for i in top_idx]

            # Seguro contra fallos de Gensim: no permitir listas vacías
            if not words:
                words = ['dummy_fallback_word']

            topics_words.append(words)

        # 3. Calcular Coherencia usando los textos reales reconstruidos
        cm = CoherenceModel(topics=topics_words, texts=real_texts,
                            dictionary=diccionario, coherence='c_v', processes=processes)
        coherence_scores.append(cm.get_coherence())

    return coherence_scores


def evaluate_lda_loglikelihood(X: Any, k_range: List[int], random_state: Optional[int],
                               max_iter: int = 20) -> Tuple[List[float], List[float]]:
    """
    Calcula log-likelihood Y perplejidad para cada K.
    Devuelve dos listas en lugar de una.
    """
    scores_ll = []
    scores_perp = []

    for k in k_range:
        lda = LatentDirichletAllocation(
            n_components=k, random_state=random_state,
            max_iter=max_iter, n_jobs=-1
        )
        lda.fit(X)
        scores_ll.append(lda.score(X))
        scores_perp.append(lda.perplexity(X))  # ← nuevo

    return scores_ll, scores_perp


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


def export_lda_dual_metric_plot(x_values: List[int], coherence_values: List[float],
                                perplexity_values: List[float], optimal_x: int,
                                title: str, filepath: str,
                                figsize: Tuple[int, int] = (14, 5), dpi: int = 200) -> None:
    """
    Genera un gráfico comparativo con dos ejes Y:
    - Izquierda: Coherencia (C_v) -> Cuanto más alto, mejor.
    - Derecha: Perplejidad -> Cuanto más bajo, mejor (métrica del codo).
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Eje primario: Coherencia
    color_coh = 'tab:blue'
    ax1.set_xlabel('Número de Topics (K)')
    ax1.set_ylabel('Coherence Score (C_v)', color=color_coh)
    ax1.plot(x_values, coherence_values, 'bo-', linewidth=2, markersize=8, label='Coherencia (C_v)')
    ax1.tick_params(axis='y', labelcolor=color_coh)
    ax1.grid(alpha=0.3)

    # Eje secundario: Perplejidad
    ax2 = ax1.twinx()
    color_perp = 'tab:red'
    ax2.set_ylabel('Perplejidad (Log-Likelihood)', color=color_perp)
    ax2.plot(x_values, perplexity_values, 'rs--', linewidth=2, markersize=8, alpha=0.6, label='Perplejidad')
    ax2.tick_params(axis='y', labelcolor=color_perp)

    # Línea vertical en el óptimo
    ax1.axvline(x=optimal_x, color='green', linestyle=':', linewidth=2,
                label=f'K óptimo detectado: {optimal_x}')

    # Unificar leyendas de ambos ejes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    plt.tight_layout()

    # Guardar y cerrar
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
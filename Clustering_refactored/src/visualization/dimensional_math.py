import numpy as np
from sklearn.manifold import TSNE
from typing import Tuple, Any, Optional


def sample_data_for_tsne(matrix_features: Any, cluster_labels: np.ndarray, max_samples: int,
                         random_state: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Toma una muestra aleatoria de la matriz de características y sus etiquetas correspondientes.
    Esto evita saturar el algoritmo t-SNE, el cual tiene una complejidad computacional elevada
    y no escala bien con conjuntos de datos masivos.

    Args:
        matrix_features (Any): Matriz de características original (puede ser dispersa o densa).
        cluster_labels (np.ndarray): Array con las etiquetas de clusters asignadas a cada fila.
        max_samples (int): Número máximo de filas que queremos en la muestra final.
        random_state (Optional[int]): Semilla para garantizar que la selección sea reproducible.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Una tupla con la matriz reducida (densa) y sus etiquetas.
    """
    # 1. Obtenemos el número total de documentos disponibles en la matriz de entrada
    n_total_docs = matrix_features.shape[0]

    # 2. Definimos el tamaño real de la muestra, asegurándonos de no pedir más datos de los que existen
    n_actual_samples = min(max_samples, n_total_docs)

    # 3. Si el usuario proporcionó una semilla, la configuramos en numpy para que el mapa t-SNE
    # no cambie de forma aleatoria en ejecuciones futuras.
    if random_state is not None:
        np.random.seed(random_state)

    # 4. Generamos una lista de índices aleatorios sin repetición basados en el tamaño total y el deseado
    sample_indices = np.random.choice(n_total_docs, n_actual_samples, replace=False)

    # 5. Extraemos las filas seleccionadas tanto de la matriz de características como del array de etiquetas
    matrix_sampled = matrix_features[sample_indices]
    labels_sampled = cluster_labels[sample_indices]

    # 6. Verificamos si la matriz es dispersa (sparse). De ser así, usamos el método toarray()
    # para convertirla en una matriz densa, ya que t-SNE suele requerir este formato.
    if hasattr(matrix_sampled, "toarray"):
        matrix_sampled = matrix_sampled.toarray()

    return matrix_sampled, labels_sampled


def reduce_dimensions_with_tsne(dense_matrix: np.ndarray, perplexity: int, random_state: Optional[int],
                                max_iter: int = 1000) -> np.ndarray:
    """
    Aplica el algoritmo t-Distributed Stochastic Neighbor Embedding (t-SNE) para proyectar
    datos de alta dimensionalidad en un espacio de 2 dimensiones, facilitando su visualización.

    Args:
        dense_matrix (np.ndarray): Matriz densa con los datos de entrada ya muestreados.
        perplexity (int): Parámetro que balancea la atención entre aspectos locales y globales de los datos.
        random_state (Optional[int]): Semilla aleatoria para la reproducibilidad del embedding.
        max_iter (int, opcional): Número máximo de iteraciones para la optimización. Por defecto 1000.

    Returns:
        np.ndarray: Una matriz de forma (n_samples, 2) con las coordenadas calculadas.
    """
    # 1. Instanciamos el modelo TSNE de scikit-learn.
    # n_components=2 indica que queremos reducir el espacio a dos dimensiones (X, Y).
    tsne_model = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=max_iter
    )

    # 2. Ejecutamos el ajuste y la transformación simultáneamente para obtener las nuevas coordenadas
    return tsne_model.fit_transform(dense_matrix)
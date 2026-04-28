import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict, Any, Optional


def evaluate_kmeans_metrics(X: Any, k_range: List[int], random_state: Optional[int], max_iter: int = 300,
                            max_silhouette_samples: int = 500) -> Tuple[List[float], List[float]]:
    """
    Calcula la inercia (para el método del codo) y la métrica Silhouette para un rango
    de valores K (número de clusters). Útil para determinar el número óptimo de agrupaciones.

    Args:
        X (Any): Los datos de entrada vectorizados o matriz de características sobre los que aplicar KMeans.
        k_range (List[int]): Lista de enteros que representan los valores K a evaluar.
        random_state (Optional[int]): Semilla aleatoria para garantizar la reproducibilidad.
        max_iter (int, opcional): Número máximo de iteraciones permitidas para el algoritmo KMeans. Por defecto 300.
        max_silhouette_samples (int, opcional): Límite de muestras a usar para calcular el Silhouette y evitar problemas de memoria. Por defecto 500.

    Returns:
        Tuple[List[float], List[float]]: Dos listas. La primera contiene las inercias calculadas
        para cada K, y la segunda contiene las puntuaciones Silhouette para cada K.
    """
    # 1. Inicializamos las listas vacías donde iremos guardando los resultados de cada iteración
    inercias = []
    silhouettes = []

    # 2. Iteramos sobre cada número de clusters (K) proporcionado en el rango
    for k in k_range:
        # 3. Instanciamos el modelo KMeans de scikit-learn indicando el número de clusters, la semilla, y el límite de iteraciones
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto', max_iter=max_iter)

        # 4. Ajustamos el modelo a los datos X y extraemos la inercia (suma de distancias al cuadrado al centroide más cercano)
        km.fit(X)
        inercias.append(km.inertia_)

        # 5. La métrica Silhouette requiere al menos 2 clusters. Verificamos si K > 1
        if k > 1:
            # 6. Solo fijamos semilla interna si el usuario definió una global en el config.
            # Le sumamos 'k' para que cada iteración tenga un muestreo distinto pero predecible.
            if random_state is not None:
                np.random.seed(random_state + k)

            # 7. Determinamos el tamaño de la muestra seleccionando el menor valor entre el límite de muestreo y el total de datos
            n_samples = min(max_silhouette_samples, X.shape[0])

            # 8. Extraemos índices aleatorios de los datos (sin reemplazo) para calcular el Silhouette más rápido
            muestra_idx = np.random.choice(X.shape[0], n_samples, replace=False)

            # 9. Filtramos las etiquetas generadas por KMeans quedándonos solo con las correspondientes a la muestra aleatoria
            labels_muestra = km.labels_[muestra_idx]

            # 10. Para que silhouette_score funcione sin error, debe haber más de 1 clase única en la muestra.
            # Le pasamos a la métrica los datos filtrados y sus etiquetas correspondientes.
            if len(np.unique(labels_muestra)) > 1:
                sil = silhouette_score(X[muestra_idx], labels_muestra)
            else:
                sil = 0.0

            silhouettes.append(sil)
        else:
            # 11. Si K es 1, no tiene sentido el Silhouette, así que asignamos un valor de 0.0
            silhouettes.append(0.0)

    return inercias, silhouettes


def get_optimal_k_from_silhouette(silhouettes: List[float], k_range: List[int]) -> int:
    """
    Determina el valor óptimo de K basándose en la puntuación máxima de Silhouette,
    que tiende a ser más robusta matemáticamente que buscar la segunda derivada en el método del codo.
    Por defecto, ignora k=2 si hay suficientes valores, ya que tiende a dar puntuaciones altas artificialmente.

    Args:
        silhouettes (List[float]): Lista con los valores de Silhouette calculados previamente.
        k_range (List[int]): Lista con los valores de K correspondientes a las puntuaciones.

    Returns:
        int: El número de clusters (K) considerado como el más óptimo.
    """
    # 1. Convertimos la lista de puntuaciones a un array de NumPy para aprovechar sus métodos de indexación
    silhouettes_arr = np.array(silhouettes)

    # 2. Buscamos el máximo de silhouette entre k=3 en adelante si hay suficientes elementos.
    # Al cortar el array con [1:] estamos omitiendo el primer índice (normalmente k=2).
    if len(silhouettes_arr) > 2:
        return k_range[np.argmax(silhouettes_arr[1:]) + 1]

    # 3. En caso de que haya muy pocos valores (o estemos evaluando rangos muy pequeños),
    # simplemente devolvemos el valor de K que corresponde al valor máximo global en el array.
    return k_range[np.argmax(silhouettes_arr)]


def fit_kmeans_model(X: Any, k: int, random_state: Optional[int], max_iter: int = 500) -> Tuple[KMeans, np.ndarray]:
    """
    Entrena el modelo K-Means definitivo utilizando el número K óptimo calculado previamente.

    Args:
        X (Any): Los datos vectorizados sobre los que se entrenará el modelo final.
        k (int): El número de clusters óptimo.
        random_state (Optional[int]): Semilla aleatoria para asegurar la reproducibilidad del entrenamiento.
        max_iter (int, opcional): Número máximo de iteraciones. Por defecto 500.

    Returns:
        Tuple[KMeans, np.ndarray]: Una tupla que contiene el objeto del modelo KMeans ya entrenado
        y un array de NumPy con las etiquetas (asignación de cluster) predichas para cada registro de X.
    """
    # 1. Instanciamos el modelo de scikit-learn con el valor k definitivo y los parámetros correspondientes
    model = KMeans(n_clusters=k, random_state=random_state, n_init='auto', max_iter=max_iter)

    # 2. Entrenamos el modelo con los datos X y al mismo tiempo generamos el array de predicciones
    labels = model.fit_predict(X)

    return model, labels


def get_top_words_per_cluster(model: KMeans, vocab: np.ndarray, k_final: int, n_top_words: int) -> Dict[int, List[str]]:
    """
    Extrae las palabras más representativas de cada cluster basándose en los pesos de los centroides
    generados por el modelo KMeans.

    Args:
        model (KMeans): El modelo KMeans de scikit-learn ya entrenado.
        vocab (np.ndarray): Un array de NumPy que contiene el vocabulario (palabras reales asociadas a las columnas).
        k_final (int): El número de clusters que se han generado en el modelo.
        n_top_words (int): El número máximo de palabras top a extraer por cada cluster.

    Returns:
        Dict[int, List[str]]: Un diccionario donde la clave es el identificador del cluster (entero)
        y el valor es una lista de cadenas (strings) representando las palabras más pesadas en ese cluster.
    """
    # 1. Inicializamos un diccionario vacío donde mapearemos el ID del cluster con sus palabras top
    palabras_por_cluster = {}

    # 2. Iteramos a través de todos los clusters identificados (desde 0 hasta k_final - 1)
    for i in range(k_final):
        # 3. Accedemos a los valores del centroide para el cluster 'i' (model.cluster_centers_[i]).
        # argsort() devuelve los índices ordenados de menor a mayor peso.
        # [::-1] invierte el array para que queden de mayor a menor peso.
        # [:n_top_words] recorta el resultado para quedarnos estrictamente con la cantidad pedida.
        top_idx = model.cluster_centers_[i].argsort()[::-1][:n_top_words]

        # 4. Recorremos los índices extraídos, buscamos la palabra real en el array 'vocab'
        # y almacenamos la lista resultante en el diccionario bajo la clave del cluster actual.
        palabras_por_cluster[i] = [vocab[j] for j in top_idx]

    return palabras_por_cluster
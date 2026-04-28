import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple


def export_elbow_and_silhouette_plot(k_range: List[int], inertias: List[float], silhouettes: List[float],
                                     title_suffix: str, filepath: str, figsize: Tuple[int, int] = (14, 5),
                                     dpi: int = 200) -> None:
    """
    Genera una figura comparativa con dos paneles: el método del Codo (Inercia) y la métrica
    Silhouette. Esta visualización es fundamental para elegir el valor K en K-Means.

    Args:
        k_range (List[int]): Lista de valores de K evaluados.
        inertias (List[float]): Valores de Inercia (suma de cuadrados intracluster) para el eje Y.
        silhouettes (List[float]): Puntuaciones Silhouette obtenidas para el eje Y.
        title_suffix (str): Texto adicional para personalizar el título de los gráficos.
        filepath (str): Ruta completa donde se exportará la imagen (formato png recomendado).
        figsize (Tuple[int, int], opcional): Dimensiones de la figura (ancho, alto). Por defecto (14, 5).
        dpi (int, opcional): Resolución de la imagen exportada. Por defecto 200.
    """
    # 1. Creamos la estructura de la figura con una fila y dos columnas utilizando subplots.
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 2. Panel 0 (Izquierdo): Graficamos la Inercia (Método del Codo).
    # 'bo-' indica color azul (blue), marcador circular (o) y línea continua (-).
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de Clusters (K)')
    axes[0].set_ylabel('Inercia (WCSS)')
    axes[0].set_title(f'Gráfico del Codo — {title_suffix}')
    axes[0].grid(alpha=0.3)  # Añadimos rejilla con transparencia para facilitar la lectura

    # 3. Panel 1 (Derecho): Graficamos la Puntuación Silhouette.
    # 'rs-' indica color rojo (red), marcador cuadrado (square) y línea continua (-).
    axes[1].plot(k_range, silhouettes, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Número de Clusters (K)')
    axes[1].set_ylabel('Puntuación Silhouette')
    axes[1].set_title(f'Métrica Silhouette — {title_suffix}')
    axes[1].grid(alpha=0.3)

    # 4. Ajustamos automáticamente el espaciado entre los paneles para evitar solapamientos.
    plt.tight_layout()

    # 5. Guardamos la figura en el disco con la resolución indicada y ajustando los márgenes.
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')

    # 6. Cerramos la figura para liberar la memoria de Matplotlib.
    plt.close()


def export_single_metric_line_plot(x_values: List[int], y_values: List[float], optimal_x: int, x_label: str,
                                   y_label: str, title: str, filepath: str, figsize: Tuple[int, int] = (9, 5),
                                   dpi: int = 200) -> None:
    """
    Genera un gráfico de línea simple para una única métrica (como Coherencia en LDA)
    y destaca visualmente el valor óptimo detectado mediante una línea vertical.

    Args:
        x_values (List[int]): Lista de valores para el eje X (generalmente K).
        y_values (List[float]): Lista de valores para el eje Y (la métrica calculada).
        optimal_x (int): El valor de X donde se encuentra el mejor resultado.
        x_label (str): Etiqueta descriptiva para el eje X.
        y_label (str): Etiqueta descriptiva para el eje Y.
        title (str): Título principal de la gráfica.
        filepath (str): Destino de guardado del archivo de imagen.
        figsize (Tuple[int, int], opcional): Tamaño de la ventana de dibujo.
        dpi (int, opcional): Calidad de la imagen de salida.
    """
    # 1. Inicializamos la figura y el eje (ax) con el tamaño especificado.
    fig, ax = plt.subplots(figsize=figsize)

    # 2. Dibujamos la línea de la métrica en color verde ('go-') con marcadores circulares.
    ax.plot(x_values, y_values, 'go-', linewidth=2, markersize=9)

    # 3. Dibujamos una línea vertical (axvline) punteada de color rojo en la posición del valor óptimo.
    ax.axvline(x=optimal_x, color='red', linestyle='--', label=f'Óptimo detectado: {optimal_x}')

    # 4. Configuramos las etiquetas de los ejes y el título.
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # 5. Mostramos la leyenda para identificar la línea del óptimo y activamos la rejilla.
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. Ajustamos el diseño y procedemos a guardar y cerrar el recurso.
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()


def export_tsne_scatter_plot(coords_2d: np.ndarray, labels: np.ndarray, n_clusters: int,
                             words_dictionary: Dict[int, List[str]], title: str, filepath: str, n_words_legend: int = 3,
                             colormap_name: str = 'tab10', figsize: Tuple[int, int] = (12, 8), dpi: int = 200) -> None:
    """
    Genera un gráfico de dispersión (Scatter Plot) en 2D basado en los resultados de t-SNE.
    Cada punto representa un documento, coloreado por su cluster y con una leyenda descriptiva
    que incluye las palabras más importantes de cada grupo.

    Args:
        coords_2d (np.ndarray): Matriz de coordenadas (X, Y) reducidas.
        labels (np.ndarray): Array de etiquetas enteras que asignan cada punto a un cluster.
        n_clusters (int): Cantidad total de grupos para la gestión de colores.
        words_dictionary (Dict[int, List[str]]): Diccionario con las top words de cada cluster.
        title (str): Título de la visualización.
        filepath (str): Ruta de salida para la imagen.
        n_words_legend (int, opcional): Cuántas palabras clave incluir en la leyenda por cluster.
        colormap_name (str, opcional): Esquema de colores de Matplotlib. Por defecto 'tab10'.
        figsize (Tuple[int, int], opcional): Tamaño de la figura.
        dpi (int, opcional): Resolución del archivo de salida.
    """
    # 1. Obtenemos un mapa de colores y lo resampleamos para asegurar que cada cluster tenga un tono único y distinguible.
    color_map = plt.colormaps.get_cmap(colormap_name).resampled(n_clusters)

    # 2. Creamos la figura y el eje de dibujo.
    fig, ax = plt.subplots(figsize=figsize)

    # 3. Iteramos por cada cluster para dibujar sus puntos de forma independiente y generar su entrada en la leyenda.
    for cluster_id in range(n_clusters):
        # 4. Creamos una máscara booleana para seleccionar solo los registros pertenecientes al cluster actual.
        mask = labels == cluster_id

        # 5. Construimos el texto de la leyenda uniendo las top N palabras separadas por una barra vertical.
        top_words_str = ' | '.join(words_dictionary[cluster_id][:n_words_legend])
        label_text = f"Grupo {cluster_id}: {top_words_str}"

        # 6. Dibujamos el scatter plot para los puntos de este cluster.
        # c: color asignado del mapa de colores.
        # alpha: transparencia de los puntos para visualizar mejor las zonas densas.
        # s: tamaño de los puntos (markersize).
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[color_map(cluster_id)],
            label=label_text,
            alpha=0.6,
            s=25
        )

    # 7. Configuramos el título y ubicamos la leyenda en la esquina superior derecha con fuente pequeña.
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

    # 8. Limpiamos los ejes numéricos. En t-SNE, los valores absolutos carecen de unidad o significado físico;
    # lo relevante es la cercanía relativa entre los puntos.
    ax.set_xticks([])
    ax.set_yticks([])

    # 9. Finalizamos el ajuste de la figura y guardamos el resultado.
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
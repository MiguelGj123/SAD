import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple


def export_elbow_and_silhouette_plot(k_range: List[int], inertias: List[float], silhouettes: List[float],
                                     title_suffix: str, filepath: str, figsize: Tuple[int, int] = (14, 5),
                                     dpi: int = 200) -> None:
    """
    Genera una figura comparativa con dos paneles: el método del Codo (Inercia) y la métrica Silhouette.
    """
    # 1. Creamos la estructura de la figura con una fila y dos columnas utilizando subplots.
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 2. Panel 0 (Izquierdo): Graficamos la Inercia (Método del Codo).
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de Clusters (K)')
    axes[0].set_ylabel('Inercia (WCSS)')
    axes[0].set_title(f'Gráfico del Codo — {title_suffix}')
    axes[0].grid(alpha=0.3)

    # 3. Panel 1 (Derecho): Graficamos la Puntuación Silhouette.
    axes[1].plot(k_range, silhouettes, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Número de Clusters (K)')
    axes[1].set_ylabel('Puntuación Silhouette')
    axes[1].set_title(f'Métrica Silhouette — {title_suffix}')
    axes[1].grid(alpha=0.3)

    # 4. Ajustes finales y guardado
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()


def export_single_metric_line_plot(x_values: List[int], y_values: List[float], optimal_x: int, x_label: str,
                                   y_label: str, title: str, filepath: str, figsize: Tuple[int, int] = (9, 5),
                                   dpi: int = 200) -> None:
    """
    Genera un gráfico de línea simple para una única métrica y destaca el valor óptimo.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_values, y_values, 'go-', linewidth=2, markersize=9)
    ax.axvline(x=optimal_x, color='red', linestyle='--', label=f'Óptimo detectado: {optimal_x}')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()


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

    # Unificar leyendas de ambos ejes para que no se pisen
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()


def export_tsne_scatter_plot(coords_2d: np.ndarray, labels: np.ndarray, n_clusters: int,
                             words_dictionary: Dict[int, List[str]], title: str, filepath: str, n_words_legend: int = 3,
                             colormap_name: str = 'tab10', figsize: Tuple[int, int] = (12, 8), dpi: int = 200) -> None:
    """
    Genera un gráfico de dispersión en 2D basado en los resultados de t-SNE.
    """
    color_map = plt.colormaps.get_cmap(colormap_name).resampled(n_clusters)
    fig, ax = plt.subplots(figsize=figsize)

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        top_words_str = ' | '.join(words_dictionary[cluster_id][:n_words_legend])
        label_text = f"Grupo {cluster_id}: {top_words_str}"

        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[color_map(cluster_id)],
            label=label_text,
            alpha=0.6,
            s=25
        )

    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
"""
HARD CLUSTERING — K-Means sobre TF-IDF
======================================
Cada reseña pertenece a UN único cluster (asignación dura).
Genera:
  - Gráfico del codo para elegir K óptimo
  - Clusters separados para reseñas POSITIVAS y NEGATIVAS
  - Palabras más significativas por cluster (TF-IDF inverso)
  - CSV con resultados para Tableau
"""

import argparse
import json
import os
import pandas as pd
from typing import Dict, Any

# Importaciones de módulos internos del proyecto
from src.data.loader import load_selected_columns_from_csv, clean_and_cast_numeric_column, drop_nulls_and_cast_to_string
from src.data.preprocessor import classify_sentiment_by_score, apply_basic_text_cleaning, filter_dataframe_by_column_value
from src.features.vectorizers import build_tfidf_vectorizer, fit_and_transform_texts
from src.models.kmeans_logic import evaluate_kmeans_metrics, get_optimal_k_from_silhouette, fit_kmeans_model, get_top_words_per_cluster
from src.visualization.dimensional_math import sample_data_for_tsne, reduce_dimensions_with_tsne
from src.visualization.plotters import export_elbow_and_silhouette_plot, export_tsne_scatter_plot
from src.utils.file_system import ensure_directory_exists
from src.utils.dataframe_exporters import add_top_words_label, prepare_neutral_reviews, concat_and_export_csv


def process_hard_sentiment_subset(df_subset: pd.DataFrame, sentiment_label: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de K-Means (Hard Clustering) para un subconjunto específico
    de datos (ej. solo reseñas positivas). Realiza desde la vectorización hasta la visualización t-SNE.

    Args:
        df_subset (pd.DataFrame): DataFrame que contiene el subconjunto de reseñas a procesar.
        sentiment_label (str): Etiqueta del sentimiento (ej. 'positivo', 'negativo') para nombres de archivos.
        cfg (Dict[str, Any]): Diccionario de configuración con hiperparámetros y rutas.

    Returns:
        pd.DataFrame: El DataFrame original con columnas adicionales de cluster_id y palabras clave.
    """
    # 1. Extraemos la semilla aleatoria de la configuración para consistencia en todo el proceso
    seed = cfg["project"]["random_state"]

    print(f"\n  [{sentiment_label.upper()}] Vectorizando con TF-IDF...")

    # 2. Construimos el modelo vectorizador TF-IDF pasando los límites de frecuencia y las stopwords definidas
    vectorizer_model = build_tfidf_vectorizer(
        cfg["features"]["max_features"], cfg["features"]["min_df"],
        cfg["features"]["max_df"], cfg["stopwords"]
    )

    # 3. Ajustamos el vectorizador y transformamos el texto limpio en una matriz numérica X
    X, _, vocab = fit_and_transform_texts(vectorizer_model, df_subset['review_limpio'].tolist())

    print(f"  [{sentiment_label.upper()}] Calculando inercia para K={cfg['model']['k_range']}...")

    # 4. Evaluamos diferentes valores de K para obtener las métricas de Inercia y Silhouette
    inertias, silhouettes = evaluate_kmeans_metrics(
        X, cfg["model"]["k_range"], seed,
        cfg["model"]["eval_max_iter"], cfg["model"]["max_silhouette_samples"]
    )

    # 5. Generamos y exportamos el gráfico comparativo (Codo y Silhouette) para este sentimiento
    export_elbow_and_silhouette_plot(
        cfg["model"]["k_range"], inertias, silhouettes, sentiment_label.capitalize(),
        os.path.join(cfg["project"]["output_dir"], f'codo_{sentiment_label}.png'),
        figsize=tuple(cfg["visualization"]["style"]["figsize_eval"]),
        dpi=cfg["visualization"]["style"]["dpi"]
    )

    # 6. Identificamos automáticamente el K óptimo basándonos en la mejor puntuación de Silhouette
    optimal_k = get_optimal_k_from_silhouette(silhouettes, cfg["model"]["k_range"])
    print(f"  K óptimo por Silhouette: {optimal_k}")

    print(f"  [{sentiment_label.upper()}] Entrenando K-Means final con K={optimal_k}...")

    # 7. Entrenamos el modelo definitivo de K-Means con el K óptimo seleccionado
    model, cluster_labels = fit_kmeans_model(X, optimal_k, seed, cfg["model"]["final_max_iter"])

    # 8. Asignamos los IDs de los clusters al DataFrame y extraemos las palabras más representativas de cada uno
    df_subset['cluster_id'] = cluster_labels
    cluster_keywords = get_top_words_per_cluster(model, vocab, optimal_k, cfg["model"]["n_top_words"])

    print(f"\n  [{sentiment_label.upper()}] Generando visualización t-SNE...")

    # 9. Tomamos una muestra de los datos para no saturar el algoritmo de reducción de dimensiones
    X_sampled, labels_sampled = sample_data_for_tsne(
        X, cluster_labels, cfg["visualization"]["tsne"]["samples"], seed
    )

    # 10. Reducimos la matriz a 2 dimensiones (X, Y) para su representación gráfica
    coords_2d = reduce_dimensions_with_tsne(
        X_sampled, cfg["visualization"]["tsne"]["perplexity"], seed, cfg["visualization"]["tsne"]["max_iter"]
    )

    # 11. Exportamos el gráfico de dispersión con los puntos coloreados por cluster y su leyenda de palabras
    export_tsne_scatter_plot(
        coords_2d, labels_sampled, optimal_k, cluster_keywords,
        f'Mapa de Clusters — Reseñas {sentiment_label.capitalize()} (t-SNE)',
        os.path.join(cfg["project"]["output_dir"], f'tsne_hard_{sentiment_label}.png'),
        n_words_legend=cfg["visualization"]["style"]["n_words_legend"],
        colormap_name=cfg["visualization"]["style"]["colormap"],
        figsize=tuple(cfg["visualization"]["style"]["figsize_tsne"]),
        dpi=cfg["visualization"]["style"]["dpi"]
    )

    # 12. Retornamos el DataFrame procesado añadiendo una columna con las etiquetas de las top palabras
    return add_top_words_label(
        df_subset, 'cluster_id', 'top_palabras', cluster_keywords, cfg["export"]["top_words_in_csv"]
    )


def run_hard_pipeline(config_path: str) -> None:
    """
    Orquestador principal que coordina la carga de datos, limpieza, segmentación por
    sentimiento y ejecución de los procesos de clustering.

    Args:
        config_path (str): Ruta al archivo JSON que contiene toda la configuración del proyecto.
    """
    # 1. Cargamos el archivo de configuración JSON
    with open(config_path, 'r', encoding='utf-8') as file:
        cfg = json.load(file)

    # 2. Aseguramos que la carpeta de salida exista antes de empezar a generar archivos
    ensure_directory_exists(cfg["project"]["output_dir"])

    print("="*55)
    print("PASO 1: Cargando y preparando los datos...")
    print("="*55)

    # 3. Leemos el CSV original cargando solo las columnas de interés
    df = load_selected_columns_from_csv(cfg["data"]["input_csv"], cfg["data"]["columns_to_load"])

    # 4. Mapeamos los nombres de columnas desde la configuración para mayor flexibilidad
    score_col = cfg["data"]["score_col"]
    text_col = cfg["data"]["text_col"]
    sentiment_col = cfg["data"]["sentiment_col"]

    # 5. Realizamos la limpieza técnica: castear a numérico (manejando errores) y asegurar strings en texto
    df = clean_and_cast_numeric_column(df, score_col)
    df = drop_nulls_and_cast_to_string(df, text_col)

    # 6. Clasificamos las reseñas en sentimientos (Positivo, Negativo, Neutro) basándonos en la puntuación
    df[sentiment_col] = df[score_col].apply(
        lambda x: classify_sentiment_by_score(x, cfg["data"]["thresholds"]["min_positive"], cfg["data"]["thresholds"]["exact_neutral"])
    )

    # 7. Aplicamos la limpieza de texto básica (minúsculas, eliminar puntuación, etc.) para el análisis
    df['review_limpio'] = df[text_col].apply(apply_basic_text_cleaning)

    # 8. Dividimos el DataFrame en tres subconjuntos según el sentimiento detectado
    labels = cfg["data"]["labels"]
    df_pos = filter_dataframe_by_column_value(df, sentiment_col, labels["positive"])
    df_neg = filter_dataframe_by_column_value(df, sentiment_col, labels["negative"])
    df_neu = filter_dataframe_by_column_value(df, sentiment_col, labels["neutral"])

    print("\n" + "="*55)
    print("PASO 2: Clustering de reseñas POSITIVAS")
    print("="*55)

    # 9. Procesamos el bloque de reseñas positivas
    df_pos_final = process_hard_sentiment_subset(df_pos, labels["positive"], cfg)

    print("\n" + "="*55)
    print("PASO 3: Clustering de reseñas NEGATIVAS")
    print("="*55)

    # 10. Procesamos el bloque de reseñas negativas
    df_neg_final = process_hard_sentiment_subset(df_neg, labels["negative"], cfg)

    # 11. Para las reseñas neutras no realizamos clustering (suelen ser pocas).
    # Las preparamos asignándoles un ID de cluster neutro por defecto.
    df_neu_final = prepare_neutral_reviews(
        df_neu, 'cluster_id', 'top_palabras', cfg["export"]["neutral_id"], labels["neutral"]
    )

    print("\n" + "="*55)
    print("PASO 4: Guardando resultados...")
    print("="*55)

    # 12. Concatenamos todos los subconjuntos procesados y exportamos el resultado final a CSV
    concat_and_export_csv(
        [df_pos_final, df_neg_final, df_neu_final],
        os.path.join(cfg["project"]["output_dir"], cfg["export"]["filename"]),
        csv_separator=cfg["export"]["separator"]
    )

    print(f"✅ Guardado: '{cfg['export']['filename']}'")
    print("\n🎉 ¡Hard clustering completado!")


if __name__ == "__main__":
    # 1. Configuramos el parser de argumentos de línea de comandos para recibir la ruta del config
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline de Hard Clustering (K-Means).")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON de configuración.")

    # 2. Parseamos los argumentos y lanzamos la ejecución del pipeline
    args = parser.parse_args()
    run_hard_pipeline(args.config)
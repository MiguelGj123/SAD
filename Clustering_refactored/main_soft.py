"""
SOFT CLUSTERING — LDA (Latent Dirichlet Allocation)
=====================================================
Cada reseña tiene una DISTRIBUCIÓN DE PROBABILIDAD sobre los topics.
No pertenece "solo" a uno: puede ser 60% topic_0, 30% topic_2, 10% topic_1.

LDA es el estándar académico para topic modeling en NLP.
"""

import argparse
import json
import os
import pandas as pd
from typing import Dict, Any, Optional

# Importaciones de módulos internos del proyecto para carga, preprocesamiento y modelado
from src.data.loader import load_selected_columns_from_csv, clean_and_cast_numeric_column, drop_nulls_and_cast_to_string
from src.data.preprocessor import classify_sentiment_by_score, apply_basic_text_cleaning, filter_dataframe_by_column_value
from src.features.vectorizers import build_count_vectorizer, fit_and_transform_texts
from src.models.lda_logic import evaluate_lda_coherence_gensim, evaluate_lda_loglikelihood, get_optimal_topics_from_scores, fit_lda_model, get_top_words_per_topic
from src.visualization.dimensional_math import sample_data_for_tsne, reduce_dimensions_with_tsne
from src.visualization.plotters import export_single_metric_line_plot, export_tsne_scatter_plot
from src.visualization.html_exporters import export_interactive_lda_html
from src.utils.file_system import ensure_directory_exists
from src.utils.dataframe_exporters import add_top_words_label, add_topic_probabilities, prepare_neutral_reviews, concat_and_export_csv


def process_soft_sentiment_subset(df_subset: pd.DataFrame, sentiment_label: str, cfg: Dict[str, Any], forced_k: Optional[int] = None) -> pd.DataFrame:
    """
    Ejecuta el pipeline de Soft Clustering (LDA) para un grupo de sentimientos.
    A diferencia de K-Means, LDA no asigna etiquetas rígidas, sino que calcula el peso
    de cada tópico en cada documento.

    Args:
        df_subset (pd.DataFrame): Subconjunto de datos (ej. solo reseñas negativas).
        sentiment_label (str): Nombre del sentimiento para organizar los archivos de salida.
        cfg (Dict[str, Any]): Diccionario con la configuración del proyecto.
        forced_k (Optional[int], opcional): Permite forzar un número de tópicos ignorando la métrica óptima.

    Returns:
        pd.DataFrame: DataFrame con el tópico dominante y las columnas de probabilidad por tópico.
    """
    # 1. Recuperamos la semilla aleatoria y preparamos la lista de textos para el procesamiento
    seed = cfg["project"]["random_state"]
    raw_texts = df_subset['review_limpio'].tolist()

    # 2. LDA requiere una matriz de conteos (Bag of Words), no TF-IDF.
    # Construimos el vectorizador pasando los parámetros de frecuencia de palabras y stopwords.
    print(f"\n  [{sentiment_label.upper()}] Vectorizando con CountVectorizer...")
    vectorizer_model = build_count_vectorizer(
        cfg["features"]["max_features"], cfg["features"]["min_df"],
        cfg["features"]["max_df"], cfg["stopwords"], tuple(cfg["features"].get("ngram_range", [1, 1]))
    )

    # 3. Ajustamos el modelo a los textos y obtenemos la matriz X, el modelo ajustado y el vocabulario
    X, vectorizer_fitted, vocab = fit_and_transform_texts(vectorizer_model, raw_texts)

    print(f"  [{sentiment_label.upper()}] Probando {cfg['model']['n_topics_range']} topics...")

    # 4. Intentamos calcular la coherencia c_v usando Gensim (métrica de calidad de tópicos).
    # Si la librería no está presente, el bloque try-except derivará al cálculo de Log-Likelihood en sklearn.
    try:
        import gensim
        tokenized_texts = [t.split() for t in raw_texts]
        scores = evaluate_lda_coherence_gensim(
            X, tokenized_texts, vocab, cfg["model"]["n_topics_range"],
            seed, cfg["model"]["processes"], cfg["model"]["eval_max_iter"]
        )
        y_label_name = "Coherencia c_v (mayor = mejor)"
    except ImportError:
        # 5. Fallback: Cálculo de la verosimilitud de los datos dado el modelo LDA
        scores = evaluate_lda_loglikelihood(
            X, cfg["model"]["n_topics_range"], seed, cfg["model"]["sklearn_eval_max_iter"]
        )
        y_label_name = "Log-Likelihood (mayor = mejor)"

    # 6. Determinamos el número de tópicos. Se prioriza el valor manual (forced_k) si existe,
    # de lo contrario se busca el máximo en la lista de scores.
    optimal_k = forced_k if forced_k else get_optimal_topics_from_scores(scores, cfg["model"]["n_topics_range"])
    if forced_k:
        print(f"  K elegido manualmente: {optimal_k}")
    else:
        print(f"  K óptimo detectado: {optimal_k}")

    # 7. Exportamos el gráfico de línea de la métrica evaluada (Coherencia o Log-Likelihood)
    export_single_metric_line_plot(
        cfg["model"]["n_topics_range"], scores, optimal_k, "Número de Topics", y_label_name,
        f"Coherencia de Topics — Reseñas {sentiment_label.capitalize()}",
        os.path.join(cfg["project"]["output_dir"], f'coherencia_soft_{sentiment_label}.png'),
        figsize=tuple(cfg["visualization"]["style"]["figsize_eval"]),
        dpi=cfg["visualization"]["style"]["dpi"]
    )

    print(f"  [{sentiment_label.upper()}] Entrenando LDA final con {optimal_k} topics...")

    # 8. Entrenamos el modelo LDA final. Obtenemos la matriz documento-tópico (soft) y el tópico dominante de cada fila.
    lda_model, doc_topic_matrix, dominant_topics = fit_lda_model(X, optimal_k, seed, cfg["model"]["final_max_iter"])

    # 9. Asignamos el tópico dominante al DataFrame y extraemos las palabras clave para cada grupo
    df_subset['topic_dominante'] = dominant_topics
    topic_keywords = get_top_words_per_topic(lda_model, vocab, optimal_k, cfg["model"]["n_top_words"])

    print(f"\n  [{sentiment_label.upper()}] Generando visualización t-SNE...")

    # 10. Muestreamos la matriz de distribución de tópicos para generar el mapa t-SNE
    X_sampled, labels_sampled = sample_data_for_tsne(
        doc_topic_matrix, dominant_topics, cfg["visualization"]["tsne"]["samples"], seed
    )

    # 11. Reducimos las dimensiones a 2D basándonos en las distancias de probabilidad entre tópicos
    coords_2d = reduce_dimensions_with_tsne(
        X_sampled, cfg["visualization"]["tsne"]["perplexity"], seed, cfg["visualization"]["tsne"]["max_iter"]
    )

    # 12. Guardamos la visualización estática t-SNE coloreada por el tópico de mayor peso
    export_tsne_scatter_plot(
        coords_2d, labels_sampled, optimal_k, topic_keywords,
        f'Mapa de Topics (Soft) — Reseñas {sentiment_label.capitalize()} (t-SNE)\n(Coloreado por topic dominante)',
        os.path.join(cfg["project"]["output_dir"], f'tsne_soft_{sentiment_label}.png'),
        n_words_legend=cfg["visualization"]["style"]["n_words_legend"],
        colormap_name=cfg["visualization"]["style"]["colormap"],
        figsize=tuple(cfg["visualization"]["style"]["figsize_tsne"]),
        dpi=cfg["visualization"]["style"]["dpi"]
    )

    # 13. Exportamos el panel interactivo HTML usando pyLDAvis para explorar los tópicos visualmente
    export_interactive_lda_html(
        lda_model, X, vectorizer_fitted,
        os.path.join(cfg["project"]["output_dir"], f'lda_interactivo_{sentiment_label}.html')
    )

    # 14. Preparamos el DataFrame de salida añadiendo las etiquetas de palabras clave
    # y expandiendo las probabilidades de cada tópico en columnas individuales
    df_out = add_top_words_label(
        df_subset, 'topic_dominante', 'top_palabras', topic_keywords, cfg["export"]["top_words_in_csv"]
    )
    return add_topic_probabilities(df_out, doc_topic_matrix, optimal_k)


def run_soft_pipeline(config_path: str) -> None:
    """
    Coordina la ejecución completa del proceso de Soft Clustering (LDA).
    Se encarga de cargar los datos, limpiarlos, dividirlos por sentimiento y exportar los resultados.

    Args:
        config_path (str): Ruta al archivo JSON de configuración.
    """
    # 1. Abrimos y leemos el archivo de configuración del proyecto
    with open(config_path, 'r', encoding='utf-8') as file:
        cfg = json.load(file)

    # 2. Creamos el directorio de salida si no existe previamente
    ensure_directory_exists(cfg["project"]["output_dir"])

    print("="*55)
    print("PASO 1: Cargando y preparando los datos...")
    print("="*55)

    # 3. Cargamos el CSV. El pipeline maneja automáticamente las filas corruptas (comas sin escapar)
    # mediante la conversión forzada a numérico y eliminación de nulos.
    df = load_selected_columns_from_csv(cfg["data"]["input_csv"], cfg["data"]["columns_to_load"])
    df = clean_and_cast_numeric_column(df, cfg["data"]["score_col"])
    df = drop_nulls_and_cast_to_string(df, cfg["data"]["text_col"])

    # 4. Clasificamos las reseñas según su puntuación numérica en sentimientos discretos
    df[cfg["data"]["sentiment_col"]] = df[cfg["data"]["score_col"]].apply(
        lambda x: classify_sentiment_by_score(x, cfg["data"]["thresholds"]["min_positive"], cfg["data"]["thresholds"]["exact_neutral"])
    )

    # 5. Ejecutamos la limpieza de texto (eliminación de caracteres especiales, stop words, etc.)
    df['review_limpio'] = df[cfg["data"]["text_col"]].apply(apply_basic_text_cleaning)

    # 6. Segmentamos el dataset original en tres grupos para analizarlos por separado
    labels = cfg["data"]["labels"]
    df_pos = filter_dataframe_by_column_value(df, cfg["data"]["sentiment_col"], labels["positive"])
    df_neg = filter_dataframe_by_column_value(df, cfg["data"]["sentiment_col"], labels["negative"])
    df_neu = filter_dataframe_by_column_value(df, cfg["data"]["sentiment_col"], labels["neutral"])

    print("\n" + "="*55)
    print("PASO 2: Soft Clustering de reseñas POSITIVAS")
    print("="*55)

    # 7. Ejecutamos el pipeline de LDA para las reseñas positivas
    df_pos_final = process_soft_sentiment_subset(df_pos, labels["positive"], cfg)

    print("\n" + "="*55)
    print("PASO 3: Soft Clustering de reseñas NEGATIVAS")
    print("="*55)

    # 8. Ejecutamos el pipeline de LDA para las reseñas negativas.
    # Aquí pasamos un K forzado si está definido en la configuración para ajustar mejor el resultado.
    df_neg_final = process_soft_sentiment_subset(df_neg, labels["negative"], cfg, forced_k=cfg["model"]["forced_k_neg"])

    # 9. Para los neutros, simplemente preparamos el DataFrame para la exportación sin aplicar LDA
    df_neu_final = prepare_neutral_reviews(
        df_neu, 'topic_dominante', 'top_palabras', cfg["export"]["neutral_id"], labels["neutral"]
    )

    print("\n" + "="*55)
    print("EJEMPLO DE ASIGNACIÓN BLANDA (Soft Clustering)")
    print("="*55)
    print("Cada reseña tiene una probabilidad sobre TODOS los topics,")
    print("no solo '1' para uno y '0' para los demás.\n")

    print("\n" + "="*55)
    print("PASO 4: Guardando resultados...")
    print("="*55)

    # 10. Concatenamos los resultados y los guardamos en un CSV.
    # El parámetro fill_na_prefix asegura que las columnas de probabilidad se rellenen con 0.0 para los neutros.
    concat_and_export_csv(
        [df_pos_final, df_neg_final, df_neu_final],
        os.path.join(cfg["project"]["output_dir"], cfg["export"]["filename"]),
        csv_separator=cfg["export"]["separator"],
        fill_na_prefix='prob_topic_'
    )

    print(f"✅ Guardado: '{cfg['export']['filename']}'")
    print("\n🎉 ¡Soft clustering completado!")


if __name__ == "__main__":
    # 1. Definimos los argumentos de entrada por terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON de configuración")

    # 2. Ejecutamos el pipeline con el archivo de configuración proporcionado
    args = parser.parse_args()
    run_soft_pipeline(args.config)
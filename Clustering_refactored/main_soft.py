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

# Importaciones de módulos internos del proyecto
from src.data.loader import load_selected_columns_from_csv, clean_and_cast_numeric_column, drop_nulls_and_cast_to_string
from src.data.preprocessor import classify_sentiment_by_score, apply_basic_text_cleaning, \
    filter_dataframe_by_column_value, lemmatize_text
from src.features.vectorizers import build_count_vectorizer, fit_and_transform_texts
from src.models.lda_logic import (
    evaluate_lda_coherence_gensim,
    evaluate_lda_loglikelihood,
    get_optimal_topics_from_scores,
    fit_lda_model,
    get_top_words_per_topic
)
from src.visualization.dimensional_math import sample_data_for_tsne, reduce_dimensions_with_tsne
from src.visualization.plotters import (
    export_tsne_scatter_plot,
    export_lda_dual_metric_plot  # <-- Usaremos esta para mostrar Coherencia vs Perplejidad
)
from src.visualization.html_exporters import export_interactive_lda_html
from src.utils.file_system import ensure_directory_exists
from src.utils.dataframe_exporters import add_top_words_label, add_topic_probabilities, concat_and_export_csv


def process_soft_sentiment_subset(df_subset: pd.DataFrame, sentiment_label: str, cfg: Dict[str, Any], forced_k: Optional[int] = None) -> pd.DataFrame:
    """
    Ejecuta el pipeline de Soft Clustering (LDA) integrando métricas de Coherencia y Perplejidad.
    """
    seed = cfg["project"]["random_state"]

    # 1. Parámetros y filtros
    features_cfg = cfg["features"].copy()
    sentiment_overrides = cfg.get("features_by_sentiment", {}).get(sentiment_label, {})
    features_cfg.update(sentiment_overrides)

    min_words = sentiment_overrides.get("min_words", cfg["data"].get("min_words", 0))
    if min_words > 0:
        mask = df_subset['review_limpio'].apply(lambda x: len(x.split()) >= min_words)
        df_subset = df_subset[mask].copy().reset_index(drop=True)
        print(f"  [{sentiment_label.upper()}] Filtro min_words={min_words}: {len(df_subset)} reseñas restantes")

    raw_texts = df_subset['review_limpio'].tolist()

    # 2. Vectorización (CountVectorizer para LDA)
    print(f"\n  [{sentiment_label.upper()}] Vectorizando con CountVectorizer...")
    vectorizer_model = build_count_vectorizer(
        features_cfg["max_features"],
        features_cfg["min_df"],
        features_cfg["max_df"],
        cfg["stopwords"],
        tuple(features_cfg.get("ngram_range", [1, 1]))
    )
    X, vectorizer_fitted, vocab = fit_and_transform_texts(vectorizer_model, raw_texts)

    # 3. Evaluación de métricas (Coherencia y Perplejidad)
    print(f"  [{sentiment_label.upper()}] Evaluando métricas para {cfg['model']['n_topics_range']} topics...")

    # Preparamos textos tokenizados para Gensim
    tokenized_texts = [t.split() for t in raw_texts]

    # Calculamos Coherencia C_V (Humana/Semántica)
    coherence_scores = evaluate_lda_coherence_gensim(
        X, tokenized_texts, vocab, cfg["model"]["n_topics_range"],
        seed, cfg["model"]["processes"], cfg["model"]["eval_max_iter"]
    )

    # Calculamos Log-Likelihood y Perplejidad (Estadística)
    ll_scores, perplexity_scores = evaluate_lda_loglikelihood(
        X, cfg["model"]["n_topics_range"], seed, cfg["model"]["sklearn_eval_max_iter"]
    )

    # 4. Determinación del K óptimo (Priorizamos Coherencia Semántica)
    optimal_k = forced_k if forced_k else get_optimal_topics_from_scores(coherence_scores, cfg["model"]["n_topics_range"])

    if forced_k:
        print(f"  [{sentiment_label.upper()}] K forzado manualmente: {optimal_k}")
    else:
        print(f"  [{sentiment_label.upper()}] K óptimo detectado (via Coherencia): {optimal_k}")

    # 5. Exportación de gráfico de diagnóstico (Dual: Coherencia vs Perplejidad)
    # Reutilizamos export_lda_dual_metric_plot ajustando labels para Coherencia y Perplejidad
    export_lda_dual_metric_plot(
        cfg["model"]["n_topics_range"],
        coherence_scores,      # Panel izquierdo: Coherencia
        perplexity_scores,     # Panel derecho: Perplejidad (Codo)
        optimal_k,
        f"{sentiment_label.capitalize()} (Coherencia vs Perplejidad)",
        os.path.join(cfg["project"]["output_dir"], f'diagnostico_lda_{sentiment_label}.png'),
        figsize=tuple(cfg["visualization"]["style"]["figsize_eval"]),
        dpi=cfg["visualization"]["style"]["dpi"]
    )

    # 6. Entrenamiento del modelo final
    print(f"  [{sentiment_label.upper()}] Entrenando LDA final con {optimal_k} topics...")
    lda_model, doc_topic_matrix, dominant_topics = fit_lda_model(X, optimal_k, seed, cfg["model"]["final_max_iter"])

    # 7. Post-procesamiento y Visualización t-SNE
    df_subset['topic_dominante'] = dominant_topics
    topic_keywords = get_top_words_per_topic(lda_model, vocab, optimal_k, cfg["model"]["n_top_words"])

    print(f"  [{sentiment_label.upper()}] Generando visualización t-SNE...")
    X_sampled, labels_sampled = sample_data_for_tsne(
        doc_topic_matrix, dominant_topics, cfg["visualization"]["tsne"]["samples"], seed
    )
    coords_2d = reduce_dimensions_with_tsne(
        X_sampled, cfg["visualization"]["tsne"]["perplexity"], seed, cfg["visualization"]["tsne"]["max_iter"]
    )

    export_tsne_scatter_plot(
        coords_2d, labels_sampled, optimal_k, topic_keywords,
        f'Mapa de Topics (Soft) — Reseñas {sentiment_label.capitalize()}\n(K={optimal_k}, Coloreado por topic dominante)',
        os.path.join(cfg["project"]["output_dir"], f'tsne_soft_{sentiment_label}.png'),
        n_words_legend=cfg["visualization"]["style"]["n_words_legend"],
        colormap_name=cfg["visualization"]["style"]["colormap"],
        figsize=tuple(cfg["visualization"]["style"]["figsize_tsne"]),
        dpi=cfg["visualization"]["style"]["dpi"]
    )

    # 8. Exportación Interactiva (pyLDAvis)
    export_interactive_lda_html(
        lda_model, X, vectorizer_fitted,
        os.path.join(cfg["project"]["output_dir"], f'lda_interactivo_{sentiment_label}.html')
    )

    # 9. Preparación de DataFrame de salida
    df_out = add_top_words_label(
        df_subset, 'topic_dominante', 'top_palabras', topic_keywords, cfg["export"]["top_words_in_csv"]
    )
    return add_topic_probabilities(df_out, doc_topic_matrix, optimal_k)


def run_soft_pipeline(config_path: str) -> None:
    """
    Coordina la ejecución completa del proceso de Soft Clustering (LDA).
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        cfg = json.load(file)

    ensure_directory_exists(cfg["project"]["output_dir"])

    print("="*55)
    print("PASO 1: Cargando y preparando los datos...")
    print("="*55)

    df = load_selected_columns_from_csv(cfg["data"]["input_csv"], cfg["data"]["columns_to_load"])
    df = clean_and_cast_numeric_column(df, cfg["data"]["score_col"])
    df = drop_nulls_and_cast_to_string(df, cfg["data"]["text_col"])

    df[cfg["data"]["sentiment_col"]] = df[cfg["data"]["score_col"]].apply(
        lambda x: classify_sentiment_by_score(x, cfg["data"]["thresholds"]["min_positive"], cfg["data"]["thresholds"]["exact_neutral"])
    )

    df['review_limpio'] = df[cfg["data"]["text_col"]].apply(apply_basic_text_cleaning)

    labels = cfg["data"]["labels"]
    df_pos = filter_dataframe_by_column_value(df, cfg["data"]["sentiment_col"], labels["positive"])
    df_neg = filter_dataframe_by_column_value(df, cfg["data"]["sentiment_col"], labels["negative"])

    print("\n" + "=" * 55)
    print("PASO 2: Soft Clustering de reseñas POSITIVAS")
    print("=" * 55)
    df_pos_final = process_soft_sentiment_subset(
        df_pos,
        labels["positive"],
        cfg,
        forced_k=cfg["model"].get("forced_k_pos")
    )

    print("\n" + "="*55)
    print("PASO 3: Soft Clustering de reseñas NEGATIVAS")
    print("="*55)
    df_neg_final = process_soft_sentiment_subset(df_neg, labels["negative"], cfg, forced_k=cfg["model"]["forced_k_neg"])

    print("\n" + "="*55)
    print("PASO 4: Guardando resultados...")
    print("="*55)

    concat_and_export_csv(
        [df_pos_final, df_neg_final],
        os.path.join(cfg["project"]["output_dir"], cfg["export"]["filename"]),
        csv_separator=cfg["export"]["separator"],
        fill_na_prefix='prob_topic_'
    )

    print(f"✅ Guardado: '{cfg['export']['filename']}'")
    print("\n🎉 ¡Soft clustering completado con métricas de coherencia!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON de configuración")
    args = parser.parse_args()
    run_soft_pipeline(args.config)
"""
SOFT CLUSTERING — FEMALE SUBSET
=====================================================
Ejecuta el mismo pipeline de LDA (Soft Clustering) pero filtrando
exclusivamente el dataset para mantener solo las reseñas de mujeres.
"""

import argparse
import json
import os
import pandas as pd

# Importaciones de tus módulos internos
from src.data.loader import load_selected_columns_from_csv, clean_and_cast_numeric_column, drop_nulls_and_cast_to_string
from src.data.preprocessor import classify_sentiment_by_score, apply_basic_text_cleaning, \
    filter_dataframe_by_column_value
from src.utils.file_system import ensure_directory_exists
from src.utils.dataframe_exporters import concat_and_export_csv

# ¡Importamos la función core de tu main_soft.py para no duplicar código!
from main_soft import process_soft_sentiment_subset


def run_female_soft_pipeline(config_path: str, gender_label: str = 'female') -> None:
    """
    Coordina la ejecución del Soft Clustering solo para el subconjunto femenino.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        cfg = json.load(file)

    # Modificamos dinámicamente las rutas de salida para no pisar el análisis general
    original_out_dir = cfg["project"]["output_dir"]
    cfg["project"]["output_dir"] = f"{original_out_dir}_femenino"

    out_filename = cfg["export"]["filename"].replace('.csv', '_femenino.csv')

    ensure_directory_exists(cfg["project"]["output_dir"])

    print("=" * 55)
    print(f"PASO 1: Cargando datos y filtrando por género ('{gender_label}')...")
    print("=" * 55)

    # Carga base
    df = load_selected_columns_from_csv(cfg["data"]["input_csv"], cfg["data"]["columns_to_load"])

    # ---------------------------------------------------------
    # EL FILTRO MÁGICO: Nos quedamos solo con la representación femenina
    # ---------------------------------------------------------
    total_reviews = len(df)
    df = df[df['gender'] == gender_label].copy().reset_index(drop=True)
    print(f"  Filtro aplicado: {len(df)} reseñas femeninas de un total de {total_reviews}.")

    if len(df) == 0:
        print(f"❌ Error: No se encontraron reseñas con el género '{gender_label}'. Revisa cómo está escrito en el CSV.")
        return

    # Limpieza estándar
    df = clean_and_cast_numeric_column(df, cfg["data"]["score_col"])
    df = drop_nulls_and_cast_to_string(df, cfg["data"]["text_col"])

    df[cfg["data"]["sentiment_col"]] = df[cfg["data"]["score_col"]].apply(
        lambda x: classify_sentiment_by_score(x, cfg["data"]["thresholds"]["min_positive"],
                                              cfg["data"]["thresholds"]["exact_neutral"])
    )
    df['review_limpio'] = df[cfg["data"]["text_col"]].apply(apply_basic_text_cleaning)

    labels = cfg["data"]["labels"]
    df_pos = filter_dataframe_by_column_value(df, cfg["data"]["sentiment_col"], labels["positive"])
    df_neg = filter_dataframe_by_column_value(df, cfg["data"]["sentiment_col"], labels["negative"])

    print("\n" + "=" * 55)
    print("PASO 2: Soft Clustering - POSITIVAS (Femenino)")
    print("=" * 55)
    # Reutilizamos la función de tu main_soft
    df_pos_final = process_soft_sentiment_subset(df_pos, labels["positive"], cfg,
                                                 forced_k=cfg["model"].get("forced_k_pos"))

    print("\n" + "=" * 55)
    print("PASO 3: Soft Clustering - NEGATIVAS (Femenino)")
    print("=" * 55)
    df_neg_final = process_soft_sentiment_subset(df_neg, labels["negative"], cfg, forced_k=cfg["model"]["forced_k_neg"])

    print("\n" + "=" * 55)
    print("PASO 4: Guardando resultados...")
    print("=" * 55)

    concat_and_export_csv(
        [df_pos_final, df_neg_final],
        os.path.join(cfg["project"]["output_dir"], out_filename),
        csv_separator=cfg["export"]["separator"],
        fill_na_prefix='prob_topic_'
    )

    print(f"✅ Guardado en: {cfg['project']['output_dir']}/{out_filename}")
    print("\n🎉 ¡Clustering del subconjunto femenino completado!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON de configuración")
    # Añadimos un argumento por si el género está escrito en español o de otra forma en tu dataset
    parser.add_argument("--gender", type=str, default="female",
                        help="Etiqueta exacta del género en el CSV (ej: 'female', 'Mujer', 'F')")
    args = parser.parse_args()

    run_female_soft_pipeline(args.config, args.gender)
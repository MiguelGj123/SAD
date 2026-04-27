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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ──────────────────────────────────────────────
# CARPETA DE SALIDA
# ──────────────────────────────────────────────
OUTPUT_DIR = "resultados_hard_clustering"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Resultados en: ./{OUTPUT_DIR}/")

# ──────────────────────────────────────────────
# 1. CARGA Y PREPROCESADO
# ──────────────────────────────────────────────
print("="*55)
print("PASO 1: Cargando y preparando los datos...")
print("="*55)

df = pd.read_csv('SoundCloud.csv')
df = df[['reviewId', 'review', 'score', 'gender', 'location', 'date']].copy()

# El CSV tiene ~13 filas con comas sin escapar en el texto de la reseña,
# lo que desplaza las columnas y mete texto en la columna 'score'.
# to_numeric con errors='coerce' convierte esas celdas corruptas a NaN
# para poder eliminarlas limpiamente antes de castear a int.
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df.dropna(subset=['review', 'score'], inplace=True)
df['score'] = df['score'].astype(int)
df['review'] = df['review'].astype(str)

# Convertimos score en sentimiento (como pide el enunciado)
def score_to_sentiment(score):
    if score >= 4:
        return 'positivo'
    elif score == 3:
        return 'neutro'
    else:
        return 'negativo'

df['sentimiento'] = df['score'].apply(score_to_sentiment)
print(f"Total reseñas: {len(df)}")
print(df['sentimiento'].value_counts())


def limpiar_texto(texto):
    """Limpieza básica multilingüe (no elimina palabras clave de idiomas distintos)"""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)           # URLs
    texto = re.sub(r'[^\w\s]', ' ', texto)                  # puntuación
    texto = re.sub(r'\d+', '', texto)                        # números
    texto = re.sub(r'\s+', ' ', texto).strip()              # espacios extra
    return texto

df['review_limpio'] = df['review'].apply(limpiar_texto)

# ──────────────────────────────────────────────
# 2. FUNCIÓN GENÉRICA DE CLUSTERING
# ──────────────────────────────────────────────

def run_kmeans_clustering(df_subset, etiqueta, k_range=range(2, 11), k_final=None):
    """
    Ejecuta el pipeline completo de hard clustering sobre un subconjunto.

    Parámetros
    ----------
    df_subset : DataFrame con columna 'review_limpio'
    etiqueta  : str, p.ej. 'positivo' o 'negativo' (para nombrar ficheros)
    k_range   : rango de K a probar en el codo
    k_final   : si None, elige automáticamente el K del codo

    Devuelve
    --------
    df_subset con columna 'cluster_id' añadida
    vectorizer, model_kmeans (para reutilizar)
    """

    textos = df_subset['review_limpio'].tolist()

    # ── 2a. TF-IDF ──────────────────────────────
    print(f"\n  [{etiqueta.upper()}] Vectorizando con TF-IDF ({len(textos)} reseñas)...")
    stopwords_extra = [
        'it', 'the', 'to', 'and', 'is', 'this', 'that', 'for', 'of', 'in',
        'my', 'me', 'you', 'your', 'so', 'but', 'not', 'be', 'are', 'was',
        'have', 'has', 'had', 'do', 'did', 'will', 'would', 'can', 'could',
        'with', 'on', 'at', 'by', 'as', 'an', 'or', 'if', 'up', 'out', 'all',
        'its', 'from', 'just', 'they', 'their', 'them', 'we', 'our', 'us',
        'he', 'she', 'been', 'more', 'than', 'when', 'what', 'how', 'now',
        'get', 'got', 'even', 'also', 'very', 'too', 'still', 'after',
        'before', 'soundcloud', 'sound', 'cloud',
    ]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.80,
        sublinear_tf=True,
        stop_words=stopwords_extra,
    )
    X = vectorizer.fit_transform(textos)
    print(f"  Matriz TF-IDF: {X.shape}")

    # ── 2b. GRÁFICO DEL CODO ─────────────────────
    print(f"  [{etiqueta.upper()}] Calculando inercia para K={list(k_range)}...")
    inercias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
        km.fit(X)
        inercias.append(km.inertia_)
        if k > 1:
            muestra_idx = np.random.choice(len(textos), min(500, len(textos)), replace=False)
            labels_muestra = km.labels_[muestra_idx]
            if len(np.unique(labels_muestra)) > 1:
                sil = silhouette_score(X[muestra_idx], labels_muestra)
            else:
                sil = 0.0
            silhouettes.append(sil)
        else:
            silhouettes.append(0)

    # Dibujamos codo + silhouette en una figura de 2 paneles
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inercias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inercia (WCSS)', fontsize=12)
    axes[0].set_title(f'Gráfico del Codo — Reseñas {etiqueta.capitalize()}', fontsize=13)
    axes[0].grid(alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Número de clusters (K)', fontsize=12)
    axes[1].set_ylabel('Puntuación Silhouette', fontsize=12)
    axes[1].set_title(f'Silhouette — Reseñas {etiqueta.capitalize()}', fontsize=13)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    nombre_codo = os.path.join(OUTPUT_DIR, f'codo_{etiqueta}.png')
    plt.savefig(nombre_codo, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Gráfico del codo guardado: '{nombre_codo}'")

    # ── 2c. ELEGIR K ÓPTIMO ──────────────────────
    if k_final is None:
        # Usamos la puntuación Silhouette (más robusta que la segunda derivada)
        # Ignoramos k=2 (índice 0) que siempre tiende a dar silhouette alto artificialmente
        silhouettes_arr = np.array(silhouettes)
        # Buscamos el máximo de silhouette entre k=3 en adelante si hay suficientes
        if len(silhouettes_arr) > 2:
            k_final = list(k_range)[np.argmax(silhouettes_arr[1:]) + 1]
        else:
            k_final = list(k_range)[np.argmax(silhouettes_arr)]
        print(f"  K óptimo por Silhouette: {k_final}")
    else:
        print(f"  K elegido manualmente: {k_final}")

    # ── 2d. CLUSTERING FINAL ─────────────────────
    print(f"  [{etiqueta.upper()}] Entrenando K-Means con K={k_final}...")
    model_kmeans = KMeans(n_clusters=k_final, random_state=42, n_init='auto', max_iter=500)
    etiquetas_cluster = model_kmeans.fit_predict(X)
    df_subset = df_subset.copy()
    df_subset['cluster_id'] = etiquetas_cluster

    print(f"\n  Distribución de clusters ({etiqueta}):")
    print(df_subset['cluster_id'].value_counts().sort_index().to_string())

    # ── 2e. PALABRAS SIGNIFICATIVAS ──────────────
    print(f"\n  [{etiqueta.upper()}] Extrayendo palabras más significativas por cluster...")
    feature_names = vectorizer.get_feature_names_out()
    centroides = model_kmeans.cluster_centers_

    palabras_por_cluster = {}
    n_top_words = 15  # cuántas palabras mostrar por cluster

    for i in range(k_final):
        # Los índices de mayor peso en el centroide = palabras más representativas
        top_idx = centroides[i].argsort()[::-1][:n_top_words]
        top_words = [feature_names[j] for j in top_idx]
        palabras_por_cluster[i] = top_words
        print(f"    Cluster {i} ({(df_subset['cluster_id']==i).sum()} reseñas): "
              f"{', '.join(top_words[:8])}")

    # ── 2f. VISUALIZACIÓN t-SNE ──────────────────
    print(f"\n  [{etiqueta.upper()}] Generando visualización t-SNE...")
    # Muestreamos para que sea manejable
    n_muestra = min(1500, len(textos))
    idx_muestra = np.random.choice(len(textos), n_muestra, replace=False)
    X_muestra = X[idx_muestra]
    labels_muestra = etiquetas_cluster[idx_muestra]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords_2d = tsne.fit_transform(X_muestra.toarray())

    colores = plt.colormaps.get_cmap('tab10').resampled(k_final)
    fig, ax = plt.subplots(figsize=(12, 8))

    for cid in range(k_final):
        mask = labels_muestra == cid
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[colores(cid)],
            label=f"Cluster {cid}: {', '.join(palabras_por_cluster[cid][:3])}",
            alpha=0.6, s=25
        )

    ax.set_title(f'Mapa de Clusters — Reseñas {etiqueta.capitalize()} (t-SNE)', fontsize=14)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    nombre_tsne = os.path.join(OUTPUT_DIR, f'tsne_hard_{etiqueta}.png')
    plt.savefig(nombre_tsne, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Mapa t-SNE guardado: '{nombre_tsne}'")

    return df_subset, vectorizer, model_kmeans, palabras_por_cluster, k_final


# ──────────────────────────────────────────────
# 3. APLICAR A POSITIVOS Y NEGATIVOS
# ──────────────────────────────────────────────

print("\n" + "="*55)
print("PASO 2: Clustering de reseñas POSITIVAS")
print("="*55)
df_pos = df[df['sentimiento'] == 'positivo'].reset_index(drop=True)
df_pos, vec_pos, km_pos, words_pos, k_pos = run_kmeans_clustering(
    df_pos, etiqueta='positivo', k_range=range(2, 9), k_final=None
)

print("\n" + "="*55)
print("PASO 3: Clustering de reseñas NEGATIVAS")
print("="*55)
df_neg = df[df['sentimiento'] == 'negativo'].reset_index(drop=True)
df_neg, vec_neg, km_neg, words_neg, k_neg = run_kmeans_clustering(
    df_neg, etiqueta='negativo', k_range=range(2, 9), k_final=None
)

# ──────────────────────────────────────────────
# 4. RESUMEN INTERPRETATIVO
# ──────────────────────────────────────────────
print("\n" + "="*55)
print("RESUMEN — PALABRAS POR CLUSTER")
print("="*55)

print(f"\n🟢 RESEÑAS POSITIVAS ({k_pos} clusters):")
for cid, words in words_pos.items():
    n = (df_pos['cluster_id'] == cid).sum()
    print(f"  Cluster {cid} ({n} reseñas): {' | '.join(words[:10])}")

print(f"\n🔴 RESEÑAS NEGATIVAS ({k_neg} clusters):")
for cid, words in words_neg.items():
    n = (df_neg['cluster_id'] == cid).sum()
    print(f"  Cluster {cid} ({n} reseñas): {' | '.join(words[:10])}")

# ──────────────────────────────────────────────
# 5. GUARDAR CSV PARA TABLEAU
# ──────────────────────────────────────────────
print("\n" + "="*55)
print("PASO 4: Guardando resultados...")
print("="*55)

# Añadimos columna con las top palabras del cluster (útil en Tableau como tooltip)
def get_label(row, words_dict):
    return ' | '.join(words_dict[row['cluster_id']][:5])

df_pos['top_palabras'] = df_pos.apply(lambda r: get_label(r, words_pos), axis=1)
df_neg['top_palabras'] = df_neg.apply(lambda r: get_label(r, words_neg), axis=1)

# Combinamos todo
df_resultado = pd.concat([df_pos, df_neg], ignore_index=True)

# Para neutros no hay clustering (pocos datos generalmente), los añadimos sin cluster
df_neu = df[df['sentimiento'] == 'neutro'].copy()
df_neu['cluster_id'] = -1
df_neu['top_palabras'] = 'neutro'
df_resultado = pd.concat([df_resultado, df_neu], ignore_index=True)

df_resultado.to_csv(os.path.join(OUTPUT_DIR, 'resultados_hard_clustering.csv'), sep='~', index=False)
print("✅ Guardado: 'resultados_hard_clustering.csv'")
print(f"   Columnas: {df_resultado.columns.tolist()}")
print("\n🎉 ¡Hard clustering completado!")
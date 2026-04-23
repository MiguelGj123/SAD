"""
SOFT CLUSTERING — LDA (Latent Dirichlet Allocation)
=====================================================
Cada reseña tiene una DISTRIBUCIÓN DE PROBABILIDAD sobre los topics.
No pertenece "solo" a uno: puede ser 60% topic_0, 30% topic_2, 10% topic_1.

LDA es el estándar académico para topic modeling en NLP.
Genera:
  - Gráfico de coherencia para elegir el número óptimo de topics (equivalente al codo)
  - Topics separados para reseñas POSITIVAS y NEGATIVAS
  - Palabras más representativas por topic
  - CSV con probabilidades por topic para Tableau
  - Visualización pyLDAvis (HTML interactivo)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# Intentamos importar gensim para la coherencia (métrica más robusta que la log-likelihood)
try:
    import gensim
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    GENSIM_OK = True
except ImportError:
    GENSIM_OK = False
    print("⚠️  gensim no instalado. Se usará log-likelihood de sklearn para la coherencia.")
    print("   Para instalarlo: pip install gensim")

# Intentamos importar pyLDAvis para la visualización interactiva
try:
    import pyLDAvis
    import pyLDAvis.lda_model
    PYLDAVIS_OK = True
except ImportError:
    PYLDAVIS_OK = False
    print("⚠️  pyLDAvis no instalado. Se omitirá el HTML interactivo.")
    print("   Para instalarlo: pip install pyLDAvis")


# ──────────────────────────────────────────────
# CARPETA DE SALIDA
# ──────────────────────────────────────────────
OUTPUT_DIR = "resultados_soft_clustering"
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

def score_to_sentiment(score):
    if score >= 4:
        return 'positivo'
    elif score == 3:
        return 'neutro'
    else:
        return 'negativo'

df['sentimiento'] = df['score'].apply(score_to_sentiment)

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

df['review_limpio'] = df['review'].apply(limpiar_texto)
print(f"Total reseñas: {len(df)}")
print(df['sentimiento'].value_counts())


# ──────────────────────────────────────────────
# 2. FUNCIÓN GENÉRICA DE SOFT CLUSTERING (LDA)
# ──────────────────────────────────────────────

def run_lda_clustering(df_subset, etiqueta, n_topics_range=range(2, 9), n_topics_final=None):
    """
    Pipeline completo de soft clustering (LDA) sobre un subconjunto.

    La diferencia clave con K-Means:
      - K-Means asigna cada doc a 1 cluster (hard)
      - LDA asigna a cada doc una DISTRIBUCIÓN sobre todos los topics (soft)

    Parámetros
    ----------
    df_subset       : DataFrame con 'review_limpio'
    etiqueta        : str, 'positivo' o 'negativo'
    n_topics_range  : rango de K a probar
    n_topics_final  : forzar un K concreto (None = automático)
    """

    textos = df_subset['review_limpio'].tolist()

    # ── 2a. BAG OF WORDS (LDA necesita conteos, no TF-IDF) ──────────
    print(f"\n  [{etiqueta.upper()}] Vectorizando con CountVectorizer ({len(textos)} reseñas)...")
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

    vectorizer = CountVectorizer(
        max_features=4000,
        min_df=5,
        max_df=0.80,
        stop_words=stopwords_extra,
    )
    X = vectorizer.fit_transform(textos)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  Vocabulario: {len(feature_names)} términos")

    # ── 2b. BÚSQUEDA DEL K ÓPTIMO — COHERENCIA / LOG-LIKELIHOOD ─────
    print(f"  [{etiqueta.upper()}] Probando {list(n_topics_range)} topics...")

    scores = []

    if GENSIM_OK:
        # Coherencia c_v con gensim (métrica recomendada en papers académicos)
        textos_tokenizados = [t.split() for t in textos]
        diccionario = Dictionary(textos_tokenizados)
        corpus_bow = [diccionario.doc2bow(t) for t in textos_tokenizados]

        for n in n_topics_range:
            lda = LatentDirichletAllocation(
                n_components=n, random_state=42, max_iter=15
            )
            lda.fit(X)

            # Convertimos los topics de sklearn al formato gensim
            topics_words = []
            for topic_idx in range(n):
                top_idx = lda.components_[topic_idx].argsort()[::-1][:20]
                topics_words.append([feature_names[i] for i in top_idx])

            cm_model = CoherenceModel(
                topics=topics_words,
                texts=textos_tokenizados,
                dictionary=diccionario,
                coherence='c_v'
            )
            coherencia = cm_model.get_coherence()
            scores.append(coherencia)
            print(f"    n_topics={n:2d}  coherencia c_v={coherencia:.4f}")

        ylabel = 'Coherencia c_v (mayor = mejor)'
        mejor_k_idx = np.argmax(scores)

    else:
        # Fallback: log-likelihood de sklearn (mayor = mejor)
        for n in n_topics_range:
            lda = LatentDirichletAllocation(
                n_components=n, random_state=42, max_iter=20, n_jobs=-1
            )
            lda.fit(X)
            scores.append(lda.score(X))
            print(f"    n_topics={n:2d}  log-likelihood={scores[-1]:.1f}")

        ylabel = 'Log-Likelihood (mayor = mejor)'
        mejor_k_idx = np.argmax(scores)

    # ── 2c. GRÁFICO DE COHERENCIA ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(n_topics_range), scores, 'go-', linewidth=2, markersize=9)
    ax.axvline(
        x=list(n_topics_range)[mejor_k_idx],
        color='red', linestyle='--',
        label=f'Óptimo: K={list(n_topics_range)[mejor_k_idx]}'
    )
    ax.set_xlabel('Número de Topics', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Coherencia de Topics — Reseñas {etiqueta.capitalize()}', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    nombre_coh = os.path.join(OUTPUT_DIR, f'coherencia_soft_{etiqueta}.png')
    plt.savefig(nombre_coh, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Gráfico de coherencia guardado: '{nombre_coh}'")

    # ── 2d. MODELO FINAL ──────────────────────────────────────────────
    if n_topics_final is None:
        n_topics_final = list(n_topics_range)[mejor_k_idx]
        print(f"  K óptimo detectado: {n_topics_final}")
    else:
        print(f"  K elegido manualmente: {n_topics_final}")

    print(f"  [{etiqueta.upper()}] Entrenando LDA final con {n_topics_final} topics...")
    lda_final = LatentDirichletAllocation(
        n_components=n_topics_final,
        random_state=42,
        max_iter=50,
        learning_method='online'
    )
    doc_topic_matrix = lda_final.fit_transform(X)
    # doc_topic_matrix.shape = (n_docs, n_topics)
    # Cada fila es la distribución de probabilidad de ese doc sobre los topics
    # ¡Esta es la clave del soft clustering!

    # ── 2e. PALABRAS REPRESENTATIVAS ─────────────────────────────────
    print(f"\n  [{etiqueta.upper()}] Palabras más representativas por topic:")
    n_top = 15
    palabras_por_topic = {}
    for topic_idx in range(n_topics_final):
        top_idx = lda_final.components_[topic_idx].argsort()[::-1][:n_top]
        top_words = [feature_names[i] for i in top_idx]
        palabras_por_topic[topic_idx] = top_words
        n_docs_principal = (doc_topic_matrix.argmax(axis=1) == topic_idx).sum()
        print(f"    Topic {topic_idx} ({n_docs_principal} docs como principal): "
              f"{' | '.join(top_words[:8])}")

    # ── 2f. TOPIC DOMINANTE Y DISTRIBUCIÓN ───────────────────────────
    df_subset = df_subset.copy()
    df_subset['topic_dominante'] = doc_topic_matrix.argmax(axis=1)

    # Guardamos la probabilidad de cada topic como columnas separadas
    for t in range(n_topics_final):
        df_subset[f'prob_topic_{t}'] = doc_topic_matrix[:, t].round(4)

    # ── 2g. VISUALIZACIÓN t-SNE COLOREADA POR TOPIC DOMINANTE ────────
    print(f"\n  [{etiqueta.upper()}] Generando visualización t-SNE...")
    n_muestra = min(1500, len(textos))
    idx_muestra = np.random.choice(len(textos), n_muestra, replace=False)
    X_muestra = doc_topic_matrix[idx_muestra]  # proyectamos en el espacio de topics
    labels_muestra = df_subset['topic_dominante'].values[idx_muestra]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords_2d = tsne.fit_transform(X_muestra)

    colores = plt.colormaps.get_cmap('tab10').resampled(n_topics_final)
    fig, ax = plt.subplots(figsize=(12, 8))

    for tid in range(n_topics_final):
        mask = labels_muestra == tid
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[colores(tid)],
            label=f"Topic {tid}: {' | '.join(palabras_por_topic[tid][:3])}",
            alpha=0.6, s=25
        )

    ax.set_title(
        f'Mapa de Topics (Soft) — Reseñas {etiqueta.capitalize()} (t-SNE)\n'
        '(Coloreado por topic dominante; cada punto tiene distribución sobre todos los topics)',
        fontsize=12
    )
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    nombre_tsne = os.path.join(OUTPUT_DIR, f'tsne_soft_{etiqueta}.png')
    plt.savefig(nombre_tsne, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE guardado: '{nombre_tsne}'")

    # ── 2h. PYLDAVIS — visualización interactiva (HTML) ───────────────
    if PYLDAVIS_OK:
        try:
            panel = pyLDAvis.lda_model.prepare(lda_final, X, vectorizer, mds='tsne')
            nombre_html = os.path.join(OUTPUT_DIR, f'lda_interactivo_{etiqueta}.html')
            pyLDAvis.save_html(panel, nombre_html)
            print(f"  Visualización interactiva guardada: '{nombre_html}'")
        except Exception as e:
            print(f"  ⚠️ pyLDAvis falló: {e}")

    return df_subset, lda_final, palabras_por_topic, n_topics_final


# ──────────────────────────────────────────────
# 3. APLICAR A POSITIVOS Y NEGATIVOS
# ──────────────────────────────────────────────

print("\n" + "="*55)
print("PASO 2: Soft Clustering de reseñas POSITIVAS")
print("="*55)
df_pos = df[df['sentimiento'] == 'positivo'].reset_index(drop=True)
df_pos, lda_pos, words_pos, k_pos = run_lda_clustering(
    df_pos, etiqueta='positivo', n_topics_range=range(2, 8)
)

print("\n" + "="*55)
print("PASO 3: Soft Clustering de reseñas NEGATIVAS")
print("="*55)
df_neg = df[df['sentimiento'] == 'negativo'].reset_index(drop=True)
df_neg, lda_neg, words_neg, k_neg = run_lda_clustering(
    df_neg, etiqueta='negativo', n_topics_range=range(2, 8), n_topics_final=5
)

# ──────────────────────────────────────────────
# 4. RESUMEN INTERPRETATIVO
# ──────────────────────────────────────────────
print("\n" + "="*55)
print("RESUMEN — TOPICS Y SUS PALABRAS CLAVE")
print("="*55)

print(f"\n🟢 RESEÑAS POSITIVAS ({k_pos} topics):")
for tid, words in words_pos.items():
    n = (df_pos['topic_dominante'] == tid).sum()
    print(f"  Topic {tid} ({n} docs como principal): {' | '.join(words[:10])}")

print(f"\n🔴 RESEÑAS NEGATIVAS ({k_neg} topics):")
for tid, words in words_neg.items():
    n = (df_neg['topic_dominante'] == tid).sum()
    print(f"  Topic {tid} ({n} docs como principal): {' | '.join(words[:10])}")

# ──────────────────────────────────────────────
# 5. EJEMPLO DE SOFT ASSIGNMENT (para entender la diferencia)
# ──────────────────────────────────────────────
print("\n" + "="*55)
print("EJEMPLO DE ASIGNACIÓN BLANDA (Soft Clustering)")
print("="*55)
print("Esto es lo que hace el soft clustering diferente al hard:")
print("Cada reseña tiene una probabilidad sobre TODOS los topics,")
print("no solo '1' para uno y '0' para los demás.\n")

for i in range(min(3, len(df_neg))):
    review_orig = df_neg.iloc[i]['review'][:100]
    probs = {t: df_neg.iloc[i][f'prob_topic_{t}'] for t in range(k_neg)}
    print(f"  Reseña: '{review_orig}...'")
    for t, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(p * 30)
        print(f"    Topic {t} ({' | '.join(words_neg[t][:3])}): {p:.3f} {bar}")
    print()

# ──────────────────────────────────────────────
# 6. GUARDAR CSV PARA TABLEAU
# ──────────────────────────────────────────────
print("="*55)
print("PASO 4: Guardando resultados...")
print("="*55)

def get_label_soft(row, words_dict):
    return ' | '.join(words_dict[row['topic_dominante']][:5])

df_pos['top_palabras'] = df_pos.apply(lambda r: get_label_soft(r, words_pos), axis=1)
df_neg['top_palabras'] = df_neg.apply(lambda r: get_label_soft(r, words_neg), axis=1)

df_resultado = pd.concat([df_pos, df_neg], ignore_index=True)

df_neu = df[df['sentimiento'] == 'neutro'].copy()
df_neu['topic_dominante'] = -1
df_neu['top_palabras'] = 'neutro'
df_resultado = pd.concat([df_resultado, df_neu], ignore_index=True)

df_resultado.to_csv(os.path.join(OUTPUT_DIR, 'resultados_soft_clustering.csv'), index=False)
print("✅ Guardado: 'resultados_soft_clustering.csv'")
print(f"   Columnas relevantes: reviewId, review, score, sentimiento, "
      f"topic_dominante, top_palabras, prob_topic_0 ... prob_topic_N")
print("\n🎉 ¡Soft clustering completado!")
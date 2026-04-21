import pandas as pd
from sentence_transformers import SentenceTransformer
import time
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
import ollama

# 1. Cargar el CSV
df_completo = pd.read_csv('SoundCloud.csv')

# 2. Nos quedamos solo con las columnas que nos interesan (Texto y puntuación para contexto)
df = df_completo[['reviewId', 'review', 'score', 'location']].copy()

# 3. Tomamos una muestra aleatoria de 1000 reseñas para prototipar rápido
# Usamos random_state=42 para que siempre elija las mismas 1000 y los resultados no cambien
df_muestra = df.sample(n=1000, random_state=42).reset_index(drop=True)

# 4. Extraemos solo los textos a una lista, que es lo que pide el algoritmo
textos_reviews = df_muestra['review'].tolist()

print(f"Lista de textos preparada con {len(textos_reviews)} reseñas.")

print("1. Cargando el modelo matemático de lenguaje...")
# Usamos un modelo multilingüe. Viendo tus datos, hay ubicaciones de todo el mundo
# (México, Indonesia, USA), así que habrá varios idiomas, jerga y emojis.
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("2. Convirtiendo las 1000 reseñas a vectores matemáticos...")
# Ponemos un temporizador para que veas cuánto tarda tu máquina
inicio = time.time()

# Esta es la línea donde ocurre la magia profunda.
# 'show_progress_bar=True' te mostrará una barrita de carga en la consola.
embeddings = modelo.encode(textos_reviews, show_progress_bar=True)

fin = time.time()

print(f"\n¡Proceso terminado en {round(fin - inicio, 2)} segundos!")
print(f"Forma de la matriz de resultados: {embeddings.shape}")

print("\n3. Buscando grupos (Clustering) con K-Means...")
# Elegimos 5 grupos para empezar (puedes cambiar este número luego)
num_grupos = 5
# n_init='auto' es para evitar un warning futuro de scikit-learn
modelo_kmeans = KMeans(n_clusters=num_grupos, random_state=42, n_init='auto')

# Entrenamos el algoritmo y obtenemos a qué grupo pertenece cada reseña
etiquetas = modelo_kmeans.fit_predict(embeddings)

# Guardamos el resultado en nuestro DataFrame original
df_muestra['cluster_id'] = etiquetas
print("¡Agrupación terminada!")

print("\n4. Comprimiendo de 384 a 2 dimensiones para dibujar el gráfico...")
# t-SNE es genial para comprimir texto y poder dibujarlo
tsne = TSNE(n_components=2, random_state=42)
vectores_2d = tsne.fit_transform(embeddings)

# Añadimos las coordenadas X e Y al DataFrame
df_muestra['coord_x'] = vectores_2d[:, 0]
df_muestra['coord_y'] = vectores_2d[:, 1]

print("5. Creando y guardando el gráfico...")
plt.figure(figsize=(10, 8)) # Tamaño de la imagen
# Dibujamos los puntos coloreados por su cluster
sns.scatterplot(
    data=df_muestra,
    x='coord_x',
    y='coord_y',
    hue='cluster_id',
    palette='Set1', # Paleta de colores diferenciados
    s=60,           # Tamaño de los puntos
    alpha=0.8       # Transparencia
)

plt.title(f'Mapa de Similitud: {num_grupos} Grupos de Reseñas de SoundCloud')
# Quitamos los números de los ejes porque en este caso no significan nada físico
plt.xticks([])
plt.yticks([])

# Guardamos la imagen en la misma carpeta donde está tu script
plt.savefig('grafico_clusters.png', dpi=300, bbox_inches='tight')
print("¡Éxito! Abre el archivo 'grafico_clusters.png' en tu carpeta para ver los grupos.")

# Por curiosidad, veamos cuántas reseñas han caído en cada grupo
print("\n=== Distribución de los grupos ===")
print(df_muestra['cluster_id'].value_counts().sort_index())

print("\n=== PASO 5: PIDIENDO A LA IA QUE INTERPRETE LOS GRUPOS ===")
print("(Esto puede tardar un par de minutos, tu CPU está pensando...)")


def nombrar_cluster(mensajes_muestra):
    """Toma una muestra de reseñas y le pide a Ollama que las resuma en una frase"""
    texto_mensajes = "\n".join([f"- {m}" for m in mensajes_muestra])

    prompt = f"""
    Eres un analista de datos revisando opiniones de la aplicación SoundCloud.
    Aquí tienes una muestra de reseñas que un algoritmo ha agrupado porque son muy similares entre sí.

    Tu tarea: Lee las reseñas y dale un título a este grupo.
    El título debe indicar el sentimiento (Positivo, Negativo o Neutro) y el tema principal del que hablan.
    REGLA ESTRICTA: Responde ÚNICAMENTE con el título de la categoría. Máximo 6 palabras. No des explicaciones.

    Reseñas:
    {texto_mensajes}
    """

    try:
        respuesta = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return respuesta['message']['content'].strip()
    except Exception as e:
        return f"Error al conectar con Ollama: {e}"


# Iteramos sobre los 5 grupos que encontró K-Means (del 0 al 4)
for cluster_id in sorted(df_muestra['cluster_id'].unique()):

    # Extraemos todas las reseñas que cayeron en este grupo
    reseñas_grupo = df_muestra[df_muestra['cluster_id'] == cluster_id]['review'].tolist()

    # Para no saturar la memoria de la IA, le enviamos solo una muestra de 15 reseñas al azar
    # (Si el grupo tiene menos de 15, tomamos las que haya)
    muestra_para_llm = random.sample(reseñas_grupo, min(15, len(reseñas_grupo)))

    # Le pasamos la muestra a la IA para que piense el nombre
    etiqueta_ia = nombrar_cluster(muestra_para_llm)

    # Imprimimos los resultados en la consola
    print(f"\n--- GRUPO {cluster_id} (Contiene {len(reseñas_grupo)} reseñas) ---")
    print(f"🤖 Etiqueta de la IA: {etiqueta_ia}")
    print("📝 Ejemplos reales de este grupo:")

    # Imprimimos 3 reseñas de ejemplo (truncadas a 120 caracteres para no manchar mucho la consola)
    for msg in muestra_para_llm[:3]:
        texto_limpio = msg.replace('\n', ' ')[:120]
        print(f"   * {texto_limpio}...")

print("\n¡PROYECTO COMPLETADO! 🎉")
# Finalmente, guardamos todo en un CSV nuevo por si quieres revisarlo en Excel
df_muestra.to_csv('resultados_clustering_muestra.csv', index=False)
print("Se ha guardado el archivo 'resultados_clustering_muestra.csv' con todas las etiquetas.")
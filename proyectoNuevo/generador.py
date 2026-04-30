import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# =================================================================
# VARIABLES DE ENTORNO Y PRUEBAS
# =================================================================
# Cambiar para generar más o menos datos sintéticos
N_MUESTRAS_GEN = 100

# Opciones: "llama3", "llama3.1" (recomendada), "gemma2:2b"
NOMBRE_MODELO = "llama3.1"

# =================================================================
# CONFIGURACIÓN DE HIPERPARÁMETROS
# =================================================================
llm_generador = OllamaLLM(
    model=NOMBRE_MODELO,
    # Temperatura a 0.4: Un poco de creatividad para usar sinónimos, pero sin delirar
    temperature=0.4,
    num_predict=150,
    repeat_penalty=1.2
)

# =================================================================
# PROCESAMIENTO DE DATOS
# =================================================================
print("Cargando datos originales de SoundCloud para buscar la clase minoritaria...")
df = pd.read_csv('SoundCloud.csv')

df['score'] = pd.to_numeric(df['score'], errors='coerce')
df = df.dropna(subset=['score', 'review'])


def obtener_clase(nota):
    if nota <= 2:
        return "negativo"
    elif nota == 3:
        return "neutro"
    else:
        return "positivo"


df['sentimiento_real'] = df['score'].apply(obtener_clase)

# =================================================================
# TAREA 2: OVERSAMPLING GENERATIVO (Clase Neutra)
# =================================================================
# Filtramos solo los comentarios neutros
comentarios_neutros = df[df['sentimiento_real'] == 'neutro'].head(N_MUESTRAS_GEN)

# PROMPT ESTRICTO DE SISTEMA EN INGLÉS
template_parafraxis = """
You are a strict paraphrasing algorithm. Rewrite the text below into a single, neutral, formal English sentence.
Do NOT output anything else. NO emojis. NO conversational text. NO examples.

Input: "{comentario}"
Output:"""

prompt_gen = PromptTemplate.from_template(template_parafraxis)
chain_gen = prompt_gen | llm_generador

print(f"\n--- Ejecutando Generación de Datos con el modelo {NOMBRE_MODELO} ---")
nuevos_datos = []

for i, row in comentarios_neutros.iterrows():
    # Obtenemos respuesta, limpiamos, cortamos en el primer salto de línea y quitamos comillas
    respuesta_bruta = chain_clase = chain_gen.invoke({"comentario": row['review']}).strip()
    parafraxis_limpia = respuesta_bruta.split('\n')[0].replace('"', '')

    nuevos_datos.append({
        "comentario_original": row['review'],
        "parafraxis_generada": parafraxis_limpia,
        "clase": "neutro"
    })

    print(f"Original : {row['review'][:80]}...")
    print(f"Generado : {parafraxis_limpia}\n")

# Guardamos el CSV final
pd.DataFrame(nuevos_datos).to_csv('parafraxis_generadas.csv', index=False)
print("¡Éxito! Archivo 'parafraxis_generadas.csv' creado.")
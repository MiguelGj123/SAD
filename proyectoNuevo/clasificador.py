import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import os


# =================================================================
# VARIABLES DE ENTORNO Y PRUEBAS
# =================================================================
# Cambiar estos valores para ajustar la velocidad de las pruebas
N_MUESTRAS_TEST = 10

# Opciones disponibles: "zero-shot" o "few-shot"
TIPO_PROMPT = "few-shot"

# Opciones de modelos
# "llama3", "llama3.1", "gemma2:2b"
NOMBRE_MODELO = "llama3"

# =================================================================
# CONFIGURACIÓN DE HIPERPARÁMETROS
# =================================================================
llm_clasificador = OllamaLLM(
    model=NOMBRE_MODELO,
    temperature=0.0,
    num_predict=15,
    seed=42,
    repeat_penalty=1.1
)

# =================================================================
# PROCESAMIENTO DE DATOS
# =================================================================
print("Cargando y limpiando datos de SoundCloud...")
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
# TAREA 1: CLASIFICACIÓN
# =================================================================
template_zero_shot = """
You are a strict and highly precise sentiment analysis algorithm.
Your only task is to read a user review and classify its sentiment.
You MUST reply with EXACTLY ONE LETTER. No explanations, no punctuation, no greetings.

A = Positive
B = Negative
C = Neutral

Review: "{comentario}"
Class:"""

template_few_shot = """
You are a strict and highly precise sentiment analysis algorithm.
Your only task is to read a user review and classify its sentiment.
You MUST reply with EXACTLY ONE LETTER. No explanations, no punctuation, no greetings.

A = Positive
B = Negative
C = Neutral

Example 1: "Great music but too many ads" -> C
Example 2: "app crashes every time I open it" -> B
Example 3: "I love it, I can listen to my old voicemails" -> A
Example 4: "don't get me wrong it's okay, but it lacks features" -> C

Review: "{comentario}"
Class:"""
# Selección del prompt
if TIPO_PROMPT == "few-shot":
    prompt_clase = PromptTemplate.from_template(template_few_shot)
else:
    prompt_clase = PromptTemplate.from_template(template_zero_shot)

chain_clase = prompt_clase | llm_clasificador

muestra_test = df.sample(N_MUESTRAS_TEST, random_state=42)
print(f"\n--- Ejecutando Clasificación con el modelo {NOMBRE_MODELO} (Modo: {TIPO_PROMPT}) ---")

valores_reales = []
valores_predichos = []

# contador para ir mostrando el porcentaje completado
total_muestras = len(muestra_test)
contador = 0
paso_10_porciento = total_muestras // 10  # Calculamos cuántas muestras son el 10%
# -----------------------------------------------------------

for index, row in muestra_test.iterrows():

    # Definimos la frase sacándola de la fila actual
    frase = row['review']

    # 1. Obtener la salida del modelo
    salida_modelo = chain_clase.invoke({"comentario": frase}).strip().upper()

    # 2. Quedarnos solo con la primera letra (ej: "A.")
    letra = salida_modelo[0] if len(salida_modelo) > 0 else "X"

    # 3. Mapear la letra a tus clases originales
    if letra == "A":
        clase_asignada = "positivo"
    elif letra == "B":
        clase_asignada = "negativo"
    elif letra == "C":
        clase_asignada = "neutro"
    else:
        clase_asignada = "desconocido"  # Fallback de seguridad

    # Guardamos los valores en las listas para el F1-Score final
    valores_reales.append(row['sentimiento_real'])
    valores_predichos.append(clase_asignada)

    # 4. Mostrar por pantalla
    print(f"Real: {row['sentimiento_real']} | IA: {clase_asignada} | Texto: {frase[:40]}...")

    # 4.1 Comprobar por pantalla la respuesta de la ia (para ver si obedece a la regla de A/B/C)
    # #print(        f"Real: {row['sentimiento_real']} | Respuesta IA: '{salida_modelo}'  | Texto: {frase[:40]}...")

    # --- Lógica para imprimir el progreso ---
    contador += 1

    if paso_10_porciento > 0 and contador % paso_10_porciento == 0:
        porcentaje = int((contador / total_muestras) * 100)
        print("-------------------------------")
        print(f"{porcentaje}% completado")
        print("-------------------------------")
    # -----------------------------------------------


# =================================================================
#EXPORTACIÓN DE RESULTADOS (CSVs)
# =================================================================
import os
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd

# 1. Crear la carpeta 'resultados' si no existe
os.makedirs('resultados', exist_ok=True)

# 2. Definimos las etiquetas y calculamos métricas
etiquetas = ['positivo', 'negativo', 'neutro']
macro_f1 = f1_score(valores_reales, valores_predichos, average='macro', labels=etiquetas, zero_division=0)
micro_f1 = f1_score(valores_reales, valores_predichos, average='micro', labels=etiquetas, zero_division=0)

# 3. CREACIÓN DEL ARCHIVO: F1-score-train.csv
# Preparamos los strings de los parámetros
hiperparametros = f"Modelo: {llm_clasificador.model} | Temp: {llm_clasificador.temperature} | max_tokens: {llm_clasificador.num_predict} | seed: {llm_clasificador.seed} | repeat_penalty: {llm_clasificador.repeat_penalty} | Prompt: {TIPO_PROMPT}"

datos_f1 = {
    "Metrica": [
        "Mejores parametros",
        "Mejor puntuacion",
        "F1-score micro",
        "F1-score macro"
    ],
    "Valor": [
        hiperparametros,
        f"{macro_f1:.4f}", #usamos la macro como puntuacion general
        f"{micro_f1:.4f}",
        f"{macro_f1:.4f}"
    ]
}

df_f1 = pd.DataFrame(datos_f1)

ruta_f1 = os.path.join('resultados', 'F1-score-train.csv')
df_f1.to_csv(ruta_f1, index=False, encoding='utf-8', header=False)

# 4. MATRIZ DE CONFUSIÓN (La mantenemos para análisis detallado)
matriz = confusion_matrix(valores_reales, valores_predichos, labels=etiquetas)
df_matriz = pd.DataFrame(matriz,
                         index=[f"Real_{e}" for e in etiquetas],
                         columns=[f"Pred_{e}" for e in etiquetas])

ruta_matriz = os.path.join('resultados', f'matriz_confusion_train.csv')
df_matriz.to_csv(ruta_matriz, encoding='utf-8')

# 5. Resumen final por consola
print("\n=======================================================")
print(f"✅ EXPORTACIÓN FINALIZADA")
print("=======================================================")
print(f"Resultados guardados en: {ruta_f1}")
print(df_f1.to_string(index=False))
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

directorio = os.path.dirname(os.path.abspath(__file__))
archivos = [f for f in os.listdir(directorio) if f.endswith('.csv') and 'resultados_' in f]

dfs_finales = []

print("--- TRANSFORMANDO AL FORMATO PERFECTO PARA TABLEAU ---")
for archivo in archivos:
    ruta = os.path.join(directorio, archivo)
    
    # 1. Leer archivo de forma segura
    try:
        df = pd.read_csv(ruta, sep=';', dtype={'reviewId': str}, on_bad_lines='warn')
        if 'reviewId' not in df.columns:
            df = pd.read_csv(ruta, sep=',', dtype={'reviewId': str}, on_bad_lines='warn')
    except Exception as e:
        print(f"Error leyendo {archivo}: {e}")
        continue
        
    if 'reviewId' not in df.columns:
        continue
        
    df['reviewId'] = df['reviewId'].astype(str).str.strip()
    
    # 2. Extraer Metadatos Clave desde el nombre del archivo
    nombre = archivo.replace('resultados_', '').replace('.csv', '')
    partes = nombre.split('_')
    
    # Averiguar Hard o Soft
    tipo_clustering = partes[0] # "hard" o "soft"
    
    # Averiguar App y Género Analizado
    app = "Desconocida"
    modelo_genero = "General"
    
    if "SoundCloud" in partes:
        app = "SoundCloud"
    elif "TIDAL" in partes:
        app = "TIDAL"
        
    if "Femenino" in partes:
        modelo_genero = "Femenino"
    elif "Masculino" in partes:
        modelo_genero = "Masculino"

    # 3. Añadir las columnas identificadoras para los filtros en Tableau
    df['App_Analizada'] = app
    df['Tipo_Clustering'] = tipo_clustering
    df['Modelo_Aplicado'] = modelo_genero
    
    # Crear un ID Único Real para evitar colisiones
    df['ID_Unico_Real'] = df['App_Analizada'] + "_" + df['reviewId']
    
    # 4. Homogeneizar columnas
    # En Hard clustering sueles tener 'cluster_id', en Soft tienes 'prob_topic_X'
    # Vamos a dejarlas tal cual, Tableau las manejará bien porque ahora las filas están separadas
    
    dfs_finales.append(df)
    print(f"Procesado: {app} | {tipo_clustering} | {modelo_genero} -> {len(df)} filas.")

# 5. Apilar (Union) todo
df_tableau = pd.concat(dfs_finales, ignore_index=True)

ruta_salida = os.path.join(directorio, "Dataset_Final_Tableau.csv")
df_tableau.to_csv(ruta_salida, index=False)

print(f"\n=> TOTAL FILAS EN DATASET: {len(df_tableau)}")
print("¡Abre este archivo en Tableau!")

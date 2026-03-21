# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""

import random
import sys
import signal
import argparse

import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm


# Funciones auxiliares

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-f", "--file", help="Fichero de datos .csv (/Path_to_file)", required=True)
    parse.add_argument("-m", "--model", help="Fichero de modelo .pkl (/Path_to_file)", required=True)
    parse.add_argument("-j", "--json", help="Fichero de configuración .json (/Path_to_file)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open(args.json) as json_file:
        config = json.load(json_file)
    
    # Juntamos los argumentos en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        #Fore sirve para dar color
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        #Quitar la columna a predecir
        data_features = data.drop(columns=[args.prediction], errors="ignore")

        # Numerical features
        numerical_feature = data_features.select_dtypes(include=['int64', 'float64']) # Columnas numéricas

        # Categorical features
        categorical_feature = data_features.select_dtypes(include='object')
        #Quedarse solo con los atributos categoricos que tengan X o menos posibles valores distintos.
        # X se define en el json en unique_category_threshold.
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
        
        # Text features
        # Selecciona todas las columnas categoricas, y solo se quedan con las que tienen más de X posibles valores distintos.
        text_feature = data_features.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
        
        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None

    Raises:
        None
    """

    global data
    try:
        # 1. Leemos las estrategias del JSON (con valores por defecto por si el usuario olvida ponerlas)
        # Asumimos que en tu JSON hay algo como: "preprocessing": {"imputer_num": "mean", "imputer_cat": "most_frequent"}
        strategy_num = args.preprocessing.get("imputer_num", "mean")
        strategy_cat = args.preprocessing.get("imputer_cat", "most_frequent")

        # --- VARIABLES NUMÉRICAS ---
        if len(numerical_feature.columns) > 0:
            # Le pasamos la variable directamente en lugar del texto a fuego
            imputer_num = SimpleImputer(strategy=strategy_num)
            data[numerical_feature.columns] = imputer_num.fit_transform(data[numerical_feature.columns])

        # --- VARIABLES CATEGÓRICAS ---
        if len(categorical_feature.columns) > 0:
            imputer_cat = SimpleImputer(strategy=strategy_cat)
            data[categorical_feature.columns] = imputer_cat.fit_transform(data[categorical_feature.columns])
        print("Valores nulos procesados")

    except Exception as e:
        print(Fore.RED + "Error al procesar los valores nulos" + Fore.RESET)
        print(e)
        sys.exit(1)

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    global data
    try:
        if len(numerical_feature.columns) > 0:
            # Leemos qué escalador quiere el usuario (ej: "preprocessing": {"scaler": "minmax"})
            tipo_scaler = args.preprocessing.get("scaler", "standard").lower()

            # Elegimos la herramienta según el JSON
            if tipo_scaler == "standard":
                scaler = StandardScaler()
            elif tipo_scaler == "minmax":
                scaler = MinMaxScaler()
            elif tipo_scaler == "robust":
                scaler = RobustScaler()
            elif tipo_scaler == "maxabs":
                scaler = MaxAbsScaler()
            elif tipo_scaler == "none":
                print(Fore.YELLOW + "No se aplica reescalado según el JSON" + Fore.RESET)
                return  # Salimos de la función sin hacer nada
            else:
                print(
                    Fore.RED + f"Escalador '{tipo_scaler}' no reconocido. Usando StandardScaler por defecto." + Fore.RESET)
                scaler = StandardScaler()

            # Aplicamos el escalador elegido
            data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
            print("Escalado completado con exito")

    except Exception as e:
        print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
        print(e)
        exit(1)


def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
    global data
    try:
        if len(categorical_feature.columns)>0 :
            estrategia = args.preprocessing.get("categorical_to_num", "none")
            if estrategia == "ordinal":
                # --- ESTRATEGIA 1: Codificación Ordinal (Label Encoding) ---
                # Convierte cada categoría en un número entero (ej. Rojo=0, Verde=1, Azul=2).
                # PROS: Mantiene una sola columna, no aumenta el tamaño de los datos. Ideal para Árboles de Decisión.
                # CONTRAS: Pésimo para kNN. kNN creerá que "Azul" (2) vale el doble que "Verde" (1), lo cual es falso si no hay orden real.

                encoder = OrdinalEncoder()
                data[categorical_feature.columns] = encoder.fit_transform(data[categorical_feature.columns])
            elif estrategia == "onehot":
                # --- ESTRATEGIA 2: One-Hot Encoding (Variables Dummy) ---
                # Crea una columna nueva por cada categoría con 0s y 1s.
                # PROS: Perfecto para kNN porque no inventa un orden o jerarquía falsa entre categorías.
                # CONTRAS: Si tienes una categoría con 100 valores distintos, te creará 100 columnas nuevas, haciendo el dataset enorme.

                data = pd.get_dummies(data, columns=categorical_feature.columns, drop_first=True)
            elif estrategia == "none":
                print(Fore.YELLOW + f"No se transforman datos categóricos a numéricos" + Fore.RESET)
                return
            else:
                print(Fore.YELLOW + f"Estrategia de transformación de datos categóricos a numéticos: '{estrategia},' no reconocida" + Fore.RESET)
                return

            print(Fore.GREEN + "Variables categóricas convertidas a numéricas con éxito" + Fore.RESET)

    except Exception as e:
        print("Error en la transformacion")
        print(e)
        exit(1)

def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....
    
    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.
    
    Retorna:
    None
    """
    global data
    try:
        if len(text_feature.columns) > 0:
            #preparamos las herramientas q usaremos luego
            stop_words = set(stopwords.words('spanish'))  # Cambia a 'english' u otros idiomas
            stemmer = PorterStemmer()

            for col in text_feature.columns:
                #Convertir el texto a minúsculas
                data[col] = data[col].astype(str).str.lower()

                #Eliminar signos de puntuación (¡, ?, ., ,)
                data[col] = data[col].str.translate(str.maketrans('', '', string.punctuation))

                #Función interna para procesar cada celda de texto
                def clean_sentence(text):
                    # Tokenizar (separar la frase en palabras sueltas)
                    tokens = word_tokenize(text)
                    # Quitar stopwords y aplicar stemming
                    clean_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
                    # Volver a unir las palabras en una frase limpia
                    return " ".join(clean_tokens)

                # Aplicamos la función a toda la columna
                data[col] = data[col].apply(clean_sentence)

            print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)

def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":               
               tfidf_vectorizer = TfidfVectorizer()
               text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
               tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
               text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
               data = pd.concat([data, text_features_df], axis=1)
               data.drop(text_feature.columns, axis=1, inplace=True)
               print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)

            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)

            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        # Quitar la columna a predecir
        data = data.drop(columns=[args.prediction], errors="ignore")

        atributos_eliminar = args.preprocessing.get("drop_features", [])
        if len(atributos_eliminar) >0:
            data = data.drop(columns=atributos_eliminar)
            print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
        else:
            print(Fore.GREEN+"Se ha decidido no eliminar ninguna columna"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Borramos columnas no necesarias (Especificarlas en .json)
        2. Separamos los datos por tipos (Categoriales, numéricos y textos)
        3. Tratamos missing values (Eliminar y imputar)
        4. Pasar los datos de categoriales a numéricos
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        7. Tratamos el texto (TF-IDF, BOW)
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    # Borrar columnas no necesarias
    drop_features()

    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)

    # Simplificamos el texto
    simplify_text(text_feature)

    # Reescalamos los datos numéricos
    reescaler(numerical_feature)
    
    # Tratamos el texto
    process_text(text_feature)

    return data

# Funciones para entrenar un modelo

def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Parámetros:
    - data: DataFrame que contiene los datos.
    - args: Objeto que contiene los argumentos necesarios para la división de datos.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    """
    # Sacamos la columna a predecir

    global data  # Usamos nuestra variable global con los datos ya limpios
    try:
        # 1. Separamos X (las pistas) de Y (la respuesta)
        X = data.drop(columns=[args.prediction])  # dropeamos todas menos la columna a predecir
        y = data[args.prediction]  # Solo la columna a predecir

        # 2. Partimos los datos en dos grupos
        x_train, x_dev, y_train, y_dev = train_test_split(
            X, y,
            test_size=0.2,       # 20% de los datos para el examen, 80% para estudiar
            random_state=42,  # Semilla para que el corte sea siempre el mismo si repites
            stratify=y  # Clave: Mantiene la proporción de las categorías
        )

        print(Fore.GREEN + "Datos divididos en Train y Dev con éxito" + Fore.RESET)
        return x_train, x_dev, y_train, y_dev

    except Exception as e:
        print(Fore.RED + "Error al dividir los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

# Funciones para predecir con un modelo

def load_model(model):
    """
    Carga el modelo desde el archivo que se pasa por consola y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open(model, 'rb') as file:
            modelo = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return modelo
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)
        
def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    prediction = model.predict(data)
    
    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)
    
# Función principal

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)

    # Cargamos el modelo
    print("\n- Cargando modelo...")
    model = load_model(args.model)
    # Predecimos
    print("\n- Prediciendo...")

    try:
        predict()
        print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
        # Guardamos el dataframe con la prediccion
        data.to_csv('output/data-prediction.csv', index=False)
        print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)

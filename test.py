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
    parse.add_argument("-s", "--separador", help='Si se usa, separa el archivo csv con el argumento que se le pase (";" o ",")', required=False, default=',')
    parse.add_argument("-st", "--sentiment", help='Si se usa, se asume que la columna de predicción es una puntuación del 1 al 5, y la transforma a "neutral","positivo" y "negativo"', required=False, action="store_true")
    parse.add_argument("-v", "--verbose", help="Muestra un resumen de los resultados por la terminal", required=False, default=False, action="store_true")
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
        # Obtener el separador.
        separador = args.separador

        if separador == ";":
            data = pd.read_csv(file, sep=";", encoding='utf-8')
        elif separador == ",":
            data = pd.read_csv(file, encoding='utf-8')
        else:
            print("No se reconoce el separador especificado en --separador")
            exit(1)

        # Eliminar columnas erroneas
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

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

        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas

        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        #Quedarse solo con los atributos categoricos que tengan X o menos posibles valores distintos.
        # X se define en el json en unique_category_threshold.
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
        
        # Text features
        # Selecciona todas las columnas categoricas, y solo se quedan con las que tienen más de X posibles valores distintos.
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

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


def cat2num(categorical_feature, cat2num_cols=None):
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

                #Usar las columnas obtenidas con el train para que sean las mismas
                data[categorical_feature.columns] = cat2num_cols.transform(data[categorical_feature.columns])

            elif estrategia == "onehot":

                data = pd.get_dummies(data, columns=categorical_feature.columns, drop_first=True)
                # alinear con train
                data = data.reindex(columns=cat2num_cols, fill_value=0)

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
            stop_words = set(stopwords.words('english'))  # Cambia a 'english' u otros idiomas
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

def process_text(text_feature, vectorizer, text_columns):
    """
    Procesa las columnas de texto del test usando el vectorizer ya entrenado.

    Parámetros:
    - text_feature (DataFrame): columnas de texto del test
    - vectorizer: TF-IDF o CountVectorizer entrenado
    - text_columns: lista de columnas de texto usadas en train

    Retorna:
    - data_test con las columnas vectorizadas
    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] != "none":
                # combinamos el texto de las columnas
                text_data = text_feature[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

                # transformamos usando el vectorizer ya entrenado
                matrix = vectorizer.transform(text_data)

                # convertimos a DataFrame con mismas columnas que el train
                text_features_df = pd.DataFrame(matrix.toarray(), columns=text_columns)

                # agregamos al DataFrame
                #data_test = pd.concat([text_feature, datos_text], axis=1)

                # eliminar las columnas originales de texto
                #data_test.drop(text_columns, axis=1, inplace=True)

                return text_features_df

            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
        return None

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

def map_sentiment(score):
    putnuacion = int(score)
    if putnuacion == 3:
        return "neutral"
    elif putnuacion > 3:
        return "positive"
    elif putnuacion < 3:
        return "negative"

def preprocesar_datos(vectorizer=None, text_columns=None, cat2num_cols=None):
    """
    Función para preprocesar los datos
        1. Borramos target si existe (generalmente no habrá porque es dataset a predecir).
        2. Borramos columnas no necesarias (Especificarlas en .json)
        2. Separamos los datos por tipos (Categoriales, numéricos y textos)
        3. Tratamos missing values (Eliminar y imputar)
        4. Pasar los datos de categoriales a numéricos
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        7. Tratamos el texto (TF-IDF, BOW)
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    global data

    # Nos quedamos solo con features, y guardamos el target por si acaso se incluye, para comparar predicciones
    # (NO SE USA EL TARGET EN LA PREDICCIÓN)
    if args.prediction in data.columns:

        # Si sentiment analysis, forzar el score a numérico, y convertir en null lo que no lo sea
        # Evitar fallos de comas en el texto.
        # Traducir score a "positivo", "negativo", "neutro" si se trata con sentiment analysis
        if args.sentiment:
            data[args.prediction] = pd.to_numeric(data[args.prediction], errors="coerce")
            data[args.prediction] = data[args.prediction].fillna(data[args.prediction].mode()[0])  # para clasificación
            data[args.prediction] = data[args.prediction].apply(map_sentiment)

        y = data[args.prediction]

        # Tratamos missing values de la target (si hace falta)
        if y.isnull().any() and not args.sentiment:
            y = y.fillna(y.mode()[0])  # para clasificación

        data = data.drop(columns=[args.prediction], errors="ignore")
    else:
        y = None #No se incluye un target para comparar

    # Borrar columnas no necesarias
    drop_features()

    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature, cat2num_cols)

    # Simplificamos el texto
    simplify_text(text_feature)

    # Tratamos el texto
    datos_text = process_text(text_feature, vectorizer, text_columns)

    #Si hay texto que ha sido tratado
    if datos_text is not None:
        # eliminar texto original
        data = data.drop(columns=text_feature.columns)

        # agregamos al DataFrame
        data = pd.concat([data, datos_text], axis=1)

    # alinear con modelo
    data = data.reindex(columns=model.feature_names_in_, fill_value=0)

    # volver a separar datos por tipos, para reconocer los nuevos datos numéricos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Reescalamos los datos numéricos
    reescaler(numerical_feature)

    # devolvemos a data los valores del target, si existe, temporalmente
    # if y is not None:
    #     data[args.prediction] = y

    return data, y

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
            saved = pickle.load(file)
            modelo = saved["gs"]
            vectorizer = saved["vectorizer"]
            text_columns = saved["text_columns"]
            cat2num_cols = saved["cat2num_cols"]
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return modelo, vectorizer, text_columns, cat2num_cols
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)
        
def predict(y_test):
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

    mostrar_resultados(prediction, y_test)
    
    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)

def calculate_classification_report(y_true, y_pred):
    """
    Genera un informe de texto con Precision, Recall y F1 para cada clase.
    """
    #Hacer el clasification report
    cr = classification_report(y_true, y_pred, zero_division=0)
    with open('output/classification_report_test.txt', 'w') as f:
        f.write(cr)

    return cr

def calculate_confusion_matrix(y_true, y_pred):
    """
    Genera la matriz de confusión para ver dónde se equivoca el modelo.
    """
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm)
    df_cm.to_csv('output/matriz_confusion_test.csv', index=False)

    return cm

def mostrar_resultados(pred, y_test):
    """
        Muestra resultados de predicción para un modelo ya entrenado sobre el conjunto de test.

        Parámetros:
        - pred: La predicción del modelo sobre el dataset
        - y_test: La columna con los valores reales del target. Estará vacío si no se incluye en el dataset.
    """

    print(Fore.MAGENTA+"\nValores únicos en y_test:"+Fore.RESET)
    print(pd.Series(y_test).unique()[:20])

    cont = pd.Series(pred).value_counts()
    print(Fore.MAGENTA + "> Distribución de predicciones:\n" + Fore.RESET)
    print(cont)

    if args.verbose and y_test is not None:
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, f1_score(y_test, pred, average='micro'))
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, f1_score(y_test, pred, average='macro'))
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_test, pred))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_test, pred))

    
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

    if args.debug:
        print(data.head())
        print(data.columns)
        print(data.dtypes)

    ###para probar sin incluir target
    #data = data.drop(columns=[args.prediction], errors="ignore")

    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")

    # Cargamos el modelo
    print("\n- Cargando modelo...")
    model, vectorizer, text_columns, cat2num_cols = load_model(args.model)

    if args.debug:
        print(Fore.MAGENTA+"\nANTES DEL PREPROCESADO"+Fore.RESET)
        cols = data.columns.tolist()
        print(Fore.MAGENTA + "Columnas test:" + Fore.RESET)
        print("Inicio:", cols[:10])
        print("Final:", cols[-10:])
    print(Fore.MAGENTA+"Columnas modelo:"+Fore.RESET, model.feature_names_in_)

    data, y = preprocesar_datos(vectorizer, text_columns, cat2num_cols)

    # Nos quedamos solo con features, y guardamos el target por si acaso se incluye, para comparar predicciones
    # (NO SE USA EL TARGET EN LA PREDICCIÓN)
    # if args.prediction in data.columns:
    #     y = data[args.prediction]
    #     data = data.drop(columns=[args.prediction], errors="ignore")
    # else:
    #     y = None

    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)

    if args.debug:
        print(Fore.MAGENTA+"\nANTES DE PREDECIR"+Fore.RESET)
        cols = data.columns.tolist()
        print(Fore.MAGENTA + "Columnas test:" + Fore.RESET)
        print("Inicio:", cols[:10])
        print("Final:", cols[-10:])
        print(Fore.MAGENTA+"Columnas modelo:"+Fore.RESET, model.feature_names_in_)

    # Predecimos
    print("\n- Prediciendo...")

    try:
        predict(y)
        print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
        # Guardamos el dataframe con la prediccion
        data.to_csv('output/data-prediction.csv', index=False)
        print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
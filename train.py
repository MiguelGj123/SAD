# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""

import random
import sys
import signal
import argparse
import re

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
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

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
    parse.add_argument("-j", "--json", help="Fichero de configuración .json (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree, random_forest, naive_bayes o logistic_regression)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-s", "--separador", help='Si se usa, separa el archivo csv con el argumento que se le pase (";" o ",")', required=False, default=',')
    parse.add_argument("-st", "--sentiment", help='Si se usa, se asume que la columna de predicción es una puntuación del 1 al 5, y la transforma a "neutral","positivo" y "negativo"', required=False, action="store_true")
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
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

def map_sentiment(score):
    putnuacion = int(score)
    if putnuacion == 3:
        return "neutral"
    elif putnuacion > 3:
        return "positive"
    elif putnuacion < 3:
        return "negative"

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

        #Eliminar columnas erroneas
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

                print(Fore.GREEN + "Variables categóricas convertidas a numéricas con éxito" + Fore.RESET)
                return encoder
            elif estrategia == "onehot":
                # --- ESTRATEGIA 2: One-Hot Encoding (Variables Dummy) ---
                # Crea una columna nueva por cada categoría con 0s y 1s.
                # PROS: Perfecto para kNN porque no inventa un orden o jerarquía falsa entre categorías.
                # CONTRAS: Si tienes una categoría con 100 valores distintos, te creará 100 columnas nuevas, haciendo el dataset enorme.

                data = pd.get_dummies(data, columns=categorical_feature.columns, drop_first=True)
                cat2num_cols = data.columns.tolist()
                print(Fore.GREEN + "Variables categóricas convertidas a numéricas con éxito" + Fore.RESET)
                return cat2num_cols
            elif estrategia == "none":
                print(Fore.YELLOW + f"No se transforman datos categóricos a numéricos" + Fore.RESET)
                return None
            else:
                print(Fore.YELLOW + f"Estrategia de transformación de datos categóricos a numéticos: '{estrategia},' no reconocida" + Fore.RESET)
                return None

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
               #con max_features limitamos la cantidad de palabras que se usan, para datasets muy grandes.
               tfidf_vectorizer = TfidfVectorizer(max_features=3500)
               text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
               tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
               text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

               # Guardar columnas procesadas
               text_cols = text_features_df.columns.tolist()

               data = pd.concat([data, text_features_df], axis=1)
               data.drop(text_feature.columns, axis=1, inplace=True)
               print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)
               return text_cols,tfidf_vectorizer

            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer(max_features=3500)
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())

                text_cols = text_features_df.columns.tolist()

                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)
                return text_cols, bow_vecotirizer

            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
        return None, None
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    global data
    try:
        estrategia = args.preprocessing.get("sampling", "none").lower()
        if estrategia == "none":
            print("no hay balanceo")
            return

        # 1. Separamos temporalmente las características (X) del objetivo (y)
        X = data.drop(columns=[args.prediction])
        y = data[args.prediction]

        # 2. Aplicamos la estrategia elegida
        if estrategia == "undersampling":
            # Reduce la clase mayoritaria
            sampler = RandomUnderSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
            print(Fore.GREEN + "Undersampling aplicado con éxito" + Fore.RESET)

        elif estrategia == "oversampling":
            # Multiplica la clase minoritaria
            sampler = RandomOverSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
            print(Fore.GREEN + "Oversampling aplicado con éxito" + Fore.RESET)

        else:
            print(Fore.YELLOW + f"Estrategia de sampling '{estrategia}' no reconocida" + Fore.RESET)
            return

        # 3. Volvemos a juntar los datos balanceados en nuestro DataFrame global
        data = pd.concat([X_res, y_res], axis=1)

    except Exception as e:
        print("Error en el under/oversampling de los datos")
        print(e)
        exit(1)
  

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
        8. Realizamos Oversampling o Undersampling
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """

    global data

    # Si sentiment analysis, forzar el score a numérico, y convertir en null lo que no lo sea
    # Evitar fallos de comas en el texto.
    # Traducir score a "positivo", "negativo", "neutro" si se trata con sentiment analysis
    if args.sentiment:
        data[args.prediction] = pd.to_numeric(data[args.prediction], errors="coerce")
        data[args.prediction] = data[args.prediction].fillna(data[args.prediction].mode()[0])  # para clasificación
        data[args.prediction] = data[args.prediction].apply(map_sentiment)

    # Guardamos la target
    y = data[args.prediction]

    # Tratamos missing values de la target (si hace falta)
    if y.isnull().any() and not args.sentiment:
      y = y.fillna(y.mode()[0])  # para clasificación

    # Nos quedamos solo con features
    data = data.drop(columns=[args.prediction])

    # Borrar columnas no necesarias
    drop_features()

    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

    # Pasar los datos a categoriales a numéricos
    cat2num_cols = cat2num(categorical_feature)

    # Simplificamos el texto
    simplify_text(text_feature)

    # Reescalamos los datos numéricos
    reescaler(numerical_feature)
    
    # Tratamos el texto
    text_cols, vectorizer = process_text(text_feature)

    # devolvemos a data los valores del target, solo habiendo procesado missing values (evitar errores)
    data[args.prediction] = y
    
    # Realizamos Oversampling o Undersampling
    over_under_sampling()

    return data, text_cols, cat2num_cols, vectorizer

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
 
 
def save_model(gs, vectorizer=None, text_columns=None, cat2num_cols=None):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        nombre = "modelo"

        modelo_completo = {
            "gs": gs,
            "vectorizer": vectorizer,
            "text_columns": text_columns,
            "cat2num_cols": cat2num_cols
        }
        with open(f'mejor-output-sin-oversampling/{nombre}.pkl', 'wb') as file:
            pickle.dump(modelo_completo, file)
            print(Fore.CYAN + "Modelo: "+ f"{nombre}" +" guardado con éxito" + Fore.RESET)

        with open('mejor-output-sin-oversampling/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])

    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """



    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def calculate_classification_report(y_true, y_pred):
    """
    Genera un informe de texto con Precision, Recall y F1 para cada clase.
    """
    #Hacer el clasification report
    cr = classification_report(y_true, y_pred, zero_division=0)
    with open('mejor-output-sin-oversampling/classification_report_train.txt', 'w') as f:
        f.write(cr)

    return cr

def calculate_confusion_matrix(y_true, y_pred):
    """
    Genera la matriz de confusión para ver dónde se equivoca el modelo.
    """
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm)
    df_cm.to_csv('output/matriz_confusion_train.csv', index=False)

    return cm

def calculate_fscore(y_true, y_pred):
    """
    Calcula el F1-Score en sus variantes Micro y Macro.
    :param y_true: Etiquetas reales del examen.
    :param y_pred: Etiquetas que ha adivinado el modelo.
    :return: Tupla con (f1_micro, f1_macro)
    """
    # Calculamos ambas versiones
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    return f1_micro, f1_macro

def kNN(vectorizer=None, text_cols=None, cat2num_cols=None):
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros

    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(KNeighborsClassifier(), args.kNN, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs,vectorizer,text_cols, cat2num_cols)

def decision_tree(vectorizer=None, text_cols=None, cat2num_cols=None):
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:

        dt = DecisionTreeClassifier(random_state=42)

        gs = GridSearchCV(estimator=dt,
                            param_grid=args.decision_tree,
                             cv=5,
                            n_jobs=args.cpu,
                            scoring=args.estimator)

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

    execution_time = end_time - start_time
    print("Tiempo de ejecución: " + Fore.MAGENTA + str(execution_time) + Fore.RESET + " segundos")

    # Mostramos los resultados en los datos de examen (Dev)
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo ganador en el disco duro
    save_model(gs, vectorizer, text_cols, cat2num_cols)
    
def random_forest(vectorizer=None, text_cols=None, cat2num_cols=None):
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """

    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()

    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:

        # 1. Instanciamos el modelo base (El Bosque)
        rf = RandomForestClassifier(random_state=42, n_jobs = 1)

        # 2. Configuramos la Búsqueda en Cuadrícula leyendo args.random_forest del JSON
        # Si no es análisis de sentiment, utilizar GridSearch para mejores resultados
        if not args.sentiment:
            gs = GridSearchCV(estimator=rf,
                              param_grid=args.random_forest,
                              cv=5,
                              n_jobs=args.cpu,
                              scoring=args.estimator,
                              verbose=2)
        # Para sentiment analisis con muchos datos, utilizar randomized search
        # 10% - 20% peores resultados que GridSearch, pero la velocidad se vuelve asequible
        else:
            gs = RandomizedSearchCV(estimator=rf,
                              param_distributions=args.random_forest,
                              n_iter=30,
                              cv=5,
                              n_jobs=args.cpu,
                              scoring=args.estimator,
                              verbose=2,
                              random_state=42)

        # 3. Entrenamos midiendo el tiempo
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        # Actualizamos la barra de progreso
        pbar.update(100)

    execution_time = end_time - start_time
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    print("Tiempo de ejecución: " + Fore.MAGENTA + str(execution_time) + Fore.RESET + " segundos")

    # Guardamos el modelo utilizando pickle
    save_model(gs, vectorizer, text_cols, cat2num_cols)


def naive_bayes(vectorizer=None, text_cols=None, cat2num_cols=None):
    """
    Función para implementar el algoritmo Naive Bayes.
    """

    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()

    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando Naive Bayes', unit='iter', leave=True) as pbar:

        #Si hacemos sentiment analysis, procesamos texto, asi que usamos MultinomialNB
        if args.sentiment:
             gs = GridSearchCV(MultinomialNB(), args.naive_bayes["multinomialnb"], cv=5, n_jobs=args.cpu, scoring=args.estimator)
        else:
            # Usamos args.naive_bayes porque en parse_args() tu grupo vuelca el JSON en args
            gs = GridSearchCV(GaussianNB(), args.naive_bayes["gaussiannb"], cv=5, n_jobs=args.cpu, scoring=args.estimator)

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        # Efecto visual de la barra de carga
        for i in range(100):
            time.sleep(random.uniform(0.01, 0.05))
            pbar.update(random.random() * 2)
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, f"{execution_time:.4f}", Fore.RESET + " segundos")

    # 3. Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

    # 4. Guardamos el modelo
    save_model(gs, vectorizer, text_cols, cat2num_cols)

def logistic_regression(vectorizer=None, text_cols=None, cat2num_cols=None):
    """
    Función para implementar Logistic Regression con búsqueda de hiperparámetros.
    """

    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()

    with (tqdm(total=100, desc='Procesando Logistic Regression', unit='iter', leave=True) as pbar):

        # Modelo base
        lr = LogisticRegression(max_iter=1000)

        #Grid de hiperparámetros
        param_grid = {
            "C": [0.01, 0.1, 1, 3],
            "penalty": ["l2"],  # l1 también posible, pero más lento
            "solver": ["lbfgs"],  # rápido para l2
            "class_weight": [None, "balanced"]
        }

        # Para sentiment → mejor Randomized o grid pequeño
        if args.sentiment:
            gs = RandomizedSearchCV(
                estimator=lr,
                param_distributions=param_grid,
                n_iter=10,
                cv=5,
                n_jobs=args.cpu,
                scoring=args.estimator,
                verbose=2,
                random_state=42
            )
        else:
            gs = GridSearchCV(
                estimator=lr,
                param_grid=param_grid,
                cv=5,
                n_jobs=args.cpu,
                scoring=args.estimator,
                verbose=2
            )

        # Entrenamiento
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        pbar.update(100)

    execution_time = end_time - start_time

    print("Tiempo de ejecución: " + Fore.MAGENTA + str(execution_time) + Fore.RESET + " segundos")

    # Resultados
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardar modelo
    save_model(gs, vectorizer, text_cols, cat2num_cols)

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
        os.makedirs('mejor-output-sin-oversampling')
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

    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")

    datos, text_cols, cat2num_cols, vectorizer = preprocesar_datos()

    #Para comprobar si todavia quedan variables categoricas o textuales a parte del target (debug)
    #print(data.select_dtypes(include=['object']).columns)

    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)

    # Ejecutamos el algoritmo seleccionado
    print("\n- Ejecutando algoritmo...")
    if args.algorithm == "kNN":
        try:
            kNN(vectorizer, text_cols, cat2num_cols)
            print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "decision_tree":
        try:
            decision_tree(vectorizer, text_cols, cat2num_cols)
            print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "random_forest":
        try:
            random_forest(vectorizer, text_cols, cat2num_cols)
            print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "naive_bayes":
        try:
            naive_bayes(vectorizer, text_cols, cat2num_cols)
            print(Fore.GREEN + "Algoritmo Naive Bayes ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "logistic_regression":
        try:
            logistic_regression(vectorizer, text_cols, cat2num_cols)
            print(Fore.GREEN + "Algoritmo Logistic Regression ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)

    else:
        print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
        sys.exit(1)
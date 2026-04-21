# -*- coding: utf-8 -*-
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

# Scikit-Learn: Herramientas de modelado y preprocesamiento
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# NLTK e Imblearn: Texto y balanceo de clases
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm


def signal_handler(sig, frame):
    # Gestiona la salida limpia con Ctrl+C
    print("\nSaliendo del programa...")
    sys.exit(0)


def parse_args():
    # Configura los argumentos de entrada por terminal.
    # Aquí puedes añadir nuevos argumentos si el examen pide flags adicionales.
    parse = argparse.ArgumentParser(description="Entrenamiento de modelos.")
    parse.add_argument("-f", "--file", required=True, help="Archivo de datos")
    parse.add_argument("-j", "--json", required=True, help="Configuración JSON")
    parse.add_argument("-a", "--algorithm", required=True, help="Algoritmo a usar. Poner: decision_tree, kNN, random_forest, naive_bayes" )
    parse.add_argument("-p", "--prediction", required=True, help="Columna target (Ej: C5) (Empieza en C1")
    parse.add_argument("-v", "--verbose", action="store_true")
    parse.add_argument("-nh", "--no_header", action="store_true")
    parse.add_argument("-t", "--task", choices=['C', 'R'], required=True ,help="Tipo de tarea. C = clasificacion ; R = regresion")
    parse.add_argument("-c", "--cpu", default=-1, type=int)  # -1 usa todos los hilos del procesador
    parse.add_argument("-e", "--estimator", default=None)

    args = parse.parse_args()
    # Carga la configuración del JSON y la inyecta en el objeto args
    with open(args.json) as f:
        config = json.load(f)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def load_data(file):
    # Carga CSV o Excel.
    # Para variar: cambia 'header=0' si quieres forzar nombres específicos o
    # modifica el encoding si el archivo tiene caracteres raros (encoding='latin1').
    global args
    try:
        ext = os.path.splitext(file)[1].lower()
        if ext == '.csv':
            data = pd.read_csv(file, header=0 if not args.no_header else None)
        elif ext in ['.xlsx', '.xls']:
            data = pd.read_excel(file, header=0 if not args.no_header else None)
        else:
            data = pd.read_csv(file)

        # ESTRATEGIA C1, C2...: Neutraliza nombres para que el Test no falle por typos.
        data.columns = [f"C{i + 1}" for i in range(len(data.columns))]
        print(Fore.CYAN + f"Aviso: Columnas renombradas de C1 a C{len(data.columns)}" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + f"Error al cargar: {e}" + Fore.RESET)
        sys.exit(1)


def select_features():
    # Separa columnas por tipo.
    # VARIAR: El 'threshold' (umbral) define cuándo algo es categoría o texto largo.
    # Si una columna tiene más de 10 valores únicos (por defecto), se tratará como texto.
    try:
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])
        categorical_feature = data.select_dtypes(include='object')

        threshold = args.preprocessing.get("unique_category_threshold", 10)
        # Filtramos: si tiene pocos valores únicos -> Categoría. Si tiene muchos -> Texto.
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= threshold]
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        # GUARDADO DEL MAPA: Crucial para que test.py sepa qué columna es qué.
        col_map = {"num": list(numerical_feature.columns), "cat": list(categorical_feature.columns),
                   "txt": list(text_feature.columns)}
        with open('output/col_types.json', 'w') as f:
            json.dump(col_map, f)

        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        sys.exit(1)


def process_missing_values(num, cat):
    # Rellena huecos (NaN).
    # VARIAR: Cambia 'mean' por 'median' o 'most_frequent' en el JSON según el tipo de dato.
    global data
    strategy_num = args.preprocessing.get("imputer_num", "mean")
    strategy_cat = args.preprocessing.get("imputer_cat", "most_frequent")
    if len(num.columns) > 0:
        imp = SimpleImputer(strategy=strategy_num)
        data[num.columns] = imp.fit_transform(data[num.columns])
    if len(cat.columns) > 0:
        imp = SimpleImputer(strategy=strategy_cat)
        data[cat.columns] = imp.fit_transform(data[cat.columns])


def reescaler(num):
    # Ajusta la escala de los números (importante para kNN y Naive Bayes).
    # VARIAR: Usa 'minmax' para rangos 0-1 o 'standard' para media 0 y varianza 1.
    global data
    if len(num.columns) > 0:
        tipo = args.preprocessing.get("scaler", "standard").lower()
        if tipo == "none": return

        scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler(),
                   "maxabs": MaxAbsScaler()}
        scaler = scalers.get(tipo, StandardScaler())

        data[num.columns] = scaler.fit_transform(data[num.columns])
        with open('output/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)  # Guardamos para el test.py


def cat2num(cat):
    # Convierte palabras en números.
    # VARIAR: 'ordinal' asigna 1, 2, 3... 'onehot' crea columnas nuevas (0/1).
    global data
    if len(cat.columns) > 0:
        est = args.preprocessing.get("categorical_to_num", "none")
        if est == "ordinal":
            # unknown_value=-1 permite que el test no pete si ve una categoría nueva.
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            data[cat.columns] = enc.fit_transform(data[cat.columns].astype(str))
            with open('output/encoder.pkl', 'wb') as f:
                pickle.dump(enc, f)
        elif est == "onehot":
            data = pd.get_dummies(data, columns=cat.columns, drop_first=True)


def simplify_text(txt):
    # Limpieza de texto: minúsculas, quitar puntuación y stopwords, y stemming.
    # VARIAR: Si quieres mantener las stopwords, comenta la línea del filtrado 'clean_tokens'.
    global data
    try:
        if len(txt.columns) > 0:
            stop_words = set(stopwords.words('spanish'))
            stemmer = PorterStemmer()
            for col in txt.columns:
                data[col] = data[col].astype(str).str.lower().str.translate(str.maketrans('', '', string.punctuation))

                def clean_sentence(text):
                    tokens = word_tokenize(str(text))
                    clean_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
                    return " ".join(clean_tokens)

                data[col] = data[col].apply(clean_sentence)
    except Exception as e:
        sys.exit(1)


def process_text(txt):
    # Vectorización de texto (TF-IDF o BoW).
    # VARIAR: TF-IDF da peso a palabras raras, BoW solo cuenta apariciones.
    global data
    if len(txt.columns) > 0:
        proc = args.preprocessing.get("text_process", "none")
        if proc == "tf-idf":
            vec = TfidfVectorizer()
        elif proc == "bow":
            vec = CountVectorizer()
        else:
            return

        combined = data[txt.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        matrix = vec.fit_transform(combined)
        text_df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())
        data = pd.concat([data.reset_index(drop=True), text_df], axis=1).drop(columns=txt.columns)

        with open(f'output/{proc}.pkl', 'wb') as f:
            pickle.dump(vec, f)


def over_under_sampling():
    # Balanceo de clases.
    # VARIAR: 'oversampling' duplica minoría, 'undersampling' borra mayoría.
    global data
    est = args.preprocessing.get("sampling", "none").lower()
    if est == "none": return
    X = data.drop(columns=[args.prediction])
    y = data[args.prediction]
    sampler = RandomUnderSampler() if est == "undersampling" else RandomOverSampler()
    X_res, y_res = sampler.fit_resample(X, y)
    data = pd.concat([X_res, y_res], axis=1)


def preprocesar():
    # Orquestador del preprocesamiento.
    global data
    y = data[args.prediction]
    if y.isnull().any(): y = y.fillna(y.mode()[0])  # Relleno rápido del target si hay nulos
    data = data.drop(columns=[args.prediction])

    num, txt, cat = select_features()
    process_missing_values(num, cat)
    cat2num(cat)
    simplify_text(txt)
    reescaler(num)
    process_text(txt)
    over_under_sampling()

    data[args.prediction] = y
    return data


def mostrar_y_guardar(gs, x_dev, y_dev):
    # Evalúa el modelo en el conjunto de validación y guarda el .pkl final.
    y_pred = gs.predict(x_dev)

    if args.verbose:
        print(Fore.MAGENTA + "Mejores Parámetros:" + Fore.RESET, gs.best_params_)

        if args.task == 'R':
            r2 = r2_score(y_dev, y_pred)
            mse = mean_squared_error(y_dev, y_pred)
            mae = mean_absolute_error(y_dev, y_pred)

            print(Fore.CYAN + "R2 Score (Precisión):" + Fore.RESET, r2)
            print(Fore.CYAN + "Error Cuadrático Medio (MSE):" + Fore.RESET, mse)
            print(Fore.CYAN + "Error Absoluto Medio (MAE):" + Fore.RESET, mae)

            # GUARDAR MÉTRICAS DE REGRESIÓN EN CSV
            df_metricas = pd.DataFrame([{'R2_Score': r2, 'MSE': mse, 'MAE': mae}])
            df_metricas.to_csv('output/metricas_train_regresion.csv', index=False)

        else:
            print(classification_report(y_dev, y_pred))

            # GUARDAR MÉTRICAS Y MATRIZ DE CONFUSIÓN EN CSV
            # 1. Guardar el reporte (Precision, Recall, F1)
            reporte_dict = classification_report(y_dev, y_pred, output_dict=True)
            df_reporte = pd.DataFrame(reporte_dict).transpose()
            df_reporte.to_csv('output/metricas_train_clasificacion.csv')

            # 2. Guardar la Matriz de Confusión
            matriz = confusion_matrix(y_dev, y_pred)
            df_matriz = pd.DataFrame(matriz)
            df_matriz.to_csv('output/matriz_confusion_train.csv', index=False)

    # Guarda las predicciones fila por fila
    res = x_dev.copy()
    res['Real'] = y_dev
    res['Pred'] = y_pred
    res.to_csv(f'output/val_pred_{args.algorithm}.csv', index=False)

    # Guarda el cerebro del modelo
    with open('output/modelo.pkl', 'wb') as f:
        pickle.dump(gs.best_estimator_, f)


def ejecutar_grid(model, params, name):
    X = data.drop(columns=[args.prediction])
    y = data[args.prediction]

    # Estratificamos SOLO si es clasificación
    estratificar = y if args.task == 'C' else None
    x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42, stratify=estratificar)

    with tqdm(total=100, desc=f'Procesando {name}', unit='iter', leave=True) as pbar:
        #  Cambiamos cómo se evalúa el GridSearchCV según la tarea
        scoring_metric = args.estimator
        if scoring_metric is None:
            scoring_metric = 'neg_mean_squared_error' if args.task == 'R' else 'f1_macro'

        gs = GridSearchCV(model, params, cv=5, n_jobs=args.cpu, scoring=scoring_metric)

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        for i in range(10):
            time.sleep(0.05)
            pbar.update(10)

    print(f"Tiempo de ejecución: {Fore.MAGENTA}{end_time - start_time:.4f}{Fore.RESET} segundos")
    mostrar_y_guardar(gs, x_dev, y_dev)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    if not os.path.exists('output'): os.makedirs('output')

    data = load_data(args.file)
    nltk.download(['stopwords', 'punkt', 'punkt_tab', 'wordnet'], quiet=True)

    print("- Preprocesando...")
    preprocesar()

    if args.task == 'R':
        algos = {
            "kNN": (KNeighborsRegressor(), args.kNN),
            "decision_tree": (DecisionTreeRegressor(random_state=42), args.decision_tree),
            "random_forest": (RandomForestRegressor(random_state=42), args.random_forest),
            "naive_bayes": (None, None)  # Naive Bayes no tiene regresor estándar aquí
        }
    else:
        algos = {
            "kNN": (KNeighborsClassifier(), args.kNN),
            "decision_tree": (DecisionTreeClassifier(random_state=42), args.decision_tree),
            "random_forest": (RandomForestClassifier(random_state=42), args.random_forest),
            "naive_bayes": (GaussianNB(), args.naive_bayes)
        }

    if args.algorithm in algos:
        model, params = algos[args.algorithm]
        if model is None:
            print(Fore.RED + f"Error: El algoritmo {args.algorithm} no soporta el modo {args.task}." + Fore.RESET)
        else:
            ejecutar_grid(model, params, args.algorithm)
    else:
        print(Fore.RED + "Algoritmo no reconocido." + Fore.RESET)
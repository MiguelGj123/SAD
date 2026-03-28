# -*- coding: utf-8 -*-
import sys
import os
import pickle
import json
import pandas as pd
import numpy as np
import string
import signal
import argparse
from colorama import Fore
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer


def signal_handler(sig, frame):
    sys.exit(0)


def parse_args():
    # VARIAR: Aquí se gestionan los argumentos.
    # Asegúrate de pasar siempre el -m (modelo) y el -j (json) correctos.
    parse = argparse.ArgumentParser(description="Fase de testeo.")
    parse.add_argument("-f", "--file", required=True)
    parse.add_argument("-m", "--model", required=True)
    parse.add_argument("-j", "--json", required=True)
    parse.add_argument("-p", "--prediction", required=True)
    parse.add_argument("-v", "--verbose", action="store_true")
    parse.add_argument("-nh", "--no_header", action="store_true")
    args = parse.parse_args()

    with open(args.json) as f:
        config = json.load(f)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def load_data(file):
    # CARGA DE DATOS: Debe ser idéntica al Train.
    # VARIAR: Si el profe te da un archivo sin títulos, USA EL FLAG -nh en la terminal.
    global args
    try:
        ext = os.path.splitext(file)[1].lower()
        h = None if args.no_header else 0

        if ext == '.csv':
            data = pd.read_csv(file, header=h)
        elif ext in ['.xlsx', '.xls']:
            data = pd.read_excel(file, header=h)
        else:
            data = pd.read_csv(file, header=h)

        # Mantenemos la estrategia C1, C2... para que los nombres coincidan con el modelo.
        data.columns = [f"C{i + 1}" for i in range(len(data.columns))]
        print(Fore.CYAN + f"Aviso: Datos neutralizados a C1-C{len(data.columns)}" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + f"Error al cargar: {e}" + Fore.RESET)
        sys.exit(1)


def select_features():
    # CONSISTENCIA: No decidimos tipos de datos, los leemos del JSON que guardó el Train.
    # Esto evita que una columna de texto se confunda con una categoría si el test es corto.
    try:
        with open('output/col_types.json', 'r') as f:
            col_map = json.load(f)

        num_cols = [c for c in col_map['num'] if c in data.columns]
        cat_cols = [c for c in col_map['cat'] if c in data.columns]
        txt_cols = [c for c in col_map['txt'] if c in data.columns]

        return data[num_cols], data[txt_cols], data[cat_cols]
    except Exception as e:
        print(Fore.RED + "Error: No se encontró col_types.json en /output. ¿Has entrenado antes?" + Fore.RESET)
        sys.exit(1)


def process_missing_values(num, cat):
    # RELLENO DE NULOS: Usa la estrategia definida en el JSON.
    global data
    strategy_num = args.preprocessing.get("imputer_num", "mean")
    strategy_cat = args.preprocessing.get("imputer_cat", "most_frequent")

    if len(num.columns) > 0:
        imputer = SimpleImputer(strategy=strategy_num)
        data[num.columns] = imputer.fit_transform(data[num.columns])
    if len(cat.columns) > 0:
        imputer = SimpleImputer(strategy=strategy_cat)
        data[cat.columns] = imputer.fit_transform(data[cat.columns])


def reescaler(num):
    # ESCALADO: Cargamos el scaler.pkl.
    # IMPORTANTE: Usamos .transform(), NUNCA .fit(), porque las escalas ya se decidieron en el Train.
    global data
    if len(num.columns) > 0 and args.preprocessing.get("scaler") != "none":
        path = 'output/scaler.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
            data[num.columns] = scaler.transform(data[num.columns])
            print("Escalado aplicado desde pkl")


def cat2num(cat):
    # CATEGORÍAS: Cargamos el encoder.pkl.
    # VARIAR: Si usas 'onehot', se crearán columnas nuevas. Si usas 'ordinal', usa el pkl.
    global data
    if len(cat.columns) > 0:
        estrategia = args.preprocessing.get("categorical_to_num", "none")
        if estrategia == "ordinal":
            path = 'output/encoder.pkl'
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    enc = pickle.load(f)
                data[cat.columns] = enc.transform(data[cat.columns].astype(str))
                print("Codificación ordinal cargada")
            else:
                print(Fore.RED + "Error: No se encontró encoder.pkl" + Fore.RESET)
        elif estrategia == "onehot":
            data = pd.get_dummies(data, columns=cat.columns, drop_first=True)


def simplify_text(txt):
    # LIMPIEZA TEXTO: Debe ser idéntica al Train para que el vectorizador reconozca las palabras.
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
            print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
    except Exception as e:
        sys.exit(1)


def process_text(txt):
    # VECTORIZACIÓN: Carga tf-idf.pkl o bow.pkl.
    # IMPORTANTE: Usamos .transform() para que use el vocabulario que aprendió en el Train.
    global data
    proc = args.preprocessing.get("text_process", "none")
    if len(txt.columns) > 0 and proc != "none":
        path = f'output/{proc}.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                vec = pickle.load(f)
            combined = data[txt.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            matrix = vec.transform(combined)
            text_df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())
            data = pd.concat([data.reset_index(drop=True), text_df], axis=1).drop(columns=txt.columns)
            print(f"Texto procesado con {proc} cargado")


def preprocesar():
    # ORQUESTADOR: Prepara los datos para la predicción.
    global data
    # El target (y) solo se guarda si el archivo de test lo incluye para poder sacar métricas.
    y = data[args.prediction] if args.prediction in data.columns else None
    if y is not None: data = data.drop(columns=[args.prediction])

    num, txt, cat = select_features()
    process_missing_values(num, cat)
    cat2num(cat)
    simplify_text(txt)
    reescaler(num)
    process_text(txt)
    return y


def predict(y_true):
    # PREDICCIÓN FINAL: Carga el modelo y genera resultados.
    global data
    # SEGURO DE VIDA: Borramos cualquier columna que no sea número para que no pete el float.
    data = data.select_dtypes(include=[np.number])

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # ALINEACIÓN DE COLUMNAS: Si al Test le falta alguna columna que el modelo espera, la crea con ceros.
    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(data.columns)
        if missing:
            print(Fore.YELLOW + f"Aviso: Columnas faltantes {missing} rellenadas con 0" + Fore.RESET)
            for m in missing: data[m] = 0
        data = data[model.feature_names_in_]  # Reordenamos para que coincidan perfectamente

    pred = model.predict(data)

    print(Fore.MAGENTA + "Distribución de Predicciones:\n" + Fore.RESET, pd.Series(pred).value_counts())

    # MÉTRICAS: Solo se muestran si el verbose está activo y el archivo de test tenía la columna Real.
    if args.verbose and y_true is not None:
        print(Fore.CYAN + "F1 Macro:" + Fore.RESET, f1_score(y_true, pred, average='macro'))
        print(classification_report(y_true, pred))

    # GUARDADO: Generamos el CSV final con la predicción.
    if y_true is not None: data['Target_Real'] = y_true.values
    data['Prediccion'] = pred
    data.to_csv('output/data-prediction.csv', index=False)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    data = load_data(args.file)
    nltk.download(['stopwords', 'punkt', 'punkt_tab', 'wordnet'], quiet=True)
    y = preprocesar()
    predict(y)
    print(Fore.GREEN + "¡Listo! Resultados en output/data-prediction.csv" + Fore.RESET)
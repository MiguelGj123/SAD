# Plantillas de predicción - algoritmos de clasificación -
El proyecto consiste en dos archivos:
- train.py: dado un archivo con datos (archivo .csv) implementa modelos de clasificación (kNN, árboles de decisión, random forest y naive bayes) para hacer un barrido de hiperpárametros y generar un mejor modelo de predicción sobre una clase.
- test.py: a partir de un mejor modelo y un archivo con datos (archivos .pkl y .csv respectivamente), genera una predicción sobre una clase.

## Requisitos

- Python 3.13.12
- pip
- pandas
- scikit-learn
- pickle
- nltk
- imblearn
- tqdm
- colorama

Las dependencias específicas están listadas en 'requirements.txt'

## Instalación en Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Ejecución

### Para train.py (argumentos mínimos):
```bash
python train.py -f archivo.csv -j clasificador.json -a kNN -p filaObjetivo
```

### Para test.py (argumentos mínimos):
```bash
python test.py -f archivo.csv -m output/modelo.pkl -j clasificador.json -p filaObjetivo
```
Los nombres de los archivos (argumentos -f, -m, -j) pueden variar dependiendo del archivo que se quiera usar.
Sucede lo mismo con la columna/atributo a predecir (argumento -p) y con el algoritmo de predicción (argumento -a).

### Para más información:
```bash
python train.py -h
python test.py -h
```
version python utilizada: Python 3.13

#preparación de entorno

Crear un nuevo proyecto (en por ejemplo pycharm), elegir Interpreter type: Base Conda, importar plantilla completa

# Instrucciones de Ejecución

A continuación se detallan los comandos necesarios para ejecutar la fase de entrenamiento y la fase de testeo, junto con la explicación de todos los argumentos disponibles.

---

## 1. Fase de Entrenamiento (`train.py`)

Este script procesa los datos, entrena el algoritmo elegido y guarda el modelo final en la carpeta `output/`.

**Ejemplo de comando para Clasificación:**

python train.py -f datos.csv -j clasificador.json -a kNN -p "ColumnaObjetivo" -t C -v

Argumentos de train.py:
-f o --file (Obligatorio): Ruta del archivo de datos a utilizar (ej: iris.csv).

-j o --json (Obligatorio): Ruta del archivo de configuración JSON (ej: clasificador.json).

-a o --algorithm (Obligatorio): Algoritmo que se desea entrenar. Opciones válidas: kNN / decision_tree / random_forest / naive_bayes (Nota: Solo compatible con tareas de clasificación).

-p o --prediction (Obligatorio): Número de la columna que se quiere predecir, estructura: C + (num. columna) ; ejemplo: C3 / C8.

-t o --task (Opcional): Define el tipo de problema matemático. Opciones: C : clasificacion  o R: regresion.

-v o --verbose (Opcional): Activa la salida detallada por consola para mostrar las métricas finales (Matriz de Confusión, MSE, etc.).

-nh o --no_header (Opcional): Añadir este flag si el archivo CSV/Excel no contiene una primera fila con los nombres de las columnas.

-c o --cpu (Opcional): Número de hilos del procesador a usar para el GridSearch. Por defecto es -1 (usa todos los disponibles).

-e o --estimator (Opcional): Métrica específica para evaluar el mejor modelo (ej: f1_macro).

## 2. Fase de Testeo (`test.py`)

Este script carga el modelo entrenado (modelo.pkl) y evalúa un conjunto de datos nuevo para generar las predicciones.

Ejemplo de comando para Clasificación:
python test.py -f examen.csv -m output/modelo.pkl -j clasificador.json -p "ColumnaObjetivo" -t C -v

Argumentos de test.py:
-f o --file (Obligatorio): Ruta del archivo de datos de prueba.

-m o --model (Obligatorio): Ruta donde está guardado el modelo entrenado (por defecto: output/modelo.pkl).

-j o --json (Obligatorio): Ruta del archivo JSON que se usó durante el entrenamiento.

-p o --prediction (Obligatorio): Nombre de la columna objetivo. Si el archivo de test la incluye, el programa mostrará la nota sacada; si no, solo hará las predicciones.

-t o --task (Obligatorio): Debe coincidir con la tarea usada en el entrenamiento. Opciones: C: clasificacion  o R: regresion.

-v o --verbose (Opcional): Imprime las métricas finales por consola si el archivo contenía las soluciones reales.

-nh o --no_header (Opcional): Añadir si el archivo de test no tiene fila de cabeceras.
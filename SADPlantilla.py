import sys
import numpy as np
import pandas as pd
import sklearn as sk
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

# Guardar las filas y columnas del csv en unas variables, y mostrarlas por pantalla
preparation_steps = []

file = sys.argv[1]

ml_dataset = pd.read_csv(file)

dtypes = {col: str(dtype) for col, dtype in ml_dataset.dtypes.items()}
print(dtypes)

print('******************************************')
print ('\nBase data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
# Five first records",
ml_dataset.head(5)



# con ml_dataset[[...]] solo reordenamos las columnas sin perder los datos importados
#ml_dataset = ml_dataset[['Especie', 'Ancho de sepalo', 'Largo de sepalo', 'Largo de petalo', 'Ancho de petalo']]


# funcion para establecer los tipos de datos de cada columna
def prepare_dataset(df):
    # guardar en la variable df una copia del dataframe para no afectarle directamente
    df = df.copy()

    # Seleccionar SOLO las columnas de tipo "objeto" y "categoria" (las que son texto) y las recorre
    for col in df.select_dtypes(include=["object", "category"]):
        # Por cada columna que es texto, se establece su tipo como "String"
        df[col] = df[col].astype("string")

    # Seleccionar SOLO las columnas de tipo "datetime64" (las que son fechas) y las recorre
    for col in df.select_dtypes(include=["datetime64"]):
        # Por cada columna que es fecha, se establece su tipo como "int64" (numero entero)
        # Transformar fecha en entero da como resultado la cantidad de nanosegundos desde 1-1-1970 hasta la fecha transformada
        # Como nos devuelve nanosegundos, hay que dividir entre 10⁹ y así transformarlo en segundos
        df[col] = df[col].astype("int64") // 10 ** 9

    # Seleccionar SOLO las columnas de tipo "int64" y "float64" (las que son numericas) y las recorre
    for col in df.select_dtypes(include=["int64", "float64"]):
        # Por cada columna numérica, establecer su tipo como "float64" (numero decimal)
        df[col] = df[col].astype("float64")

    return df

#Ejecutar funcion para establecer los tipos de cada columna
ml_dataset = prepare_dataset(ml_dataset)

#Guardar en variables separadas las columnas de texto, las numericas y las de fecha
categorical_features = ml_dataset.select_dtypes(include=["object", "category"]).columns.tolist()

numerical_features = ml_dataset.select_dtypes(include=["int64", "float64"]).columns.tolist()

datetime_features = ml_dataset.select_dtypes(include=["datetime64"]).columns.tolist()

#Crear un hashmap donde se asocia cada predicción posible a un número
target_map = {

    'Iris-versicolor': 0, 'Iris-virginica': 1, 'Iris-setosa': 2
}
#Crear una columna "__target__" que contiene lo que "Especie" pero en valores numericos.
#Utiliza map(str) para transformar las categorias de especie a string (evitar errores).
#Utiliza map(target_map) para sustituir cada categoria por su numero adecuado.
ml_dataset['__target__'] = ml_dataset['Especie'].map(str).map(target_map)
#Borra la columna Especie, para quedarse solo con su version numerica.
del ml_dataset['Especie']


# Borrar las filas en las que hay un valor vacio en la columna "Especie".
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
# Convertir el tipo de datos de la columna "target" a "int64" para que no sean strings.
ml_dataset['__target__'] = ml_dataset['__target__'].astype(np.int64)

#Separar el dataset para que una parte se use para entrenar el modelo y la otra para probarlo.
#Se guarda en dos variables distintas: "train" y "test".
train, test = train_test_split(
    #dataset que vamos a separar
    ml_dataset,
    #El 80% de las filas se dedican a entrenar
    train_size=0.8,
    #Esta es la semilla. Si siempre es la misma, se separan los datos siempre igual, si no, es random.
    random_state=42,
    #Esto se usa si los datos estan desbalanceados, los balancea.
    stratify=ml_dataset["__target__"])

print ('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
print ('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))

# Este codigo sirve para sustituir valores vacios para el resto de columnas (las que no son Especie).
# Primero se define una lista de columnas cuyas filas queremos eliminar si tienen un valor vacio.
# La lista esta vacia, a si que ahora mismo se decide no eliminar ninguna fila.
drop_rows_when_missing = []
# Ahora se define la lista de columnas cuyas filas queremos sustituir por algo si tienen un valor vacio.
# A parte del nombre de la fila, se introduce el valor por el que se quiere sustituir. En este caso la media.
impute_when_missing = [
    {'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'},
    {'feature': 'Largo de sepalo', 'impute_with': 'MEAN'},
    {'feature': 'Largo de petalo', 'impute_with': 'MEAN'},
    {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]

# Bucle para recorrer todas las columnas cuyos valores faltantes queremos eliminar.
print('\n******* Reemplazar valores faltantes ********')
print('\n')
for feature in drop_rows_when_missing:
    # Por cada columna, guardamos en train las filas que no estan vacias.
    # Para eso, se accede a los datos de cada columna y se usa notnull para conseguir solo los valores no vacios.
    train = train[train[feature].notnull()]
    test = test[test[feature].notnull()]
    print('Dropped missing records in %s' % feature)

# Bucle para recorrer todas las columnas cuyos valores faltantes queremos sustituir.
for feature in impute_when_missing:
    # Por cada columna, comprobar por que valor se quiere sustituir el vacio.
    # IMPORTANTE: La mediana, moda, media... solo se calculan con train, porque test no debe saberlas.
    if feature['impute_with'] == 'MEAN':
        # Si se quiere sustituir por la media, guardar en una variable "v" la media de los valores de la columna.
        v = train[feature['feature']].mean()

    elif feature['impute_with'] == 'MEDIAN':
        # ''' guardar la mediana en v '''
        v = train[feature['feature']].median()

    elif feature['impute_with'] == 'CREATE_CATEGORY':
        # ''' guardar el texto "NULL_CATEGORY" en v, para saber que queremos sustituirlo por otra cosa '''
        v = 'NULL_CATEGORY'

    elif feature['impute_with'] == 'MODE':
        # ''' guardar la moda en v '''
        v = train[feature['feature']].value_counts().index[0]

    elif feature['impute_with'] == 'CONSTANT':
        # ''' guardar un valor constante que se haya introducido manualmente en v '''
        v = feature['value']
    # Para todas las columnas, se sustituye el contenido vacio por lo que se haya guardado en v
    train[feature['feature']] = train[feature['feature']].fillna(v)
    test[feature['feature']] = test[feature['feature']].fillna(v)

    print(f"Imputed missing values in feature {feature['feature']} with value {v}")

#Este codigo sirve para poner todos los datos numericos en la misma escala, darles la misma importancia.
#Se guarda en un hashmap, para cada columna, que metodo utilizar de escalado.
rescale_features = {
    'Ancho de sepalo': 'AVGSTD',
    'Largo de sepalo': 'AVGSTD',
    'Largo de petalo': 'AVGSTD',
    'Ancho de petalo': 'AVGSTD'}

print('\n******* Reescalar datos: ********')
print('\n')
#Recorrer el hashmap de columnas y metodos de reescalado, guardando cada cosa en una variable distinta.
for (feature_name, rescale_method) in rescale_features.items():
    #Por cada columna y metodo de escalado, si el metodo es minmax:
    if rescale_method == 'MINMAX':
        #Se guarda en "_min" el minimo valor de la columna, y en "_max" el maximo
        _min = train[feature_name].min()
        _max = train[feature_name].max()
        #Se guarda en "scale" la resta del maximo menos el minimo, y el minimo se guarda en "shift"
        scale = _max - _min
        shift = _min
    #Si el metodo no es minmax, se usa z-scale por defecto
    else:
        #Se guarda en "scale" la desviacion tipica de los valores de la columna
        #Se guarda en "mean" la media de los valores de la columna
        shift = train[feature_name].mean()
        scale = train[feature_name].std()
    #Si "scale" es cero significa que no varian mucho los valores de la columna y no hace falta reescalarlos
    if scale == 0.:
        #Se borra la columna del train y del test.
        del train[feature_name]
        del test[feature_name]
        print ('Feature %s was dropped because it has no variance' % feature_name)
    #Si hay varianza entre los valores de la columna:
    else:
        print ('Rescaled %s' % feature_name)
        #Se aplica la formula de escalado para train y para test, dependiendo de que metodo se haya usado
        train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
        test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

#Este codigo sirve para poder trabajar mas facil con los datos
#Guardar en x_ la columna "__target__" de train y de test (axis=1 es columna, =0 es fila).
X_train = train.drop('__target__', axis=1)
X_test = test.drop('__target__', axis=1)

#Guardar en y_ la columna de "__target" de train y de test pero en forma de lista.
y_train = np.array(train['__target__'])
y_test = np.array(test['__target__'])

from sklearn.neighbors import KNeighborsClassifier

#Guardar en clf el clasificador, es decir, la configuracion del algoritmo Knn que usaremos
clf = KNeighborsClassifier(
    #valor de la k, en este caso k = 5, se miran los 5 vecinos mas cercanos
    n_neighbors=5,
    #La importancia que se le da a cada instancia. "Uniform" significa que sera la misma para todos.
    #Se puede poner "distance" y asi los mas cercanos importan mas.
    weights='uniform',
    #La manera en la que se buscan los vecinos, con "auto", escoge aleatoriamente entre las tres opciones que te da.
    algorithm='auto',
    #Tamaño de las hojas del arbol de busqueda, es mejor dejarlo en 30 y ya.
    leaf_size=30,
    #Distancia que se va a utilizar. p = 2 es euclidiana, p = 1 manhatan.
    # tambien se podria poner metric="manhattan" o "euclidean" mas especifico.
    p=2)

#entrenar el modelo
clf.fit(X_train, y_train)

#Guardamos en _predictions la prediccion, que se hace sobre los datos de x_test
#Se usa x_test porque se le ha quitado la columna "target", y asi no conoce los resultados
_predictions = clf.predict(X_test)
#Con predict_proba se predice cuantas probabilidades tiene cada fila de ser cada clase.
# por ejemplo, para cada fila daria: %80 iris virginica, 15% versicolor y 5% setosa
_probas = clf.predict_proba(X_test)
#Con pd.Series se tranforma en una columna lo que se le pase como parametros.
# en este caso se guardan las predicciones, un indice para que mantengan su orden original, y el nombre de la columna.
predictions = pd.Series(data=_predictions, index=X_test.index, name='predicted_value')

#Crear un array con varios nombres de columnas para mostrar la probabilidad que hay de cada categoria.
cols = [
    u'probability_of_value_%s' % label
    for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
]
#Con pd.DataFrame se guardan en una tabla las probabilidades predecidas, con indice para mantener su orden original
# y se agregan los nombres de las columnas con cols
probabilities = pd.DataFrame(data=_probas, index=X_test.index, columns=cols)

# Ahora se unen todos los resultados en results_test. Primero se une a X_test las predicciones.
#"left" quiere decir que se agregan a la izquierda de la tabla.
results_test = X_test.join(predictions, how='left')
# A x_test con las predicciones se unen las predicciones de probabilidad.
results_test = results_test.join(probabilities, how='left')
# A x_test con las predicciones y probabilidades se le agrega los valores reales que tenia el test originalmente.
results_test = results_test.join(test['__target__'], how='left')
#Renombrar la columna con los valores originales de test a "Especie"
results_test = results_test.rename(columns= {'__target__': 'Especie'})

#imprimir por pantalla los primeros resultados.
results_test.head()

#Importamos de sckit learn para calcular la accuracy, precision, reacall y f-score tanto macro como micro
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Accuracy. Se le pasa primero los valores reales (iris setosa, virginica, o versicolor) y se comparan con las predicciones.
accuracy = accuracy_score(y_test, _predictions)

# Precision
precision_macro = precision_score(y_test, _predictions, average='macro')
precision_micro = precision_score(y_test, _predictions, average='micro')

# Recall
recall_macro = recall_score(y_test, _predictions, average='macro')
recall_micro = recall_score(y_test, _predictions, average='micro')

# F1-score
f1_macro = f1_score(y_test, _predictions, average='macro')
f1_micro = f1_score(y_test, _predictions, average='micro')

print('\n******* Stats:  ********')
print('\n')
# Imprimir resultados. Se pone ":" a las variables para aplicarles un formato.
# .3f indica que se usan solo los 3 dec
# imales de despues del punto.
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision Macro: {precision_macro:.3f}, Precision Micro: {precision_micro:.3f}")
print(f"Recall Macro: {recall_macro:.3f}, Recall Micro: {recall_micro:.3f}")
print(f"F1-score Macro: {f1_macro:.3f}, F1-score Micro: {f1_micro:.3f}")

#Crear un mapa invertido para poder pasar las predicciones del modelo de numero a texto otra vez.
#Es decir: 0 --> versicolor, 1 --> virginica, 2 --> setosa
inv_map = { target_map[label] : label for label in target_map}
#Imprime las predicciones con su indice, para saber cual es la fila que se ha predecido
predictions.map(inv_map)
import warnings
from typing import Any

# Ignoramos warnings de deprecación internos de pyLDAvis y pandas para mantener la consola limpia
warnings.filterwarnings('ignore', category=DeprecationWarning)


def export_interactive_lda_html(lda_model: Any, bag_of_words_matrix: Any, vectorizer: Any, filepath: str,
                                mds_method: str = 'tsne') -> bool:
    """
    Genera y guarda un panel interactivo HTML utilizando la librería pyLDAvis. Esta herramienta
    permite visualizar la distribución de tópicos y las palabras clave de forma dinámica.

    Args:
        lda_model (Any): Modelo LatentDirichletAllocation de scikit-learn ya entrenado.
        bag_of_words_matrix (Any): Matriz de conteos (frecuencias) utilizada en el entrenamiento.
        vectorizer (Any): El objeto CountVectorizer que contiene el vocabulario y mapeos.
        filepath (str): Ruta completa (incluyendo nombre de archivo y extensión .html) de destino.
        mds_method (str, opcional): Algoritmo de escalado multidimensional para representar los
                                    tópicos en 2D. Por defecto es 'tsne'.

    Returns:
        bool: Retorna True si el archivo se generó correctamente, o False si hubo un error
              (como la falta de la librería instalada).
    """
    try:
        # 1. Realizamos los imports de forma local dentro del try-except.
        # Esto evita que el programa principal falle si pyLDAvis no está instalado en el entorno.
        import pyLDAvis
        import pyLDAvis.lda_model

        # 2. Preparamos los datos para la visualización.
        # El método .prepare() recibe el modelo, la matriz de datos, el vectorizador y el método de reducción.
        # Calcula internamente las frecuencias de palabras y distancias entre tópicos.
        panel = pyLDAvis.lda_model.prepare(
            lda_model,
            bag_of_words_matrix,
            vectorizer,
            mds=mds_method
        )

        # 3. Exportamos el panel procesado a un archivo físico HTML en la ruta especificada.
        pyLDAvis.save_html(panel, filepath)

        return True

    except ImportError:
        # 4. En caso de que falle la importación (librería no presente), capturamos el error.
        # Devolvemos False para que el flujo principal pueda manejar la ausencia de la visualización.
        return False
import Funciones_generales as fg
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def knn():
    # Funcion para limpiar la pantalla 
    fg.Funciones.limpiar()

    # Valores de X
    #x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
    # Valores de y
    #y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    # Clases en las que esta cada punto
    #clases = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

    # Leemos el csv
    df = pandas.read_csv("Modelos_en_1/CSV/data2.csv")

    # Recogemos los datos de x, se tiene que cambiar a columna para la logistica
    x = np.array(df['Age'])
    y = np.array(df['Experience'])
    # Sabemos si van o no
    clases = np.array(df['Go'].map(dict(YES=1, NO=0)))

    # Metemos los datos en una tapla de datos
    data = list(zip(x, y))

    # Creamos el clasificador especificando la cantidad de vecinos que tiene que tener en cuenta
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(data, clases)

    # Introducimos un nuevo punto
    new_x = 33
    new_y = 13
    new_point = [(new_x, new_y)]

    # Predecimos donde se encuentra el nuevo punto
    prediccion = knn.predict(new_point)
    
    # Tenemos que añadir el nuevo punto a los antiguos para que aparezca
    x = np.append(x, [new_x])
    y = np.append(y, [new_y])
    clases = np.append(clases, [prediccion])

    plt.scatter(x, y, c=clases)
    plt.scatter(new_x, new_y, marker='x')
    plt.show()
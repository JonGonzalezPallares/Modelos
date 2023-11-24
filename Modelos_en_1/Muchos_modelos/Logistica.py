import Funciones_generales as fg
#import pandas
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

def logistica():
    # Funcion para limpiar la pantalla 
    fg.Funciones.limpiar()

    # Tamaño del tumor, se tiene que hacer el reshape para que funcione
    X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
    print(X)
    # Representa si el tumor es canceroso o no
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    #
    #
    # Haciendo con este sale mal
    #
    #

    # Leemos el csv
    #df = pandas.read_csv("Modelos_en_1/CSV/data2.csv")

    # Recogemos los datos de x, se tiene que cambiar a columna para la logistica
    #X = np.array(df['Age']).reshape(-1, 1)
    # Sabemos si van o no
    #y = np.array(df['Go'].map(dict(YES=1, NO=0)))

    # Generamos el modelo
    logr = linear_model.LogisticRegression()
    # Encagamos la logistica con nuestros datos
    logr.fit(X, y)

    def prob(x):
        w = logr.coef_
        b = logr.intercept_
        z = w * x + b
        return 1 / (1 + np.exp(-z))

    def prob1(x):
        log_odds = logr.coef_ * x + logr.intercept_
        odds = np.exp(log_odds)
        probability = odds / (1 + odds)
        return probability
    
    X = X.reshape(1, -1)
    # Array de puntos separados de manera equitativa en un rango
    arr = np.linspace(X.min(), X.max(), 50)

    # Modelo
    modelo = prob1(arr)
    f_mod = prob(arr)

    plt.scatter(X, y)
    plt.plot(arr, modelo[0])
    plt.plot(arr, f_mod[0])
    plt.xlabel("Edad")
    plt.ylabel("Va")
    plt.title("Logistica")
    plt.show()
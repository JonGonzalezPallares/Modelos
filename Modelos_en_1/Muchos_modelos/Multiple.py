import Funciones_generales as fg
import pandas
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def multiple():
    # Funcion para limpiar la pantalla 
    fg.Funciones.limpiar()
    # Leemos el CSV
    df = pandas.read_csv("Modelos_en_1/CSV/data2.csv")

    # Seleccionamos variables independientes
    X = df[['Age', 'Experience']]
    # Variable dependiente
    y = df['Rank']

    # Generamos un objeto generico de regresion lineal
    regresion = linear_model.LinearRegression()
    # Encagamos la regresion con nuestros datos
    regresion.fit(X, y)

    # Generamos "datos de testing"
    x_linea = np.linspace(X.Age.min(), X.Age.max(), 50)
    y_linea = np.linspace(X.Experience.min(), X.Experience.max(), 50)

    # Juntamos los datos de x e y a pares. Se lo pasamos para que genera la prediccion
    valores = regresion.predict(np.concatenate((x_linea.reshape(-1, 1), y_linea.reshape(-1, 1)), axis=1))

    # Generamos la figura en 3d
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Añadimos los datos datos, la linea y los titulos
    ax.scatter(X.Age, X.Experience, y)
    ax.plot(x_linea, y_linea, valores)
    ax.set(title="Multiple", xlabel="Age", ylabel="Experience", zlabel="Rank")

    plt.show()
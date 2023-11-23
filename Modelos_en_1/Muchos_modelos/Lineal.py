import Funciones_generales as fg
from scipy import stats
import pandas
import numpy as np
import matplotlib.pyplot as plt

def lineal():
    # Funcion para limpiar la pantalla 
    fg.Funciones.limpiar()
    
    # Leemos el csv de datos
    df = pandas.read_csv("Modelos_en_1/CSV/data2.csv")

    # Recogemos los datos que queramos
    X = np.array(df['Age'])
    y = np.array(df['Experience'])

    slope, intercept, r, p, std_err = stats.linregress(X, y)

    # Funcion con la ecuacion de la recta
    def recta(x):
        return slope * x + intercept
    
    # Array de puntos en X "datos de testing"
    x = np.linspace(X.min(), X.max(), 50)

    # Prediccion con datos de testing
    modelo = list(map(recta, x))

    # Dibujamos los puntos
    plt.scatter(X, y)

    # Dibujamos el modelo
    plt.plot(x, modelo)

    # Indicamos los ejes y titulo
    plt.title("Linear")
    plt.xlabel("Age")
    plt.ylabel("Experience")

    plt.show()    
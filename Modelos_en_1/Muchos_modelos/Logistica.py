import Funciones_generales as fg
import pandas
import numpy as np

def logistica():
    # Funcion para limpiar la pantalla 
    fg.Funciones.limpiar()

    # Leemos el csv
    df = pandas.read_csv("Modelos_en_1/CSV/data2.csv")

    # Recogemos los datos de x
    X = np.array(df['Age'])
    # Sabemos si van o no
    y = np.array(df['Go'].map(dict(YES=1, NO=0)))
    print(y)
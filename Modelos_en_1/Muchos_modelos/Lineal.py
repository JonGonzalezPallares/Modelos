import Funciones_generales as fg

import pandas

def lineal():
    # Funcion para limpiar la pantalla 
    fg.Funciones.limpiar()
    
    # Leemos el csv de datos
    df = pandas.read_csv("Modelos_en_1/CSV/data2.csv")

lineal()
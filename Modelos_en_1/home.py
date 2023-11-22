# Importamos todos los modelos
import Muchos_modelos.Lineal as Lineal
import Muchos_modelos.Multiple as Multiple
import Muchos_modelos.Logistica as Logistica
import Muchos_modelos.KNN as KNN
import Muchos_modelos.Decision_tree as Dt

import Funciones_generales as fg

# Funcion para las posibilidades
def posibilidades(tipo):
    # Si son opciones del menu
    if(tipo<6 and tipo>=0):
        if(tipo==1):
            Lineal.lineal()
        if(tipo==2):
            Multiple.multiple()
        if(tipo==3):
            Logistica.logistica()
        if(tipo==4):
            KNN.knn()
        if(tipo==5):
            Dt.decision_tree()
        elif(tipo==0):
            # Funcion para limpiar la pantalla 
            fg.Funciones.limpiar()
            exit()

    # Si no es una de las posibilidades
    else:
        print("Error, opcion no disponible\nVuelva a intentarlo")
        # Funcion para esperar
        fg.Funciones.espera()
        menu()


# Funcion que crea el menu inicial
def menu():
    # Funcion para limpiar la pantalla 
    fg.Funciones.limpiar()
    print("Selecciona uno de los modelos disponibles")
    
    # Seleccionar las opciones
    tipos = input("""
                  1: Regresion Lineal\n
                  2: Regresion Multiple\n
                  3: Regresion Logistica\n
                  4: KNN\n
                  5: Decision Tree\n
                  0: Salir\n""")
    
    # Si es numero se pasa a int para ver si esta en las opciones
    if(tipos.isdigit()):
        tipos=int(tipos)
    # Si no, se pasa a un valor que no esta en las posibilidades
    else:
        tipos=-1
    posibilidades(tipos)


menu()
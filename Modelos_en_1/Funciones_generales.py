import time
import os

# Funciones generales
class Funciones():
    # Funcion para esperar dos segundos
    def espera():
        # Espera 2 segundos
        time.sleep(2)
    
    # Funcion para limpiar la consola
    def limpiar():
        clear = lambda: os.system('clear')
        clear()
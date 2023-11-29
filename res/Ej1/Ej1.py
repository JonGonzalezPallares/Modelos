import numpy as np
import matplotlib.pyplot as plt

# IMPORTAR LIBRERÍAS NECESARIAS AQUÍ ######################
from sklearn.metrics import r2_score

###########################################################

X = np.arange(0, 20, 0.2)
y = np.cos(X)

# INSERTAR CÓDIGO AQUÍ ####################################

p = 0
paso = True
while(paso):
    model = np.poly1d(np.polyfit(X, y, p))
    print("R² (x):  ", r2_score(y, model(X)))
    if(r2_score(y, model(X))>0.9):
        paso = False
        break
    else:
        p = p+1
line = np.linspace(0, 20, num=1000)
plt.plot(line, model(line), label="Funcion", color="Red")
plt.scatter(X, y, label="Puntos")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Predicción valores de función coseno en un intervalo")
plt.legend(loc="upper right")

###########################################################
plt.show()

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Pandas para leer el csv
df = pd.read_csv("Prueba shop/shopping_trends.csv")

# Puntos de x
edad = df['Age'].head(20)
# Puntos de y
cant = df['Purchase Amount (USD)'].head(20)

# Para saber los diferentes datos
slope, intercept, r, p, std_err = stats.linregress(edad, cant)
print("Pendiente (m):......................", slope)
print("Condición inicial (y0):.............", intercept)
print("Tasa de Ajuste promedio (r):........", r)
print("Coeficiente de determinación (r²):..", r ** 2)

# Funcion para la ecuacion de la recta
def func(x):
    return slope * x + intercept

# Datos de testing
x = np.linspace(edad.min(), edad.max(), len(edad))

# Obtenes prediccion
model = list(map(func, x))

# Dibujamos los puntos
plt.scatter(edad, cant)

# Generamos la recta
plt.plot(x, model)

# Mostramos
plt.show()
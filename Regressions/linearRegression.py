#!venv/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# pares de puntos utilizados para generar / entrenar el modelo
x_points = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
y_points = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

slope, intercept, r, p, std_err = stats.linregress(x_points, y_points)

print("Pendiente (m):......................", slope)
print("Condición inicial (y0):.............", intercept)
print("Tasa de Ajuste promedio (r):........", r)
print("Coeficiente de determinación (r²):..", r ** 2)


# m = (y-y0) / (x-x0)   -->
# (y-y0) = m(x-x0)      -->
# y = m(x-x0) + y0      -->  \x0=0
# f(x) = y = mx + y0

# generamos una función con la ecuación de la recta
def func(x):
    return slope * x + intercept


# array de puntos en X para dibujar la recta (""datos de testing"")
x = np.linspace(x_points.min(), x_points.max(), 100)

# obtenemos la predicción con los ""datos de testing""
model = list(map(func, x))

# dibujamos por una lado los puntos
plt.scatter(x_points, y_points)

# y por otro lado la recta de previsión
plt.plot(x, model)

# indicamos los ejes y el título del gráfico
plt.title("Linear regression with SciPy")
plt.xlabel("X axis")
plt.ylabel("Y axis")

# desplegamos el gráfico
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model

# Ejercio de regresion lineal, polinomial (plot1) y multiple (plot2)
#-----------------------------------------------------------------------------------

# leer el csv
df = pd.read_csv("Cosas_clase/Salary_Data.csv").dropna().head(100)

# variables independientes, el head coge los 100 primeras lineas
X = np.array(df[['Years of Experience','Age']])

# variables dependientes, el head coge los 100 primeras lineas
y = np.array(df['Salary'])

# coge los valores desde la primera hasta la ultima fila de la comuna 0 ('Years of Experience')
x = X[:,0]

# sacar pendiente y punto corte en Y
slope, intercept, r, p, std_err = stats.linregress(x, y)

print("Pendiente (m):......................",slope)
print("Condición inicial (y0):.............",intercept)
print("Tasa de Ajuste promedio (r):........",r)
print("Coeficiente de determinación (r²):..", r ** 2)

# funcion de la recta de la regresion lineal
def myfunc(x):
  return slope * x + intercept

# mete en una lista todos los resultados de la regresion, es decir, de la recta
# model1 = list(map(myfunc,x))

model1 = list(myfunc(x))

# primer numero --> indica las filas
# segundo numero --> indica las columnas
# tercer numero --> indica en que numero se va a colocar

plt.subplot(3, 2, 1)
plt.scatter(x, y)
plt.plot(x, model1)
plt.title("Regresion lineal")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")

# regresion polinomial de grado 10
model2 = np.poly1d(np.polyfit(x, y, 10)) 

# regresion polinomial de grado 5
model3 = np.poly1d(np.polyfit(x, y, 5)) 

# dibuja la linea
line = np.linspace(x.min(), x.max(), num=1000)

plt.subplot(3, 2, 2)
plt.scatter(x, y)
plt.plot(x, model1, label="1 grado")
plt.plot(line, model2(line), label="10 grado")
plt.legend(loc="lower right")
plt.title("Regresion polinomial")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")

plt.subplot(3, 2, 5)
plt.scatter(x, y)
plt.plot(line, model3(line), label="5 grado")
plt.title("Regresion polinomial")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")

# regresion
regr = linear_model.LinearRegression()

# entrenamiento
regr.fit(X, y)

print(regr.predict(np.array([[5,6]])))

# aqui tienes un array de 1 fila y n columnas
x_exp = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x_age = np.linspace(X[:,1].min(), X[:,1].max(), 100)

# te convierte el array en 1 columna y n filas
re_exp = x_exp.reshape(-1, 1)
re_age = x_age.reshape(-1, 1)

# concatenamos los 2 arrays
arr = np.concatenate((re_exp,re_age), axis=1)

# la linea de grafico
pred = regr.predict(arr)

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(X[:,0], X[:,1], y)
ax.plot(x_exp, x_age, pred, color='red')
ax.set_xlabel('Experiencia')
ax.set_ylabel('Edad')
ax.set_zlabel('Salario')
plt.show()
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Cargar los datos
digits = load_digits()

# Generar datos aleatorios para el ejemplo
x = np.random.randint(1, 100, size=50)
y = np.random.randint(1, 100, size=50)
classes = np.random.randint(1, 4, size=50)

data = list(zip(x, y))

# Inicializar y entrenar el clasificador KNeighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data, classes)

# Crear un nuevo punto para la predicción
new_x = 50
new_y = 50
new_point = np.array([[new_x, new_y]])

# Realizar la predicción para el nuevo punto
prediction = knn.predict(new_point)
print(f'Predicción para el nuevo punto: {prediction[0]}')

# Visualizar los datos y el nuevo punto
for label in np.unique(classes):
    indices = np.where(classes == label)
    plt.scatter(x[indices], y[indices], label=f'Clase {label}')

plt.scatter(new_x, new_y, c=prediction, marker='x', label=f'Nuevo punto, clase: {prediction[0]}')
plt.legend(loc="upper right")
plt.show()

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

# Lee el archivo JSON
df = pd.read_json('Cosas_clase/Salary_data.json')

# Mapea las etiquetas de género a valores numéricos
d = {'Male': 0, 'Female': 1}
df['Gender'] = df['Gender'].map(d)

# Selecciona las primeras 100 filas para X
X = np.array(df['Salary'].head(100)).reshape(-1, 1)

# Selecciona las primeras 100 filas para y
y = np.array(df['Gender'].head(100)).astype(int)

# Crea una instancia del modelo de regresión logística
logr = LogisticRegression()

# Ajusta el modelo con los datos de entrenamiento
logr.fit(X, y)

# Realiza una predicción para un nuevo valor de X
predicted = logr.predict(np.array([[5000]]))  # Cambié el valor de entrada a 1 para Female
print(predicted)

# Crea un conjunto de prueba con valores de X para graficar la curva de probabilidad
X_test = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)

# Realiza predicciones de probabilidad en el conjunto de prueba
predict_proba = logr.predict_proba(X_test)[:, 1]

# Grafica la curva de probabilidad
plt.plot(X_test, predict_proba, color='green')

# Clasifica los puntos según sus etiquetas y colorea
plt.scatter(X[y == 0], y[y == 0], color='blue', label='Male')  
plt.scatter(X[y == 1], y[y == 1], color='red', label='Female')  

plt.legend()  # Muestra la leyenda con los colores
plt.xlabel('Gender (0: Male, 1: Female)')
plt.ylabel('Probability of being Female')
plt.title('Logistic Regression - Gender Classification')
plt.show()

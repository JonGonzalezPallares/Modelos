from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# Datos de entrada
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Etiquetas de salida (hombre o mujer)
Y = ['hombre', 'hombre', 'mujer', 'mujer', 'hombre', 'hombre', 'mujer', 'mujer', 'mujer', 'hombre', 'hombre']

# Mapeo de etiquetas a números
d = {'hombre': 1, 'mujer': 2}
y_nueva = [d[etiqueta] for etiqueta in Y]

# Crear DataFrame
X = pd.DataFrame(X, columns=['Altura', 'Peso', 'Zapato'])

# Inicializar y entrenar el clasificador
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y_nueva)

# Predecir para un nuevo punto
nuevo_punto = [[181, 80, 44]]
prediccion = dtree.predict(nuevo_punto)
print(f'Predicción para el nuevo punto: {prediccion[0]}')

# Visualizar el árbol de decisión
plt.figure(figsize=(15, 10))
tree.plot_tree(dtree, feature_names=X.columns)
plt.show()


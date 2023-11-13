import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import utils

# Cargamos los datos de iris

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print(X[:5])

print('\nClass labels:', np.unique(y))

# 80 % del conjunto de datos para entrenamiento y 20 % para validacion
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)
print('Numero de muestras en y:', np.bincount(y))
print('Numero de muestras en y_train:', np.bincount(y_train))
print('Numero de muestras en y_test:', np.bincount(y_test))

# Estandarizar los datos
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# Crear el modelo para ajustar

bosque = RandomForestClassifier(n_estimators=25,
                                criterion='entropy',
                                max_features='sqrt',
                                max_depth=10)

bosque.fit(X_train_std, y_train)

# Precision global de clasificación corecta
print('Train Accuracy : %.5f' % bosque.score(X_train_std, y_train))
print('Test Accuracy : %.5f' % bosque.score(X_test_std, y_test))


# Graficar Region de desición
X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

utils.plot_decision_regions(X_combined, y_combined,
                      classifier=bosque,
                      test_idx=range(105, 150))

plt.xlabel('Longitud de pétalo [cm]')
plt.ylabel('Ancho de pétalo [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/03_20.png', dpi=300)

y_pred = bosque.predict(X_test_std)
cm = confusion_matrix(y_test, y_pred, normalize='true')
cm_display = ConfusionMatrixDisplay(cm, display_labels=['I', 'S', 'V'])
cm_display.plot()
cm_display.ax_.set(title='RF2022', xlabel='Clases predichas', ylabel='Clases verdaderas')

plt.show()

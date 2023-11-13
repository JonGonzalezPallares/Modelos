from sklearn import datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import utils

# Loading the iris plants dataset (classification)


# creating dataframe of IRIS dataset
iris = datasets.load_iris()
data = pd.DataFrame({'petallength': iris.data[:, 2],
                     'petalwidth': iris.data[:, 3],
                     'species': iris.target})
# printing the top 5 datasets in iris dataset
print(data.head())

# dividing the datasets into two parts i.e. training datasets and test datasets
# Splitting arrays or matrices into random train and test subsets
# i.e. 70 % training dataset and 30 % test datasets
X = data[['petallength', 'petalwidth']]
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# metrics are used to find accuracy or error

# creating an RF classifier
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
rfc = RandomForestClassifier(n_estimators=20, criterion='entropy',
                             max_features='sqrt',
                             max_depth=10)
rfc.fit(X_train_std, y_train)

# performing predictions on the test dataset
y_pred = rfc.predict(X_test)
# using metrics module for accuracy calculation
# predicting which type of flower it is.
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

utils.plot_decision_regions(X_combined, y_combined,
                            classifier=rfc,
                      test_idx=range(105, 150))

plt.xlabel('Longitud de pétalo [cm]')
plt.ylabel('Ancho de pétalo [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/03_20.png', dpi=300)

y_pred = rfc.predict(X_test_std)
cm = confusion_matrix(y_test, y_pred, normalize='true')

cm_display = ConfusionMatrixDisplay(cm, display_labels=['I', 'S', 'V'])
cm_display.plot()
cm_display.ax_.set(title='RF2022', xlabel='Clases predichas', ylabel='Clases verdaderas')

plt.show()

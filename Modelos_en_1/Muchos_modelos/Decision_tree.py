import pandas
import Funciones_generales as fg
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def decision_tree():
    # Funcion para limpiar la pantalla 
    fg.Funciones.limpiar()

    # Leemos el csv de datos
    df = pandas.read_csv("Modelos_en_1/CSV/data2.csv")
    
    # todos los datos tienen que ser numéricos.
    # generamos una enumeración para la 'Nationality'
    # y modificamos los valores del dataframe
    d = {'UK': 0, 'USA': 1, 'N': 2}
    df['Nationality'] = df['Nationality'].map(d)
    # Lo mismo con 'Go'
    d = {'YES': 1, 'NO': 0}
    df['Go'] = df['Go'].map(d)

    # Identificamos las columnas 'feature' y 'target'
    features = ['Age', 'Experience', 'Rank', 'Nationality']

    X = df[features]
    y = df['Go']

    # Creamos el clasificador e introducimos los datos
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    
    tree.plot_tree(dtree, feature_names=features)
    plt
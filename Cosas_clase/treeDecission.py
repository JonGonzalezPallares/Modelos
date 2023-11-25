import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("Cosas_clase/data.csv")

# Coge los valores unicos
coche = df['Car'].unique()
modelo = df['Model'].unique()

#pone un numero a cada modelo o coche, empezando en 0 y va sumando +1
d1 = {car: index for index, car in enumerate(coche)}
d2 = {modelo: index for index, modelo in enumerate(modelo)}

df['Car'] = df['Car'].map(d1)
df['Model'] = df['Model'].map(d2)

print(df['Car'] )

features = ['Car','Model','Volume','Weight','CO2']

X = df[features]
y = df['Comprar']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

print(dtree.predict([[1,2,300,2342,129]]))

plt.subplots(figsize=(15, 10))

tree.plot_tree(dtree, feature_names=features)

plt.show()

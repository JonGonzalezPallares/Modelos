import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# Creamos nuestros datos artificiales, donde buscaremos clasificar
# dos anillos concéntricos de datos.
X, Y = make_circles(n_samples=500, factor=0.5, noise=0.05)

#Elevación al cuadrado
#X_power = X**2

# Resolución del mapa de predicción.
res = 100

# Coordendadas del mapa de predicción.
_x0 = np.linspace(-1.5, 1.5, res)
_x1 = np.linspace(-1.5, 1.5, res)

# Input con cada combo de coordenadas del mapa de predicción.
_pX = np.array(np.meshgrid(_x0, _x1)).T.reshape(-1, 2)

# Objeto vacio a 0.5 del mapa de predicción.
_pY = np.zeros((res, res)) + 0.5

# Visualización del mapa de predicción.
plt.figure(figsize=(8, 8))
plt.pcolormesh(_x0, _x1, _pY, cmap="coolwarm", vmin=0, vmax=1)

# Visualización de la nube de datos.
plt.scatter(X[Y == 0,0], X[Y == 0,1], c="skyblue")
plt.scatter(X[Y == 1,0], X[Y == 1,1], c="salmon")

plt.tick_params(labelbottom=False, labelleft=False)

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from matplotlib import animation
from IPython.core.display import display, HTML

# Definimos los puntos de entrada de la red, para la matriz X e Y.
iX = tf.compat.v1.placeholder('float', shape=[None, X.shape[1]])
iY = tf.compat.v1.placeholder('float', shape=[None])

lr = 0.01  # learning rate
nn = [2, 16, 8, 1]  # número de neuronas por capa.
w = []
b = []
activaciones = [iX]
for i in range(1,len(nn)):

    W = tf.Variable(tf.compat.v1.random_normal([nn[i-1], nn[i]]), name=f'Weights_{i}')
# Bias de la primera capa oculta de neuronas
    b_var = tf.Variable(tf.compat.v1.random_normal([nn[i]]), name=f'bias_{i}')
    w.append(W)
    b.append(b_var)
#Estableciendo la Función de activación de la primera capa de neuronas
    activacion_capa = tf.nn.relu(tf.add(tf.matmul(activaciones[-1], W), b_var))

    activaciones.append(activacion_capa)

#l1 = activaciones[2]

W_last = tf.Variable(tf.compat.v1.random_normal([nn[-2], 1]), name='Weights_last')

b_last = tf.Variable(tf.compat.v1.random_normal([1]), name='bias_last')

activacion_capa = tf.add(tf.matmul(activaciones[-2], w[-1]), b[-1])

pY = tf.nn.sigmoid(activacion_capa)[:, 0]
# Evaluación de las predicciones.
loss = tf.compat.v1.losses.mean_squared_error(pY, iY)

# Definimos al optimizador de la red, para que minimice el error.
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

n_steps = 1000  # Número de ciclos de entrenamiento.

iPY = []  # Aquí guardaremos la evolución de las predicción, para la animación.

with tf.compat.v1.Session() as sess:
    # Inicializamos todos los parámetros de la red, las matrices W y b.
    sess.run(tf.compat.v1.global_variables_initializer())

    # Iteramos n pases de entrenamiento.
    for step in range(n_steps):

        # Evaluamos al optimizador, a la función de coste y al tensor de salida pY.
        # La evaluación del optimizer producirá el entrenamiento de la red. Elevada en este caso al cuadrado
        _, _loss, _pY = sess.run([optimizer, loss, pY], feed_dict={iX: X, iY: Y})

        # Cada 25 iteraciones, imprimimos métricas.
        if step % 25 == 0:
            # Cálculo del accuracy.
            acc = np.mean(np.round(_pY) == Y)

            # Impresión de métricas.
            print('Step', step, '/', n_steps, '- Loss = ', _loss, '- Acc =', acc)

            # Obtenemos predicciones para cada punto de nuestro mapa de predicción _pX. En este caso elevada al cuadrado
            _pY = sess.run(pY, feed_dict={iX: _pX}).reshape((res, res))

            # Y lo guardamos para visualizar la animación.
            iPY.append(_pY)
# ----- CÓDIGO ANIMACIÓN ----- #
ims = []

fig = plt.figure(figsize=(10, 10))

print("--- Generando animación ---")

for fr in range(len(iPY)):
    im = plt.pcolormesh(_x0, _x1, iPY[fr], cmap="coolwarm", animated=True)

    # Visualización de la nube de datos.
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c="skyblue")
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c="salmon")

    # plt.title("Resultado Clasificación")
    plt.tick_params(labelbottom=False, labelleft=False)

    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ani.save('animation_Sin_Elevar_al_Cuadrado2.mp4')


import tensorflow._api.v2.compat.v1 as tf

class NeuralNet:
# INSERTAR CÓDIGO AQUÍ SEGUNDA TAREA#######################
    def __init__(self):
        self.iX = 0
        self.iY = 0
        self.pY = 0
        self.loss = 0
        self.optimizer = 0
        pass

    def create_nn(self, topology_v, lr):
        self.iX = tf.compat.v1.placeholder('float', shape=[None, topology_v[0]])
        self.iY = tf.compat.v1.placeholder('float', shape=[None])
        w = []
        b = []
        activaciones = [self.iX]
        for i in range(1, len(topology_v)):
            W = tf.Variable(tf.compat.v1.random_normal([topology_v[i-1], topology_v[i]]), name=f'Weights_{i}')
            b_var = tf.Variable(tf.compat.v1.random_normal([topology_v[i]]), name=f'bias_{i}')
            w.append(W)
            b.append(b_var)
            activacion_capa = tf.nn.relu(tf.add(tf.matmul(activaciones[-1], W), b_var))
            activaciones.append(activacion_capa)
        activacion_capa = tf.add(tf.matmul(activaciones[-2], w[-1]), b[-1])
        self.pY = tf.nn.sigmoid(activacion_capa)[:, 0]
        self.loss = tf.compat.v1.losses.mean_squared_error(self.pY, self.iY)
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.loss)

        pass
    pass
###########################################################
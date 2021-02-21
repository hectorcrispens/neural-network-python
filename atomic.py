import numpy as np
import scipy as sc
import os
import time


from sklearn.datasets import make_circles

# CREAR DATASET
n = 10 # FILAS DEL DATASET
p = 2 # COLUMNAS DEL DATASET

xi = [[-1, 0],[1, 0], [-0.5, 0], [0.5, 0]]
yi = [0,0,1,1]

x, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
y = y[:, np.newaxis]

#x = np.array(xi)
#y = np.array(yi)

# # CLASE DE LA CAPA DE LA RED
class neural_layer():
    def __init__(self, w, b, act_f):
        self.act_f = act_f
        # vector de bias, se puede multiplicar porque type es numpy.ndarray
        # multiplica por 2 - 1, para que los valores de rand vayan de -1 < x < 1 pero el shape es [1, n_neur] y [n_conn, n_neur]
        self.b = b
        self.w = w
       



# # FUNCION DE ACTIVACION
sigm = (lambda x: 1 / (1 + np.e ** ( - x )),
        lambda x: x * (1 - x))

relu = lambda x: np.maximum(0, x) 


def _create_nn(topology, act_f):
    nn = []
    for l, value in enumerate(topology[:-1]):
        nn.append(neural_layer(np.random.randn(value,topology[l+1]), np.random.randn(1, topology[l+1]), act_f))
    return nn
        

def create_nn(act_f):

    nn = []
    
    #layer 1
    nn.append(neural_layer(w=np.array([[-0.13, -0.52], [-0.35, -0.18]]), b=np.array([-0.7, 0.3]),act_f=act_f))

    #layer 2
    nn.append(neural_layer(w=np.array([[-0.5, 0.2], [0.07, 0.55]]), b=np.array([0.55, -0.9]),act_f=act_f))

    #layer 3
    nn.append(neural_layer(w=np.array([-0.2, -0.6]), b=np.array([-0.4]),act_f=act_f))
    
    return nn

topology=[p, 4, 8, 4, 1]
neural_net = _create_nn(topology,sigm)
#print(red)
# ERROR CUADRATICO MEDIO
l2_cost = (lambda Yp, Yr: np.mean((Yp-Yr) ** 2),
           lambda Yp, Yr: (Yp-Yr))

def train(neural_net, x, y, l2_cost, rate = 0.5):

    out = [(None,x) ]
    zetas = [x]
    alphas = [None]
    #FORWARD PASS
    for l, layer in enumerate(neural_net):
        
        z = np.dot(out[-1][1], neural_net[l].w) + neural_net[l].b
        a = neural_net[l].act_f[0](z)       
        zetas.append(z)
        alphas.append(a)
        out.append((z,a))

    
    deltas = []
    # neural_net[0-6]
    for l in reversed(range(0, len(neural_net))):
        # los indices out[l+1] son porque en out[0] hemos guardado los datos que provienen del dataset
        z = out[l+1][0]
        a = out[l+1][1]
        
        # BACKWARD PASS
        if l == len(neural_net)-1:           
           # calcular delta en ultima capas (derivada del costo* derivada de la activacion)
           deltas.insert(0, l2_cost[1](a, y) * neural_net[l].act_f[1](a))

        else:
            deltas.insert(0, np.dot(deltas[0],_W.T) * neural_net[l].act_f[1](a))

        _W =  neural_net[l].w
        
        # GRADIENT DESCENT
        neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis = 0, keepdims=True) * rate
        neural_net[l].w = neural_net[l].w - np.dot(out[l][1].T,deltas[0]) * rate
    time.sleep(0.1)
    os.system('clear')
    print("\n Salida de la Red Neuronal")
    print(out[-1][1])
    print("\n Salida Esperada")
    print(y)

for g in range(9000):    
    train(neural_net, x, y, l2_cost, 0.9)

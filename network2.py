import numpy as np
import random
import json


class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []
        self.init()

    def init(self):
        self.biases = [np.random.randn(x, 1) for x in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def MGD(self, training_data, epochs, mini_batch_size, eta, lmbda, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, eta, lmbda, n)
            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {i} complete")

    def update_batch(self, mini_batch, eta, lmbda, train_size):
        n_batch = len(mini_batch)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [(1 - eta * lmbda / train_size) * w - (eta / n_batch) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / n_batch) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # Forward propagation
        activations = [x]
        activation = x
        zs = []
        nabla_w = []
        nabla_b = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward propagation
        delta = cost_derivative(activations[-1], y)
        nabla_w.append(np.dot(delta, activations[-2].transpose()))
        nabla_b.append(delta)
        for i in range(2, self.num_layers):
            sp = sigmoid_prime(zs[-i])
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            nabla_b.insert(0, delta)
            res = np.dot(delta, activations[-i-1].transpose())
            nabla_w.insert(0, res)
        return nabla_w, nabla_b




    def forward_prop(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        res = [(np.argmax(self.forward_prop(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in res)
    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net



def cost_derivative(a, y):
    return a - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

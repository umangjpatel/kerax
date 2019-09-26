import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from jax import grad, jit
from sklearn.metrics import accuracy_score
from tqdm import tqdm


@jit
def relu(x):
    return np.where(x > 0, x, 0.0)


@jit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@jit
def tanh(x):
    return np.tanh(x)


@jit
def binary_crossentropy(a, y):
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))


activation_dict = {'sigmoid': sigmoid, 'relu': relu, 'tanh': tanh}
loss_fn_dict = {'binary_crossentropy': binary_crossentropy}


class FC:

    def __init__(self, units, activation):
        self.units, self.activation = units, activation


class DNet:

    def __init__(self):
        self.layers = []

    @staticmethod
    def compute_activation(act, x):
        return activation_dict.get(act)(x)

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, epochs, lr=1e-3):
        self.loss_fn = loss_fn_dict.get(loss)
        self.epochs, self.alpha = epochs, lr

    def fit(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train
        self.m_train, self.nx = self.x_train.shape
        self.init_weights()
        self.gradient_descent()

    def init_weights(self):
        self.layers.insert(0, FC(units=self.nx, activation=None))
        self.weights = []
        key = random.PRNGKey(0)
        for i, layer in enumerate(self.layers[1:]):
            key, subkey = random.split(key)
            W = random.normal(subkey, shape=(self.layers[i].units, layer.units)) * 0.01
            b = np.zeros(layer.units)
            self.weights.append({'W': W, 'b': b})

    def compute_predictions(self, weights, inputs):
        A = inputs
        for i, layer_weights in enumerate(weights):
            W, b = layer_weights.get('W'), layer_weights.get('b')
            Z = np.dot(A, W) + b
            A = self.compute_activation(self.layers[i + 1].activation, Z)
        return A

    def compute_cost(self, weights, inputs, targets):
        preds = self.compute_predictions(weights, inputs)
        loss = self.loss_fn(a=preds, y=targets)
        return loss

    def gradient_descent(self):
        self.cost = []
        grad_fn = jit(grad(self.compute_cost))
        for _ in tqdm(range(self.epochs), desc="Training the model"):
            self.cost.append(self.compute_cost(self.weights, self.x_train, self.y_train))
            grads = grad_fn(self.weights, self.x_train, self.y_train)
            for i, grad_layer_weights in enumerate(grads):
                self.weights[i]['W'] -= self.alpha * grad_layer_weights['W']
                self.weights[i]['b'] -= self.alpha * grad_layer_weights['b']

    def plot_losses(self):
        plt.plot(range(self.epochs), self.cost, color='red')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def evaluate(self, x_test, y_test):
        preds = self.compute_predictions(self.weights, x_test)
        pred_labels = np.where(preds >= 0.7, 1, 0)
        accuracy = accuracy_score(y_test, pred_labels)
        print("Accuracy : {:0.2f}".format(accuracy))

    def predict(self, inputs):
        return self.compute_predictions(self.weights, inputs)
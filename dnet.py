import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from jax import grad, jit
from tqdm import tqdm

import nn.activations as activations
import nn.losses as losses
from nn.layers import FC


class DNet:

    def __init__(self):
        self.layers = []

    @staticmethod
    def compute_activation(act, x):
        return getattr(activations, act)(x)

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, epochs, lr=1e-3, weight_scale=0.01):
        self.loss_fn = getattr(losses, loss)
        self.epochs, self.alpha = epochs, lr
        self.weight_scale = weight_scale

    def fit(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train
        self.m_train, self.nx = self.x_train.shape
        self.init_network_params()
        self.gradient_descent()

    def init_network_params(self):
        self.layers.insert(0, FC(units=self.nx, activation=None))
        self.weights = []
        key = random.PRNGKey(0)
        for i, layer in enumerate(self.layers[1:]):
            key, subkey = random.split(key)
            W = random.normal(subkey, shape=(self.layers[i].units, layer.units)) * self.weight_scale
            b = np.zeros(layer.units)
            self.weights.append({'W': W, 'b': b})

    def compute_predictions(self, weights, inputs, train=True):
        A = inputs
        key = random.PRNGKey(0)
        for i, layer_weights in enumerate(weights):
            key, subkey = random.split(key)
            fc_layer = self.layers[i + 1]
            W, b = layer_weights.get('W'), layer_weights.get('b')
            Z = np.dot(A, W) + b
            A = self.compute_activation(fc_layer.activation, Z)
            A *= (random.bernoulli(subkey, fc_layer.keep_prob, shape=A.shape) / fc_layer.keep_prob) if train else 1.0
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
        preds = self.compute_predictions(self.weights, x_test, train=False).flatten()
        pred_labels = np.where(1 - preds > preds, 0, 1).flatten()
        return np.mean(pred_labels == y_test)

    def predict(self, inputs):
        return self.compute_predictions(self.weights, inputs, train=False)

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class DNet():

  def __init__(self):
    np.random.seed(42)
    self.layers = []
    global _model; _model = self

  def add(self, fc):
    self.layers.append(fc)

  def summary(self):
    for l, layer in enumerate(self.layers):
        print("Layer {} :=> units = {}, activation = {}".format(l+1, layer.units, layer.activation.act))
        print("Weights : {}, Bias : {}".format(layer.W.shape, layer.b.shape), "\n\n")

  def compile(self, epochs, lr):
    self.epochs, self.alpha = epochs, lr

  def fit(self, X_train, Y_train):
    self.X_train, self.Y_train = X_train, Y_train
    self.init_weights()
    self.init_comp_graph()

  def init_weights(self):
    self.nx, self.m_train = self.X_train.shape
    for i, layer in enumerate(self.layers):
      layer.W = np.random.randn(layer.units, self.nx) * 0.01 if i == 0 else np.random.randn(layer.units, self.layers[i-1].units)
      layer.b = np.random.randn(layer.units, 1) * 0.01

  def init_comp_graph(self):
    self.graph = CompGraph()
    self.graph.gradient_descent()
    self.cost = self.graph.cost

  def plot_losses(self):
    plt.plot(range(self.epochs), self.cost, color='red')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

  def predict(self, X_test, Y_test):
    self.X_test, self.Y_test = X_test, Y_test
    self.m_test = self.X_test.shape[1]
    self.graph.forward_pass(train=False)
    cost, acc = self.graph.compute_cost(train=False)
    print("Test set loss : {0:.2f}".format(cost))
    print("Test set acc : {0:.2f}".format(acc))

class Activation():

  def __init__(self, act):
    self.activation_funcs = {
        'sigmoid' : self.sigmoid,
        'tanh' : self.tanh,
        'relu' : self.relu
    }
    self.activation_funcs_grads = {
        'sigmoid' : self.sigmoid_derivative,
        'tanh' : self.tanh_derivative,
        'relu' : self.relu_derivative
    }
    self.act = act

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def tanh(self, z):
    return np.tanh(z)

  def relu(self, z):
    return np.where(z > 0, z, 0)

  def sigmoid_derivative(self, z):
    g = self.sigmoid(z)
    return g * (1-g)

  def tanh_derivative(self, z):
    g = self.tanh(z)
    return 1 - np.square(g)

  def relu_derivative(self, z):
    return np.where(z >= 0, 1, 0)

  def compute_activation(self, z):
    return self.activation_funcs.get(self.act)(z)

  def compute_activation_gradient(self, z):
    return self.activation_funcs_grads.get(self.act)(z)

class FC():

  def __init__(self, units, activation):
    self.units, self.activation = units, Activation(activation)
    self.Z, self.A = None, None
    self.W, self.b = None, None
    self.dZ, self.dA = None, None
    self.dW, self.db = None, None

class CompGraph():

  def __init__(self):
    self.layers = _model.layers.copy()
    self.A0 = _model.X_train.copy()
    self.cost = []

  def gradient_descent(self):
    for _ in tqdm(range(_model.epochs), desc="Training the model"):
      self.forward_pass()
      self.compute_cost()
      self.backward_pass()
      self.update_weights()
    print("\n", "*" * 10)

  def forward_pass(self, train=True):
    if train:
      for i, layer in enumerate(self.layers):
        layer.Z = (np.dot(layer.W, self.A0) + layer.b if i == 0 else np.dot(layer.W, self.layers[i-1].A) + layer.b)
        layer.A = layer.activation.compute_activation(layer.Z)
    else:
      A = _model.X_test.copy()
      for layer in self.layers:
        Z = np.dot(layer.W, A) + layer.b
        A = layer.activation.compute_activation(Z)
      self.preds = A.copy()

  def compute_cost(self, train=True):
    if train:
      cost = np.sum(_model.Y_train * np.log(self.layers[-1].A) + (1 - _model.Y_train) * np.log(1 - self.layers[-1].A)) / -_model.m_train
      self.cost.append(cost)
    else:
      cost = np.sum(_model.Y_test * np.log(self.preds) + (1 - _model.Y_test) * np.log(1 - self.preds)) / -_model.m_test
      acc = 1 - cost
      return cost, acc

  def backward_pass(self):
    for i, layer in reversed(list(enumerate(self.layers))):
      layer.dZ = (layer.A - _model.Y_train if i+1 == len(self.layers) else layer.dA * layer.activation.compute_activation_gradient(layer.Z))
      layer.dW = np.dot(layer.dZ, self.layers[i-1].A.T) / _model.m_train
      layer.db = np.sum(layer.dZ, keepdims=True, axis=1) / _model.m_train
      self.layers[i-1].dA = np.dot(layer.W.T, layer.dZ)

  def update_weights(self):
    for layer in self.layers:
      layer.W -= _model.alpha * layer.dW
      layer.b -= _model.alpha * layer.db

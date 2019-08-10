import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class DNet():

  def __init__(self, X_train, Y_train):
    np.random.seed(69)
    self.init_data(X_train, Y_train)
    self.init_params()

  def init_data(self, X_train, Y_train):
    self.X_train, self.Y_train = X_train, Y_train
    self.n_train, self.m_train = self.X_train.shape
    self.Z, self.A = [None], [X_train]
    self.dZ, self.dA = [None], [None]
    self.cost, self.acc = [], []

  def init_params(self):
    self.W, self.b, self.layer_units = [None], [None], []
    self.dW, self.db = [None], [None]
    self.layer_units.append(self.n_train)

  def set_arch(self, hidden_units):
    self.set_layer_units(hidden_units)
    self.set_weights()
    self.set_cache()

  def set_layer_units(self, hidden_units):
    self.layer_units += hidden_units
    self.layer_units.append(1)

  def set_weights(self):
    for i, l in enumerate(self.layer_units[1:]):
      W = np.random.randn(l, self.layer_units[i]) * 0.01
      b = np.random.randn(l, 1) * 0.01
      self.W.append(W); self.b.append(b)
      self.dW.append(None); self.db.append(None)

  def set_cache(self):
    for _ in self.layer_units[1:]:
      self.Z.append(None); self.A.append(None)
      self.dZ.append(None); self.dA.append(None)

  def train(self, epochs, lr=0.01):
    self.epochs, self.alpha = epochs, lr
    self.gradient_descent()

  def gradient_descent(self):
    for _ in tqdm(range(self.epochs), desc='Training the model'):
      self.forward_pass()
      self.compute_cost()
      self.backward_pass()
      self.update_weights()
    print("\n", "*" * 10)

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def relu(self, z):
    return np.where(z > 0, z, 0)

  def sigmoid_derivative(self, z):
    g = self.sigmoid(z)
    return g * (1 - g)

  def relu_derivative(self, z):
    return np.where(z >= 0, 1, 0)

  def forward_pass(self, train=True):
    if train:
      for l in range(len(self.layer_units[1:])):
        self.Z[l+1] = np.dot(self.W[l+1], self.A[l]) + self.b[l+1]
        self.A[l+1] = (self.sigmoid(self.Z[l+1]) if l+1 == len(self.layer_units[1:]) else self.relu(self.Z[l+1]))
    else:
      for l in range(len(self.layer_units[1:])):
        Z = np.dot(self.W[l+1], self.A_test) + self.b[l+1]
        self.A_test = (self.sigmoid(Z) if l+1 == len(self.layer_units[1:]) else self.relu(Z))

  def compute_cost(self, train=True):
    if train:
      cost = np.sum(self.Y_train * np.log(self.A[-1]) + (1 - self.Y_train) * np.log(1 - self.A[-1])) / -self.m_train
      self.cost.append(cost)
      self.acc.append(1 - cost)
    else:
      cost = np.sum(self.Y_test * np.log(self.A_test) + (1 - self.Y_test) * np.log(1 - self.A_test)) / -self.m_test
      acc = 1 - cost
      return cost, acc

  def backward_pass(self):
    for l in range(len(self.layer_units[1:]), 0, -1):
      if l == len(self.layer_units[1:]):
        self.dZ[l] = self.A[l] - self.Y_train
      else:
        self.dZ[l] = self.dA[l] * self.relu_derivative(self.Z[l])
      self.dW[l] = np.dot(self.dZ[l], self.A[l-1].T) / self.m_train
      self.db[l] = np.sum(self.dZ[l], keepdims=True, axis=1) / self.m_train
      self.dA[l-1] = np.dot(self.W[l].T, self.dZ[l])

  def update_weights(self):
    for l in range(len(self.layer_units[1:]), 0, -1):
      self.W[l] -= self.alpha * self.dW[l]
      self.b[l] -= self.alpha * self.db[l]

  def plot_losses(self):
    plt.plot(range(self.epochs), self.cost, color='red')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

  def plot_acc(self):
    plt.plot(range(self.epochs), self.acc, color='blue')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

  def predict(self, X_test, Y_test):
    self.X_test, self.Y_test = X_test, Y_test
    self.m_test = self.X_test.shape[1]
    self.A_test = self.X_test
    self.forward_pass(train=False)
    cost, acc = self.compute_cost(train=False)
    print("Loss : {0:.2f}".format(cost))
    print("Accuracy : {0:.2f}".format(acc))

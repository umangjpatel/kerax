import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class DNet(object):

  def __init__(self):
    np.random.seed(42)
    self.cost, self.acc = [], []

  def tanh(self, z):
    return np.tanh(z)

  def tanh_derivative(self, z):
    return 1 - np.square(self.tanh(z))

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def init_params(self, X_train, Y_train, epochs, lr, hidden_units):
    self.X_train, self.Y_train = X_train, Y_train
    self.epochs, self.alpha = epochs, lr
    self.hidden_units = hidden_units
    self.n_train, self.m_train = self.X_train.shape

  def init_weights(self):
    self.W1 = np.random.randn(self.hidden_units, self.n_train) * 0.01
    self.b1 = np.random.randn(self.hidden_units, 1) * 0.01
    self.W2 = np.random.randn(1, self.hidden_units) * 0.01
    self.b2 = np.random.randn(1, 1) * 0.01

  def fit(self, X_train, Y_train, epochs, hidden_units, lr=0.01):
    self.init_params(X_train, Y_train, epochs, lr, hidden_units)
    self.init_weights()
    self.gradient_descent()

  def gradient_descent(self):
    for _ in tqdm(range(self.epochs), desc='Training the model'):
      self.forward_pass()
      self.update_cost()
      self.backward_pass()
      self.update_weights()
    print("*" * 10)

  def forward_pass(self, train=True):
    if train:
      self.Z1 = np.dot(self.W1, self.X_train) + self.b1
      self.A1 = self.tanh(self.Z1)
      self.Z2 = np.dot(self.W2, self.A1) + self.b2
      self.A2 = self.sigmoid(self.Z2)
    else:
      Z1 = np.dot(self.W1, self.X_test) + self.b1
      A1 = self.tanh(Z1)
      Z2 = np.dot(self.W2, A1) + self.b2
      A2 = self.sigmoid(Z2)
      self.preds = A2

  def update_cost(self, train=True):
    if train:
      cost = np.sum(self.Y_train * np.log(self.A2) + (1 - self.Y_train) * np.log(1 - self.A2)) / -self.m_train
      acc = 1 - cost
      self.cost.append(cost)
      self.acc.append(acc)
    else:
      cost = np.sum(self.Y_test * np.log(self.preds) + (1 - self.Y_test) * np.log(1 - self.preds)) / -self.m_test
      acc = 1 - cost
      return cost, acc

  def backward_pass(self):
    dZ2 = self.A2 - self.Y_train
    self.dW2 = np.dot(dZ2, self.A1.T) / self.m_train
    self.db2 = np.sum(dZ2, keepdims=True) / self.m_train
    dA1 = np.dot(self.W2.T, dZ2)
    dZ1 = dA1 * self.tanh_derivative(self.Z1)
    self.dW1 = np.dot(dZ1, self.X_train.T) / self.m_train
    self.db1 = np.sum(dZ1, keepdims=True) / self.m_train

  def update_weights(self):
    self.W2 -= self.alpha * self.dW2
    self.b2 -= self.alpha * self.db2
    self.W1 -= self.alpha * self.dW1
    self.b1 -= self.alpha * self.db1

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
    self.forward_pass(train=False)
    cost, acc = self.update_cost(train=False)
    print("Loss : {0:.2f}".format(cost))
    print("Accuracy : {0:.2f}".format(acc))

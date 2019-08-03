import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class DNet(object):

  def __init__(self):
    self.cost = []

  def sigmoid(self, Z):
    return 1 / (1 + np.exp(-Z))

  def fit(self, X_train, Y_train, epochs = 20, lr = 0.03):
    self.init_params(X_train, Y_train, epochs, lr)
    self.init_weights()
    self.gradient_descent()

  def init_params(self, X_train, Y_train, epochs, lr):
    self.X_train, self.Y_train = X_train, Y_train
    self.n_train, self.m_train = X_train.shape
    self.epochs, self.alpha = epochs, lr

  def init_weights(self):
     self.W = np.zeros((self.n_train, 1))
     self.b = 0

  def gradient_descent(self):
    for _ in tqdm(range(self.epochs), desc="Training the model..."):
      self.forward_pass()
      self.update_cost()
      self.backward_pass()
      self.update_weights()

  def forward_pass(self, train=True):
    if train:
      Z = np.dot(self.W.T, self.X_train) + self.b
      self.A = self.sigmoid(Z)
    else:
      Z = np.dot(self.W.T, self.X_test) + self.b
      self.preds = self.sigmoid(Z)

  def update_cost(self, train=True):
    if train:
      cost = np.sum(self.Y_train * np.log(self.A) + (1 - self.Y_train) * np.log(1 - self.A)) / -self.m_train
      self.cost.append(cost)
    else:
      cost = np.sum(self.Y_test * np.log(self.preds) + (1 - self.Y_test) * np.log(1 - self.preds)) / -self.m_test
      return cost

  def backward_pass(self):
    dZ = self.A - self.Y_train
    self.dW = np.dot(self.X_train, dZ.T) / self.m_train
    self.db = np.sum(dZ) / self.m_train

  def update_weights(self):
    self.W -= self.alpha * self.dW
    self.b -= self.alpha * self.db

  def plot_losses(self):
    plt.plot(range(self.epochs), self.cost, color='red')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

  def plot_acc(self):
    plt.plot(range(self.epochs), 1 - np.array(self.cost), color='blue')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

  def predict(self, X_test, Y_test):
    self.X_test, self.Y_test = X_test, Y_test
    self.n_test, self.m_test = self.X_test.shape
    self.forward_pass(train=False)
    pred_loss = self.update_cost(train=False)
    print("Prediction Accuracy : {0:.2f}".format(1 - pred_loss))

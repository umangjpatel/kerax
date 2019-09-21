import jax.numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit
from tqdm import tqdm


@jit
def compute_activation(x):
    return 1 / (1 + np.exp(-x))


@jit
def compute_predictions(weights, inputs):
    W, b = weights.get('W'), weights.get('b')
    Z = np.dot(inputs, W) + b
    A = compute_activation(Z)
    return A


@jit
def compute_cost(weights, inputs, targets):
    predictions = compute_predictions(weights, inputs)
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return loss


class DNet:

    def fit(self, x_train, y_train, epochs, lr=1e-3):
        self.x_train, self.y_train = x_train, y_train
        self.epochs, self.alpha = epochs, lr
        self.gradient_descent()

    def init_weights(self):
        self.m, self.nx = self.x_train.shape
        W = np.zeros((self.nx, 1))
        b = np.zeros(1)
        self.weights = {'W': W, 'b': b}

    def gradient_descent(self):
        self.cost = []
        grad_fn = jit(grad(compute_cost))
        self.init_weights()

        for _ in tqdm(range(self.epochs), desc="Training the model"):
            self.cost.append(compute_cost(self.weights, self.x_train, self.y_train))
            grads = grad_fn(self.weights, self.x_train, self.y_train)
            self.weights['W'] -= self.alpha * grads['W']
            self.weights['b'] -= self.alpha * grads['b']

    def plot_losses(self):
        plt.plot(range(self.epochs), self.cost, color='red')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def evaluate(self, x_test, y_test):
        predictions = compute_predictions(self.weights, x_test)
        prediction_labels = np.where(predictions >= 0.7, 1, 0)
        accuracy = np.mean(prediction_labels == y_test)
        print("Accuracy : {:0.2f}".format(accuracy))

    def predict(self, inputs):
        return compute_predictions(self.weights, inputs)

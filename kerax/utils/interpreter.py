from typing import Iterable, List

import matplotlib.pyplot as plt


class Interpreter:
    """
    Interpreter class which is used for analysis of the training results.
    """

    def __init__(self, **config):
        """
        Initializes the class.
        :param config: a dictionary consisting of the training results.
        """
        self._config = config

    def plot_losses(self):
        """
        Plots the loss curves (both training and validation) in Matplotlib.
        :return: a Matplotlib chart.
        """
        epochs: Iterable[int] = range(1, self._config.get("epochs") + 1)
        train_losses: List[float] = self._config.get("metrics").get("loss_per_epoch").get("train")
        val_losses: List[float] = self._config.get("metrics").get("loss_per_epoch").get("valid")
        plt.plot(epochs, train_losses, color="red", label="Training")
        plt.plot(epochs, val_losses, color="green", label="Validation")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        """
        Plots the accuracy curves (both training and validation) in Matplotlib.
        :return: a Matplotlib chart.
        """
        epochs: Iterable[int] = range(1, self._config.get("epochs") + 1)
        if "binary_accuracy" in self._config.get("metrics").keys():
            train_acc: List[float] = self._config.get("metrics").get("binary_accuracy_per_epoch").get("train")
            val_acc: List[float] = self._config.get("metrics").get("binary_accuracy_per_epoch").get("valid")
        else:
            train_acc: List[float] = self._config.get("metrics").get("accuracy_per_epoch").get("train")
            val_acc: List[float] = self._config.get("metrics").get("accuracy_per_epoch").get("valid")
        plt.plot(epochs, train_acc, color="red", label="Training")
        plt.plot(epochs, val_acc, color="green", label="Validation")
        plt.title("Accuracy Curve")
        plt.ylim([0.0, 1.05])
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

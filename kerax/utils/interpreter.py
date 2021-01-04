from typing import Iterable, List

import matplotlib.pyplot as plt


class Interpreter:

    def __init__(self, **config):
        self._config = config

    def plot_losses(self):
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

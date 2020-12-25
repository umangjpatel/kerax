from typing import Iterable, List


class Interpreter:

    def __init__(self, **config):
        self._config = config

    def plot_losses(self):
        import matplotlib.pyplot as plt
        epochs: Iterable[int] = range(self._config.get("epochs"))
        train_losses: List[float] = self._config.get("metrics").get("train_loss")
        val_losses: List[float] = self._config.get("metrics").get("val_loss")
        plt.plot(epochs, train_losses, color="red", label="Training")
        plt.plot(epochs, val_losses, color="green", label="Validation")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

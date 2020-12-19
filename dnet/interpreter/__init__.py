class Interpreter:

    def __init__(self, **config):
        self._config = config

    def plot_losses(self):
        import matplotlib.pyplot as plt
        plt.plot(range(self._config.get("epochs")), self._config.get("losses"), color="red")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

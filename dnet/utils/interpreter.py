class Interpreter:

    def __init__(self, **config):
        self._config = config

    def plot_losses(self):
        losses = self._config.get("metrics").get("losses")
        import matplotlib.pyplot as plt
        plt.plot(range(self._config.get("epochs")), losses, color="red")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

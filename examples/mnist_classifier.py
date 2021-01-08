from kerax.datasets import mnist
from kerax.layers import Flatten, Dense, Relu, LogSoftmax
from kerax.losses import CCELoss
from kerax.metrics import accuracy
from kerax.models import Sequential
from kerax.optimizers import RMSProp

data = mnist.load_dataset(batch_size=1024)
model = Sequential([Flatten, Dense(100), Relu, Dense(10), LogSoftmax])
model.compile(loss=CCELoss, optimizer=RMSProp(step_size=0.001), metrics=[accuracy])
model.fit(data, epochs=10)
model.save("tfds_mnist_v1")
interp = model.get_interpretation()
interp.plot_losses()

from kerax.datasets import binary_tiny_mnist
from kerax.layers import Dense, Relu, Sigmoid
from kerax.losses import BCELoss
from kerax.metrics import binary_accuracy
from kerax.models import Sequential
from kerax.optimizers import SGD

data = binary_tiny_mnist.load_dataset(batch_size=200)
model = Sequential([Dense(100), Relu, Dense(1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.003), metrics=[binary_accuracy])
model.fit(data=data, epochs=10)
model.save(file_name="tiny_mnist_binary_classifier_v1")
interp = model.get_interpretation()
interp.plot_losses()

new_model = Sequential()
new_model.load(file_name="tiny_mnist_binary_classifier_v1")
# model already compiled when loaded from serialized file
new_model.fit(data=data, epochs=50)
new_model.save(file_name="tiny_mnist_binary_classifier_v2")
interp = new_model.get_interpretation()
interp.plot_losses()

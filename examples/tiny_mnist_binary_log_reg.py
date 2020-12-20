from dnet.datasets import binary_tiny_mnist
from dnet.models import Sequential
from dnet.layers import Dense, Sigmoid
from dnet.optimizers import SGD
from dnet.losses import BCELoss

train_images, train_labels = binary_tiny_mnist.load_data()

model = Sequential([Dense(1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01))
model.fit(inputs=train_images, targets=train_labels, epochs=10)

interp = model.get_interpretation()
interp.plot_losses()

# TODO : 1) Study Course 1 and Course 2 notes
# TODO : 2) Plan out library functionality

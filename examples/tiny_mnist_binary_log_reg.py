from dnet.datasets import binary_tiny_mnist
from dnet.layers import Dense, Sigmoid, Relu
from dnet.losses import BCELoss
from dnet.models import Module
from dnet.optimizers import SGD

train_images, train_labels = binary_tiny_mnist.load_data()

model = Module([Dense(10), Relu, Dense(1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01))
model.fit(inputs=train_images, targets=train_labels, epochs=10)

model.save(fname="log_reg")
# model.load(file_name="log_reg")

# interp = model.get_interpretation()
# interp.plot_losses()

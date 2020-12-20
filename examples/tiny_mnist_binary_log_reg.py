from dnet.datasets import binary_tiny_mnist
from dnet.models import Module
from dnet.layers import Dense, Sigmoid
from dnet.optimizers import SGD
from dnet.losses import BCELoss

train_images, train_labels = binary_tiny_mnist.load_data()

model = Module([Dense(10), Dense(1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01))
model.fit(inputs=train_images, targets=train_labels, epochs=10)

model.save(file_name="log_reg")

# interp = model.get_interpretation()
# interp.plot_losses()

from dnet.datasets import binary_tiny_mnist
from dnet.layers import Dense, Sigmoid, Relu
from dnet.losses import BCELoss
from dnet.models import Module
from dnet.optimizers import SGD

(train_images, train_labels), (val_images, val_labels) = binary_tiny_mnist.load_data()

model = Module([Dense(10), Relu, Dense(1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01))
model.fit(inputs=train_images, targets=train_labels,
          validation_data=(val_images, val_labels), epochs=10)
model.save(file_name="log_reg")
interp = model.get_interpretation()
interp.plot_losses()

model = Module()
model.load(file_name="log_reg")
# model already compiled when loaded from serialized file
model.fit(inputs=train_images, targets=train_labels,
          validation_data=(val_images, val_labels), epochs=100)
model.save(file_name="log_reg_v2")
interp = model.get_interpretation()
interp.plot_losses()

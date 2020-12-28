from dnet.datasets import binary_tiny_mnist
from dnet.layers import Dense, Relu, Sigmoid
from dnet.losses import BCELoss
from dnet.models import Module
from dnet.optimizers import SGD

data = binary_tiny_mnist.load_data()

model = Module([Dense(100), Relu, Dense(1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01))
model.fit(data=data, epochs=10)
model.save(file_name="log_reg")
interp = model.get_interpretation()
interp.plot_losses()

model = Module()
model.load(file_name="log_reg")
# model already compiled when loaded from serialized file
model.fit(data=data, epochs=100)
model.save(file_name="log_reg_v2")
interp = model.get_interpretation()
interp.plot_losses()

from dnet.datasets import binary_tiny_mnist
from dnet.models import Sequential
from dnet.layers import Dense, Sigmoid
from dnet.optimizers import SGD
from dnet.losses import BCELoss

train_images, train_labels = binary_tiny_mnist.load_data()

model = Sequential([Dense(out_dim=1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01))
model.fit(inputs=train_images,
          targets=train_labels,
          epochs=10)
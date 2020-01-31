from dnet import datasets
from dnet.layers import FC
from dnet.models import Sequential

(x_train, y_train), (x_val, y_val) = datasets.mnist_tiny(flatten=True)

model = Sequential()
model.add(FC(units=500, activation="relu"))
model.add(FC(units=50, activation="relu"))
model.add(FC(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="momentum", lr=1e-03, bs=128)
model.fit(inputs=x_train, targets=y_train, epochs=20, validation_data=(x_val, y_val))

model.plot_losses()
model.plot_accuracy()

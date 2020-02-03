from dnet import datasets
from dnet.layers import FC
from dnet.models import Sequential

(x_train, y_train), (x_val, y_val) = datasets.mnist(flatten=True, one_hot_encoding=True)

model = Sequential()
model.add(FC(units=500, activation="relu"))
model.add(FC(units=50, activation="relu"))
model.add(FC(units=10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", lr=1e-03, bs=512)
model.fit(inputs=x_train, targets=y_train, epochs=20, validation_data=(x_val, y_val))

model.plot_losses()
model.plot_accuracy()

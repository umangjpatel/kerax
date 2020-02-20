from dnet import datasets
from dnet.layers import Conv2D, MaxPool2D, Flatten, FC
from dnet.models import Sequential

(x_train, y_train), (x_val, y_val) = datasets.tiny_mnist(flatten=False, one_hot_encoding=True)

model = Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(5, 5), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    FC(units=120, activation="relu"),
    FC(units=84, activation="relu"),
    FC(units=10, activation="softmax")
])
# model.add(Conv2D(filters=6, kernel_size=(5, 5), activation="relu"))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(FC(units=120, activation="relu"))
# model.add(FC(units=84, activation="relu"))
# model.add(FC(units=10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", lr=1e-03, bs=512)
model.fit(inputs=x_train, targets=y_train, epochs=10, validation_data=(x_val, y_val))

model.plot_losses()
model.plot_accuracy()

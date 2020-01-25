from pathlib import Path

import jax.numpy as tensor
import pandas as pd

from dnet.layers import Conv2D, MaxPool2D, Flatten, FC
from dnet.models import Sequential

current_path = Path("..")
train_path = current_path / "datasets" / "mnist_small" / "mnist_train_small.csv"
test_path = current_path / "datasets" / "mnist_small" / "mnist_test.csv"

training_data = pd.read_csv(train_path, header=None)

y_train = tensor.asarray(pd.get_dummies(training_data[0]))  # One-hot encoding output shape : (m, nc)
x_train = tensor.asarray(training_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)) / 255.0  # shape = (m, h, w, c)

testing_data = pd.read_csv(test_path, header=None)

y_val = tensor.asarray(pd.get_dummies(testing_data[0]))  # One-hot encoding output shape : (m, nc)
x_val = tensor.asarray(testing_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)) / 255.0  # shape = (m, h, w, c)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(FC(units=120, activation="relu"))
model.add(FC(units=84, activation="relu"))
model.add(FC(units=10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", lr=1e-03, bs=512)
model.fit(inputs=x_train, targets=y_train, epochs=10, validation_data=(x_val, y_val))

model.plot_losses()
model.plot_accuracy()

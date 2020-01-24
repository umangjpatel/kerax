from pathlib import Path

import jax.numpy as tensor
import pandas as pd

from dnet.layers import Conv2D, MaxPool2D, Flatten, FC
from dnet.models import Sequential

current_path = Path("..")
train_path = current_path / "datasets" / "mnist_small" / "mnist_train_small.csv"
test_path = current_path / "datasets" / "mnist_small" / "mnist_test.csv"

training_data = pd.read_csv(train_path, header=None)
training_data = training_data.loc[training_data[0].isin([0, 1])]  # Classification between digits 0 and 1

y_train = tensor.asarray(training_data[0].values.reshape(-1, 1))  # shape : (m, 1)
x_train = tensor.asarray(training_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)) / 255.0  # shape = (m, h, w, c)

testing_data = pd.read_csv(test_path, header=None)
testing_data = testing_data.loc[testing_data[0].isin([0, 1])]

y_val = tensor.asarray(testing_data[0].values.reshape(-1, 1))  # shape : (m, 1)
x_val = tensor.asarray(testing_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)) / 255.0  # shape = (m, h, w, c)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(FC(units=120, activation="relu"))
model.add(FC(units=84, activation="relu"))
model.add(FC(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="sgd", lr=1e-02, bs=x_train.shape[0])
model.fit(inputs=x_train, targets=y_train, epochs=5, validation_data=(x_val, y_val))

model.plot_losses()
model.plot_accuracy()

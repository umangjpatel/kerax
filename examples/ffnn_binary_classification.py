from pathlib import Path

import jax.numpy as tensor
import pandas as pd

from dnet.layers import FC
from dnet.models import Sequential

current_path = Path("..")
train_path = current_path / "datasets" / "mnist_small" / "mnist_train_small.csv"
test_path = current_path / "datasets" / "mnist_small" / "mnist_test.csv"

training_data = pd.read_csv(train_path, header=None)
training_data = training_data.loc[training_data[0].isin([0, 1])]  # Classification between digits 0 and 1

y_train = tensor.asarray(training_data[0].values.reshape(-1, 1))  # shape : (m, 1)
x_train = tensor.asarray(training_data.iloc[:, 1:].values) / 255.0  # shape = (m, n)

testing_data = pd.read_csv(test_path, header=None)
testing_data = testing_data.loc[testing_data[0].isin([0, 1])]

y_val = tensor.asarray(testing_data[0].values.reshape(-1, 1))  # shape : (m, 1)
x_val = tensor.asarray(testing_data.iloc[:, 1:].values) / 255.0  # shape = (m, n)

model = Sequential()
model.add(FC(units=500, activation="relu"))
model.add(FC(units=50, activation="relu"))
model.add(FC(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="momentum", lr=1e-03, bs=128)
model.fit(inputs=x_train, targets=y_train, epochs=20, validation_data=(x_val, y_val))

model.plot_losses()
model.plot_accuracy()

from pathlib import Path

import jax.numpy as tensor
import pandas as pd

from dnet.layers import FC
from dnet.nn import Sequential

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
model.add(FC(units=500, activation="mish", weight_scheme="glorot_uniform", input_dim=x_train.shape[-1]))
model.add(FC(units=10, activation="mish", weight_scheme="glorot_uniform"))
model.add(FC(units=1, activation="sigmoid", weight_scheme="glorot_uniform"))
model.compile(loss="binary_crossentropy", optimizer="sgd", lr=1e-02, bs=x_train.shape[0])
model.fit(inputs=x_train, targets=y_train, epochs=20, validation_data=(x_val, y_val))

model.plot_losses()
model.plot_accuracy()

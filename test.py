from pathlib import Path

import jax.numpy as tensor
import pandas as pd

from dnet.layers import FC
from dnet.nn import Sequential

dataset_path = Path("datasets")
train_path = dataset_path / "mnist_small" / "mnist_train_small.csv"
test_path = dataset_path / "mnist_small" / "mnist_test.csv"

training_data = pd.read_csv(train_path, header=None)
training_data = training_data.loc[training_data[0].isin([0, 1])]

y_train = tensor.array(training_data[0].values.reshape(1, -11))  # shape : (1. m)
x_train = tensor.array(training_data.iloc[:, 1:].values.T) / 255.0  # shape = (n, m)

testing_data = pd.read_csv(test_path, header=None)
testing_data = testing_data.loc[testing_data[0].isin([0, 1])]

y_val = tensor.array(testing_data[0].values.reshape(1, -1))  # shape : (1, m)
x_val = tensor.array(testing_data.iloc[:, 1:].values.T) / 255.0  # shape = (n, m)

model = Sequential()
model.add(FC(units=500, activation="sigmoid", input_dim=784))
model.add(FC(units=10, activation="sigmoid"))
model.add(FC(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="sgd", lr=1e-02)
model.fit(inputs=x_train, targets=y_train, epochs=50, validation_data=(x_val, y_val))

model.plot_losses()

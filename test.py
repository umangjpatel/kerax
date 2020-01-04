import torch
import pandas as pd
from pathlib import Path

from dnet.nn import Sequential
from dnet.layers import FC

dataset_path = Path("datasets")
train_path = dataset_path / "mnist_small" / "mnist_train_small.csv"
test_path = dataset_path / "mnist_small" / "mnist_test.csv"

training_data = pd.read_csv(train_path, header=None)
training_data = training_data.loc[training_data[0].isin([0, 1])]

y_train = torch.from_numpy(training_data[0].values.reshape(1, -11))  # shape : (1. m)
x_train = torch.from_numpy(training_data.iloc[:, 1:].values.T) / 255.0  # shape = (n, m)

testing_data = pd.read_csv(test_path, header=None)
testing_data = testing_data.loc[testing_data[0].isin([0, 1])]

y_val = torch.from_numpy(testing_data[0].values.reshape(1, -1))  # shape : (1, m)
x_val = torch.from_numpy(testing_data.iloc[:, 1:].values.T) / 255.0  # shape = (n, m)

net = Sequential()
net.add(FC(units=500, activation="mish"))
net.add(FC(units=10, activation="relu"))
net.add(FC(units=1, activation="sigmoid"))
net.compile(optimizer="sgd", loss="binary_crossentropy", lr=1e-02)
net.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

net.plot_losses()

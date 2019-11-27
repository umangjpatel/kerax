import os

import jax.numpy as tensor
import pandas as pd

from dnet.layers import FC
from dnet.nn import Sequential

mnist_dataset_path = os.path.join("datasets", "mnist")
mnist_train_path = os.path.join(mnist_dataset_path, "mnist_train_small.csv")
mnist_test_path = os.path.join(mnist_dataset_path, "mnist_test.csv")

training_data = pd.read_csv(mnist_train_path, header=None)
training_data = training_data.loc[training_data[0].isin([0, 1])]

y_train = tensor.array(training_data[0].values.reshape(1, -1))  # shape : (1, m)
x_train = tensor.array(training_data.iloc[:, 1:].values.T)  # shape = (n, m)

testing_data = pd.read_csv(mnist_test_path, header=None)
testing_data = testing_data.loc[testing_data[0].isin([0, 1])]

y_val = tensor.array(testing_data[0].values.reshape(1, -1))  # shape : (1, m)
x_val = tensor.array(testing_data.iloc[:, 1:].values.T)  # shape = (n, m)

model = Sequential()
model.add(FC(units=500, activation="relu"))
model.add(FC(units=50, activation="relu"))
model.add(FC(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", epochs=20, lr=0.01)
model.fit(x_train, y_train)

model.plot_losses()
model.plot_accuracy()

train_acc_score = model.evaluate(x_train, y_train)
print("Training accuracy : {0:.6f}".format(train_acc_score))
val_acc_score = model.evaluate(x_val, y_val)
print("Validation accuracy : {0:.6f}".format(val_acc_score))

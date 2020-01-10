![dnet library logo](assets/logo.png "DNet library")
# DNet
Neural Network Library written in Python and built on top of JAX, an open-source high-performance deep learning library.

## Packages used
* [JAX](https://github.com/google/jax) for automatic differentiation.
* [Mypy](https://github.com/python/mypy) for static typing Python3 code.
* [Matplotlib](https://github.com/matplotlib/matplotlib) for plotting.
* [Pandas](https://github.com/pandas-dev/pandas) for data analysis / manipulation.
* [tqdm](https://github.com/tqdm/tqdm) for displaying progress bar.

## Features
* Enables high-performance machine learning research.
* Easy to use with high-level Keras-like APIs.
* Runs seamlessly on GPU and even TPU!.

## Getting started

Here's the Sequential model :
```python3
model = Sequential()
```
Add the fully-connected layers / densely-connected layers :
```python3
model.add(FC(units=500, activation="mish"))
model.add(FC(units=10, activation="relu"))
model.add(FC(units=1, activation="sigmoid"))
```
Compile the model with the hyperparameters :
```python3
model.compile(loss="binary_crossentropy", optimizer="sgd", lr=1e-02)
```
Train the model (with validation data) :
```python3
model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val)
```
Plot the loss curves :
```python3
model.plot_losses()
```

## Toy Example

### Code
```python3
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

y_train = tensor.array(training_data[0].values.reshape(-1, 1))  # shape : (m, 1)
x_train = tensor.array(training_data.iloc[:, 1:].values) / 255.0  # shape = (m, n)

testing_data = pd.read_csv(test_path, header=None)
testing_data = testing_data.loc[testing_data[0].isin([0, 1])]

y_val = tensor.array(testing_data[0].values.reshape(-1, 1))  # shape : (m, 1)
x_val = tensor.array(testing_data.iloc[:, 1:].values) / 255.0  # shape = (m, n)

model = Sequential()
model.add(FC(units=500, activation="mish", input_dim=784))
model.add(FC(units=10, activation="relu"))
model.add(FC(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="sgd", lr=1e-02)
model.fit(inputs=x_train, targets=y_train, epochs=50, validation_data=(x_val, y_val))

model.plot_losses()
```

### Outputs
```
/usr/local/bin/python3.7 DNet/test.py
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/jax/lib/xla_bridge.py:120: UserWarning: No GPU/TPU found, falling back to CPU.
  warnings.warn('No GPU/TPU found, falling back to CPU.')
Training your model: 100%|██████████| 50/50 [00:02<00:00, 17.21it/s]
```
![Toy example loss curves](assets/toy_example_loss_curves.png "Loss Curves")
```
Process finished with exit code 0
```


## Roadmap
Check the [roadmap](https://github.com/umangjpatel/dnet/projects/2) of this project. This will show you the progress in the development of this library.

## Developers
* [Umang Patel](https://github.com/umangjpatel)

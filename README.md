![dnet library logo](assets/logo.png "DNet library")
# DNet
Neural Network Library written in Python and built on top of JAX, an open-source high-performance machine learning library.

## Packages used
* [JAX](https://github.com/google/jax) for automatic differentiation.
* [Mypy](https://github.com/python/mypy) for static typing Python3 code.
* [Matplotlib](https://github.com/matplotlib/matplotlib) for plotting.
* [Pandas](https://github.com/pandas-dev/pandas) for data analysis / manipulation.

## Features
* Enables high-performance machine learning research.
* Easy to use with high-level APIs.
* Runs seamlessly on GPUs and even TPUs.

## Getting started

Here's the Sequential model :
```python3
model = Sequential()
```
Add the fully-connected layers / densely-connected layers :

```python3
model.add(FC(units=500, activation="relu"))
model.add(FC(units=50, activation="relu"))
model.add(FC(units=1, activation="sigmoid"))
```
Compile the model with the hyperparameters :
```python3
model.compile(loss="binary_crossentropy", epochs=20, lr=0.01)
```
Train the model :
```python3
model.fit(x_train, y_train)
```
Plot the training curves :
```python3
model.plot_losses()
model.plot_accuracy()
```
Compute accuracy :
```
train_acc_score = model.evaluate(x_train, y_train)
print("Training accuracy : {0:.6f}".format(train_acc_score))
val_acc_score = model.evaluate(x_val, y_val)
print("Validation accuracy : {0:.6f}".format(val_acc_score))
```

## Roadmap
Check the [roadmap](https://github.com/umangjpatel/dnet/projects/2) of this project. This will give you the idea of the progress in the development of this library.

## Developers
* [Umang Patel](https://github.com/umangjpatel)

# DNet
Neural Network Library built from scratch... :D

## Modules / Packages required
* <a href="http://github.com/google/jax">JAX</a> , for automatic differentiation
* <a href="https://github.com/pandas-dev/pandas">Pandas</a> , for manipulating the data
* <a href="https://github.com/matplotlib/matplotlib">Matplotlib</a> , for data visualization
* <a href="https://github.com/tqdm/tqdm">tqdm</a> , for displaying the model training progress

## Features
* Fast, fluid and flexible deep learning framework
* High-level APIs for faster experimentation
* Works on CPU/GPU and even TPU (Thanks to Google)


## Usage
Download the repository and start using the library as follows :
```python
from dnet import *
...
...
...

# Create model object
model = DNet()

# Define the model architecture
model.add(FC(units=500, activation='relu', keep_prob=0.5))
model.add(FC(units=50, activation='relu', keep_prob=0.5))
model.add(FC(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', epochs=20, lr=3e-2)

# Train the model
model.fit(X_train, Y_train)

# Plot the training loss curve
model.plot_losses()

# Evaluate the model on validation data
val_acc_score = model.evaluate(X_val, Y_val, threshold=1.0)
print("Validation accuracy : {0:.6}".format(val_acc_score))

# Make predictions on test data
model.predict(X_test)
```

## Upcoming features

Please check the link : [Roadmap](https://github.com/umangjpatel/DNet/projects/2) to track the progress of the project so far.

## Notes
If any bugs/errors, don't hesitate to raise issues for the project. Your honest review can significantly impact the output of this project.

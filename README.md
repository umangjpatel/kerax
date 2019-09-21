# DNet
Neural Network Library built from scratch... :D

## Modules / Packages required
* NumPy, for vectorization
* Pandas, for manipulating the data
* Matplotlib, for data visualization
* tqdm, for displaying the model training progress
* JAX, for automatic differentiation

## Usage
Download the dnet.py script and start using the library as follows :
```python
from dnet import *
...
...
...

# Create model object
model = DNet()

# Train the model
model.fit(X_train, Y_train, epochs=100, lr=0.01)

# Plot the training loss curve
model.plot_losses()

# Evaluate the model on validation data
model.evaluate(X_val, Y_val)

# Make predictions on test data
model.predict(X_test
```

## Roadmap for the project
Please check the link : [Roadmap](https://github.com/umangjpatel/DNet/projects/2)

## Notes
If any bugs/errors, don't hesitate to raise issues for the project. Your honest review can significantly impact the output of this project.

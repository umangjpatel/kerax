# DNet
Neural Network Library built from scratch... :D

## Modules / Packages required
* NumPy, for the computation engine
* Pandas, for manipulating the data
* Matplotlib, for data visualization
* tqdm, for displaying the model training progress

## Usage
Download the dnet.py script and start using the library as follows :
```python
import dnet
...
...
...

#Define your model
model = dnet.DNet(X_train, Y_train)

#Define units in the hidden layers
model.set_arch(hidden_units=[100,200,...])

#Train your model
model.train(epochs, lr)

#Visualize the loss curve and accuracy curve during training process
model.plot_losses()
model.plot_acc()

#Make predictions on unseen data using your model
model.predict(X_test, Y_test)
```

## Roadmap for the project
Please check the link : [Roadmap](https://github.com/umangjpatel/DNet/projects/2)

## Notes
If any bugs/errors, don't hesitate to raise issues for the project. Your honest review can significantly impact the output of this project.

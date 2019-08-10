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
from dnet import *
...
...
...

#Create model object
model = DNet()

#Define neural network architecture
model.add(FC(units = 500, activation = 'relu'))
model.add(FC(units = 50, activation = 'relu'))
model.add(FC(units = 1, activation ='sigmoid'))

#Compile the model with epochs and learning rate
model.compile(epochs = 50, lr = 0.01)

#Train the model
model.fit(train_features, train_labels)

#Plot the Loss Curve during training
model.plot_losses()

#Test model on unseen data
model.predict(test_features, test_labels)
```

## Roadmap for the project
Please check the link : [Roadmap](https://github.com/umangjpatel/DNet/projects/2)

## Notes
If any bugs/errors, don't hesitate to raise issues for the project. Your honest review can significantly impact the output of this project.

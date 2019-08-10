import pandas as pd
from dnet import *

train_data = pd.read_csv("datasets/mnist/mnist_train_small.csv", header=None)
test_data = pd.read_csv("datasets/mnist/mnist_test.csv", header=None)

options = [0, 1]
train_dataset = train_data.loc[train_data[0].isin(options)]
test_dataset = test_data.loc[test_data[0].isin(options)]

train_features = train_dataset.iloc[:, 1:].values.T / 255
train_labels = train_dataset.iloc[:, 0].values
train_labels = train_labels.reshape(1, train_labels.shape[0])

test_features = test_dataset.iloc[:, 1:].values.T / 255
test_labels = test_dataset.iloc[:, 0].values
test_labels = test_labels.reshape(1, test_labels.shape[0])

print("Training data shapes : ", train_features.shape, train_labels.shape)
print("Testing data shape : ", test_features.shape, test_labels.shape)

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

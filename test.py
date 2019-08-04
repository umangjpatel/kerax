import pandas as pd
import dnet

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

model = dnet.DNet()
model.fit(train_features, train_labels, epochs=100, hidden_units=200, lr=0.03) #Train the model

model.plot_losses() #Plot the Loss Curve during training
model.plot_acc() #Plot the Accuracy Curve during training

model.predict(test_features, test_labels) #Test model on unseen data

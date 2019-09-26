import warnings

import pandas as pd

from dnet import *

# Avoid CPU execution warning (for tqdm)
warnings.simplefilter("ignore", UserWarning)

# Loading the training dataset
trainer = pd.read_csv("datasets/mnist/mnist_train_small.csv", header=None)
train_data = trainer.loc[(trainer[0] == 1) | (trainer[0] == 0)]
train_features, train_labels = train_data.iloc[:, 1:].values, train_data[1].values
print("Training data -> features shape : {}, labels shape : {}".format(train_features.shape, train_labels.shape))

# Loading the training dataset
tester = pd.read_csv("datasets/mnist/mnist_test.csv", header=None)
test_data = tester.loc[(tester[0] == 1) | (tester[0] == 0)]
test_features, test_labels = test_data.iloc[:, 1:].values, test_data[1].values
print("Testing data -> features shape : {}, labels shape : {}".format(test_features.shape, test_labels.shape))

# Create model object
model = DNet()

# Define the model architecture
model.add(FC(units=500, activation='mish'))
model.add(FC(units=50, activation='mish'))
model.add(FC(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', epochs=10, lr=0.003)

# Train the model
model.fit(train_features, train_labels)

# Plot the training loss curve
model.plot_losses()

# Evaluate the model on validation data
model.evaluate(test_features, test_labels)

# Make predictions on unseen data
# model.predict(test_features)

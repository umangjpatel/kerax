import os

import pandas as pd

from dnet import *


def standardization(x):
    return (x - np.mean(x)) / np.std(x)


# Loading the paths to the data
data_path = os.path.join("datasets", "dummy_regression")
train_data_path = os.path.join(data_path, 'regression.csv')

# Loading the training dataset
trainer = pd.read_csv(train_data_path)
train_features, train_targets = trainer['x'].values.reshape(-1, 1), trainer['y'].values.reshape(-1, 1)

train_features, train_targets = standardization(train_features), standardization(train_targets)

print("Training data -> features shape : {}, targets shape : {}".format(train_features.shape, train_targets.shape))

# Create model object
model = DNet()

# Define the model architecture
model.add(FC(units=1, activation='linear'))
model.add(FC(units=1, activation='linear'))

# Compile the model
model.compile(loss='mse', epochs=200, lr=1e-02)

# Train the model
model.fit(train_features, train_targets)

# Plot the training loss curve
model.plot_losses()

# Evaluate the model on validation data
val_loss = model.evaluate(train_features[:10], train_targets[:10])
print("Validation loss : {0:.6}".format(val_loss))

# Make predictions on unseen data
# print(model.predict(x_test))
